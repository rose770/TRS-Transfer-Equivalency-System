"""
FastAPI Backend — Course Equivalency System
============================================
Tables used:
  - taibah_courses          <- Taibah University courses (admin imported)
  - taibah_course_content   <- Sections for Taibah courses
  - external_courses        <- Student uploaded course specs
  - external_course_content <- Sections for external courses
  - student_external_courses<- Links students to external courses
  - similarity_cache        <- Cached comparison results
  - students                <- Student info
  - student_transcript_courses <- Courses from student transcript
  - equivalency_form        <- One per equivalency request
  - course_comparison_results  <- Per-course comparison results

Run:
    uvicorn main:app --host 127.0.0.1 --port 8000 --reload
    frontend:
    cd D:\GP\OCR_trial\equivalency-frontend
    npx vite --port 3000


FIXES APPLIED:
  1. insert_transcript_to_db now deduplicates.
  2. run_equivalency_comparison deduplicates student_courses.
  3. weighted_similarity returns (final, content_sim, desc_sim) tuple.
  4. Content assembly always stores richest possible content_text.
  5. Domain boost capped correctly and applied consistently.
  6. validate_uuid() added to all endpoints.
  7. s_institution normalized before cache lookup/save.
  8. Grade and credit rejections now store similarity_percentage=None
     and Arabic rejection reason instead of 0 and English text.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import re
import json
import uuid
import numpy as np
from pathlib import Path
from openai import OpenAI
from supabase import create_client, Client
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
import subprocess

load_dotenv()

# ── Init ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Course Equivalency API", version="1.0.0")

@app.on_event("startup")
async def reset_stuck_forms():
    """Reset any forms stuck in 'processing' from a previous crashed/reloaded run."""
    try:
        result = supabase.table("equivalency_form") \
            .update({"status": "failed"}) \
            .eq("status", "processing") \
            .execute()
        if result.data:
            print(f"  Reset {len(result.data)} stuck 'processing' form(s) to 'failed'")
    except Exception as e:
        print(f"  Could not reset stuck forms: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
OPENAI_KEY   = os.environ.get("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client    = OpenAI(api_key=OPENAI_KEY)

import httpx
supabase.postgrest.session = httpx.Client(http2=False)

SIMILARITY_THRESHOLD = 0.70
MIN_GRADE            = 70.0

DOMAIN_KEYWORDS = [
    {"programming", "program", "coding", "code", "software", "introduction to programming", "fundamentals of programming"},
    {"network", "networks", "networking", "protocol", "internet"},
    {"computation", "automata", "turing", "computability", "formal", "theory"},
    {"intelligence", "intelligent", "ai", "artificial", "machine learning"},
    {"database", "databases", "sql", "data management"},
    {"algorithm", "algorithms", "data structure", "data structures"},
    {"operating system", "os", "kernel", "process", "scheduler"},
    {"security", "cryptography", "cybersecurity", "encryption"},
    {"architecture", "organization", "assembly", "hardware", "digital", "logic", "digital logic", "circuit", "boolean"},
    {"mathematics", "calculus", "algebra", "discrete"},
    {"statistics", "probability"},
]

LETTER_TO_NUMERIC = {
    "أ+": 95, "أ": 90, "ب+": 85, "ب": 80,
    "ج+": 75, "ج": 70, "د+": 65, "د": 60,
    "هـ": 0,  "هـ+": 0,
    "A+": 95, "A": 90, "A-": 87,
    "B+": 85, "B": 80, "B-": 77,
    "C+": 75, "C": 70, "C-": 67,
    "D+": 65, "D": 60, "F": 0, "W": 0,
    "غ": 0, "راسب": 0, "محروم": 0,
}

FAILING_GRADES = ["F", "D", "D+", "ع", "راسب", "محروم"]

print("Loading sentence transformer models...")
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SENTENCE_MODEL      = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    TITLE_CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("Sentence transformer models loaded.")
except Exception as e:
    print(f"Warning: Could not load sentence transformer: {e}")
    SENTENCE_MODEL      = None
    TITLE_CROSS_ENCODER = None

BASE_DIR        = Path(__file__).parent
OCR_SCRIPT      = BASE_DIR / "1_ocr_pipeline.py"
EXTRACT_SCRIPT  = BASE_DIR / "2_extraction_pipeline.py"
JSON_OUTPUT_DIR = BASE_DIR / "json_outputs"
INPUT_PDF_DIR   = BASE_DIR / "input_pdfs"


# ── Input validation ──────────────────────────────────────────────────────────

def validate_uuid(value: str, field_name: str = "id"):
    if not value or value.strip() in ("undefined", "null", ""):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} is missing. Make sure you are logged in and try again.",
        )
    try:
        uuid.UUID(str(value))
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} '{value}' is not a valid UUID.",
        )


# ── Grade helpers ──────────────────────────────────────────────────────────────

def grade_to_numeric(grade: str) -> float:
    if not grade:
        return 0.0
    grade = str(grade).strip()
    try:
        return float(grade)
    except ValueError:
        return float(LETTER_TO_NUMERIC.get(grade, 0))

def grade_passed(grade_letter: str, grade_numeric) -> bool:
    if grade_letter in FAILING_GRADES:
        return False
    if grade_numeric is not None:
        try:
            if float(grade_numeric) < MIN_GRADE:
                return False
        except Exception:
            pass
    else:
        numeric = grade_to_numeric(grade_letter)
        if numeric > 0 and numeric < MIN_GRADE:
            return False
    return True


# ── Domain helpers ─────────────────────────────────────────────────────────────

def same_domain(title1: str, title2: str) -> bool:
    t1, t2 = title1.lower(), title2.lower()
    for domain in DOMAIN_KEYWORDS:
        if any(k in t1 for k in domain) and any(k in t2 for k in domain):
            return True
    return False

def detect_domain(title: str) -> Optional[str]:
    title = title.lower()
    if any(k in title for k in ["programming", "program"]):
        return "programming"
    if any(k in title for k in ["network", "networks"]):
        return "networks"
    if any(k in title for k in ["intelligence", "intelligent", "ai"]):
        return "artificial_intelligence"
    if any(k in title for k in ["computation", "automata", "theory"]):
        return "theory"
    if any(k in title for k in ["architecture", "organization"]):
        return "architecture"
    if any(k in title for k in ["digital", "logic", "circuit"]):
        return "logic_design"
    if any(k in title for k in ["database", "databases"]):
        return "database"
    if any(k in title for k in ["algorithm", "data structure"]):
        return "algorithms"
    if any(k in title for k in ["security", "cryptography"]):
        return "security"
    if any(k in title for k in ["discrete"]):
        return "discrete"
    return None


# ── Similarity helpers ─────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_academic_title(text: str) -> str:
    text = str(text).lower()
    synonyms = {
        "introduction to": "intro",
        "intro to":        "intro",
        "fundamentals of": "intro",
        "fundamental of":  "intro",
        "principles of":   "intro",
        "basics of":       "intro",
        "computer programming":    "programming",
        "artificial intelligence": "intelligent systems",
    }
    for old, new in synonyms.items():
        text = text.replace(old, new)
    text = re.sub(r'^[a-z]{2,6}\s?\d{3,4}[a-z]?(-\d)?\s*', '', text)
    text = re.sub(r'^\d{4}[a-z]{2,4}-?\d?\s*', '', text)
    text = re.sub(r'\biii\b', '3', text)
    text = re.sub(r'\bii\b',  '2', text)
    text = re.sub(r'\biv\b',  '4', text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_institution(name: str) -> str:
    text = str(name).lower().strip()
    text = re.sub(r"['\u2018\u2019\u02bc]", "", text)
    text = text.rstrip(".,").strip()
    for word in ["university", "univ", "college", "institute", "of", "the", "and", "&"]:
        text = re.sub(rf'\b{word}\b', '', text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def institutions_match(inst1: str, inst2: str) -> bool:
    n1 = normalize_institution(inst1)
    n2 = normalize_institution(inst2)
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True
    if n1 in n2 or n2 in n1:
        return True
    words1  = set(n1.split())
    words2  = set(n2.split())
    shorter = min(words1, words2, key=len)
    if not shorter:
        return False
    return len(words1 & words2) / len(shorter) >= 0.5

def titles_match(title1: str, title2: str) -> bool:
    t1 = normalize_academic_title(title1)
    t2 = normalize_academic_title(title2)
    if not t1 or not t2:
        return False
    if t1 == t2:
        return True
    if t1 in t2 or t2 in t1:
        return True
    words1  = set(t1.split())
    words2  = set(t2.split())
    shorter = min(words1, words2, key=len)
    if not shorter:
        return False
    return len(words1 & words2) / len(shorter) >= 0.6

def strip_level(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'\s*\bii\b\s*$', '', text)
    text = re.sub(r'\s*\bi\b\s*$',  '', text)
    text = re.sub(r'\s*\b[12]\b\s*$', '', text)
    return text.strip()

def detect_level(title: str, code: str) -> str:
    t = str(title).lower()
    if re.search(r'\bii\b', t) or re.search(r'\b2\b', t):
        return "2"
    if re.search(r'\bi\b', t) or re.search(r'\b1\b', t):
        return "1"
    return None

def semantic_similarity(text1: str, text2: str) -> float:
    if not SENTENCE_MODEL:
        return 0.0
    t1 = normalize_text(text1)
    t2 = normalize_text(text2)
    if not t1 or not t2:
        return 0.0
    vecs = SENTENCE_MODEL.encode([t1, t2], convert_to_numpy=True, normalize_embeddings=True)
    return float(np.dot(vecs[0], vecs[1]))

def lexical_similarity(text1: str, text2: str) -> float:
    t1 = normalize_text(text1)
    t2 = normalize_text(text2)
    if not t1 or not t2:
        return 0.0
    try:
        mat = TfidfVectorizer().fit_transform([t1, t2])
        return float(sk_cosine(mat[0:1], mat[1:2])[0][0])
    except Exception:
        return 0.0

def hybrid_similarity(text1: str, text2: str) -> float:
    return round(semantic_similarity(text1, text2) * 0.75 + lexical_similarity(text1, text2) * 0.25, 4)

def get_title_score(s_title: str, t_title: str, verbose: bool = False) -> float:
    if normalize_academic_title(s_title) == normalize_academic_title(t_title):
        if verbose:
            print(f"         Title Tier 1 — Exact match after normalization  -> 1.0")
        return 1.0
    s_subj = strip_level(s_title)
    t_subj = strip_level(t_title)
    if normalize_academic_title(s_subj) == normalize_academic_title(t_subj):
        if verbose:
            print(f"         Title Tier 2 — Same subject, different level    -> 0.90")
        return 0.90
    if TITLE_CROSS_ENCODER:
        score = TITLE_CROSS_ENCODER.predict([s_subj, t_subj])
        result = float(1 / (1 + np.exp(-score)))
        if verbose:
            print(f"         Title Tier 3 — CrossEncoder score (sigmoid)     -> {round(result, 4)}")
        return result
    return hybrid_similarity(s_title, t_title)

def weighted_similarity(
    t_content: str, s_content: str,
    t_desc: str,    s_desc: str,
    t_title: str = "", s_title: str = "",
    t_code: str  = "", s_code: str  = "",
    verbose: bool = False,
) -> tuple:
    sem_sim  = semantic_similarity(t_content, s_content)
    lex_sim  = lexical_similarity(t_content, s_content)
    content_sim = round(sem_sim * 0.75 + lex_sim * 0.25, 4)
    title_sim   = get_title_score(s_title, t_title, verbose=verbose)

    desc_gate = 0.20 if title_sim >= 0.85 else 0.40

    if title_sim >= 1.0:
        base = round(content_sim * 0.40 + title_sim * 0.60, 4)
        _formula = "Content x0.40 + Title x0.60"
        _formula_label = "exact match"
    elif title_sim >= 0.85:
        base = round(content_sim * 0.70 + title_sim * 0.30, 4)
        _formula = "Content x0.70 + Title x0.30"
        _formula_label = "strong title match"
    elif len(t_content.strip()) < 1000:
        base = round(content_sim * 0.50 + title_sim * 0.50, 4)
        _formula = "Content x0.50 + Title x0.50"
        _formula_label = "thin content"
    else:
        base = round(content_sim * 0.85 + title_sim * 0.15, 4)
        _formula = "Content x0.85 + Title x0.15"
        _formula_label = "normal"

    if content_sim >= desc_gate and t_desc and s_desc:
        desc_sim = hybrid_similarity(t_desc, s_desc)
        blended  = round(base * 0.70 + desc_sim * 0.30, 4)
        final    = max(base, blended)
        _desc_blend = f"max({round(base*100,1)}%, {round(base*100,1)}%x0.70 + {round(desc_sim*100,1)}%x0.30) = {round(final*100,1)}%"
    else:
        desc_sim = 0.0
        final    = base
        _desc_blend = "skipped"

    if verbose:
        print(f"  STEP 5 — Weighted Similarity: {s_code} vs {t_code}")
        print(f"    Semantic similarity:   {round(sem_sim*100,1)}%")
        print(f"    Lexical similarity:    {round(lex_sim*100,1)}%")
        print(f"    Content similarity:    {round(content_sim*100,1)}%  (75% semantic + 25% lexical)")
        print(f"    Title score:           {round(title_sim,2)}    ({_formula_label})")
        print(f"    Formula: {_formula}")
        print(f"    Base score:            {round(base*100,1)}%")
        print(f"    Description blend:     {_desc_blend}")

    if "program" in s_title.lower() or "program" in t_title.lower():
        s_level = detect_level(s_title, s_code)
        t_level = detect_level(t_title, t_code)
        if s_level is not None and t_level is not None and s_level != t_level:
            final = round(final - 0.15, 4)
            if verbose:
                print(f"  STEP 6 — Programming Level Penalty")
                print(f"    Both are programming courses")
                print(f"    Student level: {s_level}  |  Taibah level: {t_level}  -> Penalty applied: -15%")
                print(f"    Score after penalty: {round(final*100,1)}%")
        else:
            if verbose:
                print(f"  STEP 6 — Programming Level Penalty")
                print(f"    Both are programming courses")
                print(f"    Student level: {s_level or 'ambiguous'}  |  Taibah level: {t_level or 'ambiguous'}  -> No penalty")

    return final, content_sim, desc_sim

def make_decision(similarity: float) -> str:
    return "Equivalent" if similarity >= SIMILARITY_THRESHOLD else "Not Equivalent"


# ── Cache helpers ──────────────────────────────────────────────────────────────

def check_similarity_cache(taibah_code: str, ext_code: str, institution: str):
    try:
        result = supabase.table("similarity_cache") \
            .select("similarity_percentage, trs_decision") \
            .eq("taibah_course_code", taibah_code) \
            .eq("external_course_code", ext_code) \
            .eq("external_institution", institution) \
            .limit(1) \
            .execute()
        if result.data:
            return result.data[0]
    except Exception:
        pass
    return None

def save_similarity_cache(taibah_code: str, ext_code: str, institution: str, sim_pct: float, decision: str):
    try:
        supabase.table("similarity_cache").upsert({
            "taibah_course_code":    taibah_code,
            "external_course_code":  ext_code,
            "external_institution":  institution,
            "similarity_percentage": sim_pct,
            "trs_decision":          decision,
        }).execute()
    except Exception as e:
        print(f"     Cache save skipped: {e}")


# ── Content fetch helpers ──────────────────────────────────────────────────────

def get_taibah_content(course_code: str) -> Optional[str]:
    result = supabase.table("taibah_course_content") \
        .select("content_text") \
        .eq("course_code", course_code) \
        .execute()
    if not result.data:
        return None
    parts = [r["content_text"] for r in result.data if r.get("content_text")]
    return "\n".join(parts) if parts else None

def get_taibah_description(course_code: str) -> Optional[str]:
    result = supabase.table("taibah_courses") \
        .select("general_description") \
        .eq("course_code", course_code) \
        .single() \
        .execute()
    return result.data.get("general_description") if result.data else None

def get_external_content(external_course_id: str) -> Optional[str]:
    result = supabase.table("external_course_content") \
        .select("content_text") \
        .eq("external_course_id", external_course_id) \
        .execute()
    if not result.data:
        return None
    parts = [r["content_text"] for r in result.data if r.get("content_text")]
    return "\n".join(parts) if parts else None

def get_taibah_courses_by_domain(student_title: str) -> list:
    all_courses = supabase.table("taibah_courses") \
        .select("course_code, course_title_en, general_description, credit_hours, domain") \
        .eq("is_cs_related", True) \
        .execute().data or []

    domain_courses = [
        c for c in all_courses
        if same_domain(student_title, c.get("course_title_en", ""))
    ]
    return domain_courses if domain_courses else all_courses


# ── OCR + Extraction ───────────────────────────────────────────────────────────

def run_ocr_and_extract(pdf_path: Path) -> Optional[dict]:
    import sys
    python   = sys.executable
    pdf_name = pdf_path.stem
    env      = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"]       = "1"

    JSON_OUTPUT_DIR.mkdir(exist_ok=True)
    existing_json = list(JSON_OUTPUT_DIR.glob(f"{pdf_name}_*.json"))
    if existing_json:
        print(f"     JSON already exists for {pdf_name} — skipping OCR+extraction")
        try:
            return json.loads(existing_json[0].read_text(encoding="utf-8"))
        except Exception as e:
            print(f"     Could not read existing JSON, will re-run: {e}")

    print(f"   Running OCR on {pdf_path.name}...")
    ocr_result = subprocess.run(
        [python, str(OCR_SCRIPT), "--file", str(pdf_path)],
        capture_output=True, cwd=str(BASE_DIR), env=env
    )
    if ocr_result.returncode != 0:
        print(f"   OCR failed: {ocr_result.stderr.decode('utf-8', errors='replace')[:500]}")
        return None

    print(f"   Running extraction on {pdf_name}...")
    extract_result = subprocess.run(
        [python, str(EXTRACT_SCRIPT), "--name", pdf_name],
        capture_output=True, cwd=str(BASE_DIR), env=env
    )
    if extract_result.returncode != 0:
        print(f"   Extraction failed: {extract_result.stderr.decode('utf-8', errors='replace')[:500]}")
        return None

    json_files = list(JSON_OUTPUT_DIR.glob(f"{pdf_name}_*.json"))
    if not json_files:
        print(f"    No JSON output found for {pdf_name}")
        return None

    try:
        return json.loads(json_files[0].read_text(encoding="utf-8"))
    except Exception as e:
        print(f"    Could not read JSON: {e}")
        return None


# ── DB Insert helpers ──────────────────────────────────────────────────────────

def insert_transcript_to_db(student_id: str, data: dict):
    student_info = data.get("student_info", {})

    update_data = {}
    if student_info.get("student_name"):
        update_data["student_name"] = student_info["student_name"]
    if student_info.get("national_id"):
        update_data["national_id"]  = student_info["national_id"]
    if student_info.get("institution"):
        update_data["institution"]  = student_info["institution"]
    if student_info.get("college"):
        update_data["college"]      = student_info["college"]
    if student_info.get("major"):
        update_data["major"]        = student_info["major"]
    if student_info.get("student_id"):
        update_data["college_id"]   = student_info["student_id"]

    if update_data:
        supabase.table("students") \
            .update(update_data) \
            .eq("student_id", student_id) \
            .execute()

    existing_rows = supabase.table("student_transcript_courses") \
        .select("course_code, course_name") \
        .eq("student_id", student_id) \
        .execute().data or []

    existing_pairs = {
        (r.get("course_code", "").strip().upper(), (r.get("course_name") or "").strip().lower())
        for r in existing_rows
    }

    courses  = data.get("courses", [])
    inserted = 0
    skipped  = 0

    for course in courses:
        if not course.get("course_code"):
            continue

        code = course.get("course_code", "").strip().upper()
        name = (course.get("course_name") or "").strip().lower()

        if (code, name) in existing_pairs:
            skipped += 1
            continue

        try:
            supabase.table("student_transcript_courses").insert({
                "student_id":    student_id,
                "institution":   student_info.get("institution"),
                "course_code":   code,
                "course_name":   course.get("course_name"),
                "credit_hours":  course.get("credit_hours"),
                "grade_letter":  course.get("grade_letter"),
                "grade_numeric": course.get("grade_numeric"),
                "grade_points":  course.get("grade_points"),
                "semester":      course.get("semester"),
            }).execute()
            inserted += 1
            existing_pairs.add((code, name))
        except Exception as e:
            print(f"     Could not insert course {code}: {e}")

    print(f"    Inserted {inserted} courses, skipped {skipped} duplicates into student_transcript_courses")


def insert_external_course_spec(student_id: str, data: dict) -> Optional[str]:
    course_code = data.get("course_code", "").strip().upper()
    institution = data.get("institution", "")
    title       = data.get("course_title", "")

    transcript_course = supabase.table("student_transcript_courses") \
        .select("credit_hours") \
        .eq("student_id", student_id) \
        .eq("course_code", course_code) \
        .limit(1) \
        .execute()

    credit_hours = None
    if transcript_course.data:
        credit_hours = transcript_course.data[0].get("credit_hours")

    existing = supabase.table("external_courses") \
        .select("id") \
        .eq("course_title", title) \
        .eq("institution", institution) \
        .limit(1) \
        .execute()

    if existing.data:
        ext_id = existing.data[0]["id"]
        print(f"     External course '{title}' from '{institution}' already exists — reusing")
    else:
        result = supabase.table("external_courses").insert({
            "course_code":         course_code,
            "course_title":        title,
            "institution":         institution,
            "general_description": data.get("general_description"),
            "credit_hours":        credit_hours,
            "is_cs_related":       data.get("is_cs_related", False),
            "domain":              detect_domain(title),
        }).execute()
        ext_id = result.data[0]["id"]

        for section in data.get("content_sections", []):
            content_text = section.get("content_text", "").strip()

            if not content_text:
                parts     = []
                heading   = section.get("heading", "")
                topics    = section.get("topics", [])
                practical = section.get("practical_topics", [])
                if heading:
                    parts.append(heading)
                if topics:
                    parts.append(", ".join(str(t) for t in topics))
                if practical:
                    parts.append("Practical: " + ", ".join(str(t) for t in practical))
                content_text = ". ".join(filter(None, parts))
            else:
                topics    = section.get("topics", [])
                practical = section.get("practical_topics", [])
                extra     = []
                if topics:
                    extra.append(", ".join(str(t) for t in topics))
                if practical:
                    extra.append("Practical: " + ", ".join(str(t) for t in practical))
                if extra:
                    content_text = content_text + "\n" + ". ".join(extra)

            if content_text.strip():
                supabase.table("external_course_content").insert({
                    "external_course_id": ext_id,
                    "section_heading":    section.get("heading", ""),
                    "topics":             section.get("topics", []),
                    "practical_topics":   section.get("practical_topics", []),
                    "content_text":       content_text,
                }).execute()

        print(f"    External course '{title}' inserted")

    try:
        supabase.table("student_external_courses").upsert({
            "student_id":         student_id,
            "external_course_id": ext_id,
        }).execute()
    except Exception as e:
        print(f"     Could not link student to external course: {e}")

    return ext_id


# ── Core comparison logic ──────────────────────────────────────────────────────

def build_spec_content(ext_id: str, desc: str) -> tuple:
    """Fetch content text and description for an external course spec."""
    content = get_external_content(ext_id) or ""
    return content, desc or ""


def run_equivalency_comparison(form_id: str, student_id: str):
    try:
        print(f" Running equivalency for form {form_id}, student {student_id}")

        raw_student_courses = supabase.table("student_transcript_courses") \
            .select("*") \
            .eq("student_id", student_id) \
            .execute().data or []

        if not raw_student_courses:
            supabase.table("equivalency_form") \
                .update({"status": "failed"}) \
                .eq("form_id", form_id) \
                .execute()
            return

        seen_pairs      = set()
        student_courses = []
        for sc in raw_student_courses:
            key = (
                (sc.get("course_code") or "").strip().upper(),
                (sc.get("course_name") or "").strip().lower(),
            )
            if key not in seen_pairs:
                seen_pairs.add(key)
                student_courses.append(sc)

        print(f"    {len(raw_student_courses)} transcript rows -> {len(student_courses)} unique courses after dedup")

        all_ext = supabase.table("external_courses") \
            .select("id, general_description, course_title, institution") \
            .execute().data or []

        for s_course in student_courses:
            s_code         = (s_course.get("course_code") or "").strip()
            s_name         = s_course.get("course_name", "")
            s_grade_letter = s_course.get("grade_letter", "")
            s_grade_num    = s_course.get("grade_numeric")
            s_credit       = s_course.get("credit_hours")
            # FIX: normalize institution name to avoid cache misses from trailing
            # punctuation or apostrophe variants (e.g. "University of Hail." vs "University of Hail")
            s_institution  = (s_course.get("institution") or "").strip().rstrip(".,").strip()

            if not s_code:
                continue

            print(f"")
            print(f" {'━'*54}")
            print(f"  COURSE: {s_code} — {s_name}")
            print(f" {'━'*54}")

            # ── Grade check ──
            if not grade_passed(s_grade_letter, s_grade_num):
                print(f"  STEP 1 — Eligibility Check")
                print(f"    Grade: {s_grade_letter or s_grade_num} (failed — below minimum {MIN_GRADE}%)")
                print(f"    Course excluded from equivalency")
                print(f"    {s_code} — grade too low ({s_grade_letter or s_grade_num})")
                taibah_courses_for_rejection = get_taibah_courses_by_domain(s_name or s_code)
                for t_course in taibah_courses_for_rejection:
                    try:
                        supabase.table("course_comparison_results").upsert({
                            "form_id":               form_id,
                            "taibah_course_code":    t_course["course_code"],
                            "student_course_code":   s_code,
                            # FIX: use None instead of 0 so frontend shows reason not 0%
                            "similarity_percentage": None,
                            "trs_suggestion":        "Not Equivalent",
                            "trs_rejection_reason":  f"لم تتم المقارنة — الدرجة أقل من الحد الأدنى المطلوب ({MIN_GRADE}%)",
                        }).execute()
                    except Exception as e:
                        print(f"     Could not save grade rejection: {e}")
                continue

            # ── Find matching course spec ──
            ext_match = None
            for ext in all_ext:
                ext_institution = ext.get("institution", "")
                ext_title       = ext.get("course_title", "")

                if institutions_match(s_institution, ext_institution) and \
                   titles_match(s_name, ext_title):
                    ext_match = ext
                    print(f"    Spec match: '{s_name}' @ '{s_institution}'"
                          f" -> '{ext_title}' @ '{ext_institution}'")
                    break

            if not ext_match:
                print(f"     {s_code} ({s_name}) — no matching course spec uploaded")
                continue

            ext_id    = ext_match["id"]
            s_desc    = ext_match.get("general_description", "") or ""
            print(f"  STEP 1 — Eligibility Check")
            print(f"    Grade: {s_grade_letter or s_grade_num} (passed)")
            print(f"    Credit hours: {s_credit or 'N/A'} (meets requirement)")

            s_content = get_external_content(ext_id)

            if not s_content:
                print(f"     {s_code} ({s_name}) — spec found but content is empty, skipping")
                continue

            # ── Domain filtering ──
            taibah_courses = get_taibah_courses_by_domain(s_name or s_code)
            print(f"  STEP 3 — Domain Filtering")
            _dom = detect_domain(s_name or s_code) or "general"
            print(f"    Student domain: {_dom}")
            print(f"    Matched {len(taibah_courses)} Taibah courses in same domain")

            title_ranked = sorted(
                taibah_courses,
                key=lambda t: get_title_score(s_name, t.get("course_title_en", "")),
                reverse=True
            )

            print(f"  STEP 4 — Title Pre-Ranking")
            for _t in title_ranked:
                _ts = get_title_score(s_name, _t.get("course_title_en", ""))
                _tier = "Tier 1 Exact Match" if _ts == 1.0 else ("Tier 2 Same Subject" if _ts == 0.90 else "Tier 3 CrossEncoder")
                print(f"    {_t['course_code']:<8} {_t.get('course_title_en',''):<38} -> {_tier:<22} {round(_ts,2)}")

            for t_course in title_ranked:
                t_code   = t_course["course_code"]
                t_title  = t_course.get("course_title_en", "")
                t_credit = t_course.get("credit_hours")
                t_desc   = t_course.get("general_description") or get_taibah_description(t_code) or ""

                # ── Credit hours check ──
                if s_credit and t_credit:
                    try:
                        if float(s_credit) < float(t_credit):
                            print(f"")
                            print(f"  STEP 1 — Eligibility Check")
                            print(f"    Credit hours: {s_credit}cr (less than Taibah {t_credit}cr — skipped)")
                            supabase.table("course_comparison_results").upsert({
                                "form_id":               form_id,
                                "taibah_course_code":    t_code,
                                "student_course_code":   s_code,
                                "similarity_percentage": None,
                                "trs_suggestion":        "Not Equivalent",
                                "trs_rejection_reason":  f"لم تتم المقارنة — عدد الساعات المعتمدة للمقرر ({s_credit}) أقل من ساعات المقرر في جامعة طيبة ({t_credit})",
                            }).execute()
                            continue
                        else:
                            pass  # credit ok, continue
                    except Exception:
                        pass

                # ── Cache check ──
                cached = check_similarity_cache(t_code, s_code, s_institution)
                if cached:
                    print(f"  STEP 2 — Cache Check")
                    print(f"    Cached result found — score: {cached['similarity_percentage']}%  ->  {cached['trs_decision']}")
                    supabase.table("course_comparison_results").upsert({
                        "form_id":               form_id,
                        "taibah_course_code":    t_code,
                        "student_course_code":   s_code,
                        "similarity_percentage": cached["similarity_percentage"],
                        "trs_suggestion":        cached["trs_decision"],
                    }).execute()
                    continue
                else:
                    print(f"  STEP 2 — Cache Check")
                    print(f"    No cached result found — proceeding to full comparison")

                # ── Similarity scoring ──
                t_content = get_taibah_content(t_code)
                if not t_content:
                    continue

                final, content_sim, desc_sim = weighted_similarity(
                    t_content, s_content,
                    t_desc,    s_desc,
                    t_title,   s_name,
                    t_code,    s_code,
                    verbose=True,
                )

                # ── Domain boost ──
                if same_domain(s_name, t_title):
                    pre_boost = final
                    final = min(round(final + 0.03, 4), 1.0)
                    print(f"  STEP 7 — Domain Boost")
                    print(f"    Same domain -> +3%")
                    print(f"    Final score: {round(final*100,1)}% -> {'EQUIVALENT' if final >= SIMILARITY_THRESHOLD else 'NOT EQUIVALENT'}")
                else:
                    print(f"  STEP 7 — Domain Boost")
                    print(f"    Different domain — no boost")
                    print(f"    Final score: {round(final*100,1)}% -> {'EQUIVALENT' if final >= SIMILARITY_THRESHOLD else 'NOT EQUIVALENT'}")

                sim_pct  = round(final * 100, 2)
                decision = make_decision(final)
                print(f" {'━'*54}")

                save_similarity_cache(t_code, s_code, s_institution, sim_pct, decision)

                supabase.table("course_comparison_results").upsert({
                    "form_id":               form_id,
                    "taibah_course_code":    t_code,
                    "student_course_code":   s_code,
                    "similarity_percentage": sim_pct,
                    "trs_suggestion":        decision,
                }).execute()

        # ── Combined comparison pass ──────────────────────────────────────────
        supabase.table("equivalency_form") \
            .update({"status": "pending_committee_review"}) \
            .eq("form_id", form_id) \
            .execute()

        print(f"")
        print(f" {'━'*54}")
        print(f"  EQUIVALENCY ANALYSIS COMPLETE")
        print(f" {'━'*54}")
        print(f" Form {form_id} completed")

    except Exception as e:
        import traceback
        print(f" Error processing form {form_id}: {e}")
        traceback.print_exc()
        supabase.table("equivalency_form") \
            .update({"status": "failed"}) \
            .eq("form_id", form_id) \
            .execute()


# ── Request models ─────────────────────────────────────────────────────────────

class EquivalencyRequest(BaseModel):
    student_id: str


# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.post("/upload/transcript")
async def upload_transcript(
    student_id: str = Form(...),
    file: UploadFile = File(...)
):
    validate_uuid(student_id, "student_id")
    allowed = [".pdf", ".jpg", ".jpeg", ".png"]
    ext     = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"File type {ext} not supported.")

    student = supabase.table("students") \
        .select("student_id") \
        .eq("student_id", student_id) \
        .single() \
        .execute()
    if not student.data:
        raise HTTPException(status_code=404, detail="Student not found")

    INPUT_PDF_DIR.mkdir(exist_ok=True)
    save_path = INPUT_PDF_DIR / f"transcript_{student_id}{ext}"
    with open(save_path, "wb") as f:
        f.write(await file.read())

    data = run_ocr_and_extract(save_path)
    if not data:
        raise HTTPException(status_code=500, detail="OCR or extraction failed")

    insert_transcript_to_db(student_id, data)

    return {
        "message":       "Transcript processed successfully",
        "student_name":  data.get("student_info", {}).get("student_name", ""),
        "courses_found": len(data.get("courses", [])),
    }


@app.post("/upload/course-specs")
async def upload_course_specs(
    student_id: str = Form(...),
    files: Optional[List[UploadFile]] = File(default=None),
    file: Optional[UploadFile] = File(default=None),
):
    validate_uuid(student_id, "student_id")
    all_files: List[UploadFile] = []
    if files:
        all_files.extend(files)
    if file is not None:
        all_files.append(file)

    if not all_files:
        raise HTTPException(status_code=400, detail="No files provided.")

    INPUT_PDF_DIR.mkdir(exist_ok=True)
    results = []

    for f in all_files:
        ext = Path(f.filename).suffix.lower()
        if ext != ".pdf":
            results.append({"file": f.filename, "status": "skipped", "reason": "not a PDF"})
            continue

        file_bytes    = await f.read()
        safe_filename = re.sub(r"[^\w\-.]", "_", f.filename)
        save_path     = INPUT_PDF_DIR / f"spec_{student_id}_{safe_filename}"

        with open(save_path, "wb") as out:
            out.write(file_bytes)

        data = run_ocr_and_extract(save_path)
        if not data:
            results.append({"file": f.filename, "status": "failed", "reason": "OCR/extraction failed"})
            continue

        ext_id = insert_external_course_spec(student_id, data)

        results.append({
            "file":         f.filename,
            "status":       "success",
            "course_code":  data.get("course_code"),
            "course_title": data.get("course_title"),
            "external_id":  ext_id,
        })

    return {
        "message": f"Processed {len(all_files)} file(s)",
        "results": results,
    }


@app.post("/equivalency/request")
async def submit_equivalency_request(
    request: EquivalencyRequest,
    background_tasks: BackgroundTasks
):
    student_id = request.student_id
    validate_uuid(student_id, "student_id")

    student = supabase.table("students") \
        .select("student_id, student_name") \
        .eq("student_id", student_id) \
        .single() \
        .execute()
    if not student.data:
        raise HTTPException(status_code=404, detail="Student not found")

    courses = supabase.table("student_transcript_courses") \
        .select("id") \
        .eq("student_id", student_id) \
        .limit(1) \
        .execute()
    if not courses.data:
        raise HTTPException(status_code=400, detail="No courses found. Upload your transcript first.")

    form_result = supabase.table("equivalency_form") \
        .insert({"student_id": student_id, "status": "processing"}) \
        .execute()

    form_id = form_result.data[0]["form_id"]

    background_tasks.add_task(
        run_equivalency_comparison,
        form_id=form_id,
        student_id=student_id,
    )

    return {
        "message": "Equivalency request submitted. Analysis is underway.",
        "form_id": form_id,
        "status":  "processing",
        "student": student.data["student_name"],
    }


@app.get("/equivalency/{form_id}/status")
async def get_form_status(form_id: str):
    validate_uuid(form_id, "form_id")
    form = supabase.table("equivalency_form") \
        .select("form_id, status, updated_at") \
        .eq("form_id", form_id) \
        .single() \
        .execute()
    if not form.data:
        raise HTTPException(status_code=404, detail="Form not found")

    status_messages = {
        "processing":               "Analysis in progress — please wait",
        "pending_committee_review": "Analysis complete — awaiting committee review",
        "completed":                "Committee review completed",
        "failed":                   "An error occurred during analysis",
    }

    return {
        "form_id":    form_id,
        "status":     form.data["status"],
        "message":    status_messages.get(form.data["status"], form.data["status"]),
        "updated_at": form.data.get("updated_at"),
    }


@app.get("/equivalency/{form_id}")
async def get_equivalency_form(form_id: str):
    validate_uuid(form_id, "form_id")
    form = supabase.table("equivalency_form") \
        .select("*") \
        .eq("form_id", form_id) \
        .single() \
        .execute()
    if not form.data:
        raise HTTPException(status_code=404, detail="Form not found")

    results = supabase.table("course_comparison_results") \
        .select("*") \
        .eq("form_id", form_id) \
        .execute()

    return {
        "form":    form.data,
        "courses": results.data or [],
    }


@app.get("/equivalency/student/{student_id}")
async def get_student_forms(student_id: str):
    validate_uuid(student_id, "student_id")
    forms = supabase.table("equivalency_form") \
        .select("*") \
        .eq("student_id", student_id) \
        .order("creation_date", desc=True) \
        .execute()
    return {"forms": forms.data or []}


@app.get("/committee/form/{form_id}/full")
async def get_form_full(form_id: str):
    validate_uuid(form_id, "form_id")
    form = supabase.table("equivalency_form") \
        .select("*") \
        .eq("form_id", form_id) \
        .single() \
        .execute()
    if not form.data:
        raise HTTPException(status_code=404, detail="Form not found")

    student = supabase.table("students") \
        .select("*") \
        .eq("student_id", form.data["student_id"]) \
        .single() \
        .execute()

    transcript = supabase.table("student_transcript_courses") \
        .select("*") \
        .eq("student_id", form.data["student_id"]) \
        .execute()

    comparisons = supabase.table("course_comparison_results") \
        .select("*") \
        .eq("form_id", form_id) \
        .execute()

    linked_specs = supabase.table("student_external_courses") \
        .select("external_courses(id, course_title, general_description)") \
        .eq("student_id", form.data["student_id"]) \
        .execute()

    spec_desc_map = {}
    for link in (linked_specs.data or []):
        ec = link.get("external_courses") or {}
        if ec.get("id"):
            spec_desc_map[ec["id"]] = ec.get("general_description")

    comp_by_code = {}
    for row in (comparisons.data or []):
        code = row.get("student_course_code")
        if code not in comp_by_code:
            comp_by_code[code] = []
        comp_by_code[code].append(row)

    all_ext = supabase.table("external_courses") \
        .select("id, course_title, institution, general_description") \
        .execute().data or []

    seen_pairs        = set()
    unique_transcript = []
    for t in (transcript.data or []):
        key = (
            (t.get("course_code") or "").strip().upper(),
            (t.get("course_name") or "").strip().lower(),
        )
        if key not in seen_pairs:
            seen_pairs.add(key)
            unique_transcript.append(t)

    courses = []
    for t in unique_transcript:
        s_code        = t.get("course_code")
        s_name        = t.get("course_name", "")
        s_institution = t.get("institution", "")
        matches       = comp_by_code.get(s_code, [])

        best = None
        if matches:
            # Priority 1: committee-approved match
            approved = [r for r in matches if r.get("committee_decision") == "Approved"]
            if approved:
                best = max(approved, key=lambda x: x.get("similarity_percentage") or 0)
            else:
                # Priority 2: highest similarity (for committee view before decision)
                best = max(matches, key=lambda x: x.get("similarity_percentage") or 0)

        uploaded_desc = None
        has_spec      = False
        for ext in all_ext:
            if institutions_match(s_institution, ext.get("institution", "")) and \
               titles_match(s_name, ext.get("course_title", "")):
                uploaded_desc = ext.get("general_description")
                has_spec      = True
                break

        courses.append({
            "student_course_code": s_code,
            "student_course_name": s_name,
            "grade_letter":        t.get("grade_letter"),
            "credit_hours":        t.get("credit_hours"),
            "semester":            t.get("semester"),
            "has_spec":            has_spec,
            "best_match": {
                "target_course_code":          best.get("taibah_course_code") if best else None,
                "similarity_percentage":       best.get("similarity_percentage") if best else None,
                "trs_suggestion":              best.get("trs_suggestion") if best else None,
                "trs_rejection_reason":        best.get("trs_rejection_reason") if best else None,
                "committee_decision":          best.get("committee_decision") if best else None,
                "uploaded_course_description": uploaded_desc,
                "form_id":                     form_id,
            } if best else None,
            "all_matches": [
                {
                    "course_code":                  r.get("taibah_course_code"),
                    "similarity_percentage":        r.get("similarity_percentage"),
                    "trs_suggestion":               r.get("trs_suggestion"),
                    "trs_rejection_reason":         r.get("trs_rejection_reason"),
                    "committee_decision":           r.get("committee_decision"),
                    "committee_rejection_reason":   r.get("committee_rejection_reason"),
                    "form_id":                      form_id,
                }
                for r in matches
            ],
        })

    # ── Combined equivalency results ──────────────────────────────────────────
    # These are rows where student_course_code contains " + " (e.g. "CSC1102 + CSC1103")
    # They don't map to any single transcript course so we return them separately.
    combined_results = []
    for code, rows in comp_by_code.items():
        if " + " in (code or ""):
            for r in rows:
                combined_results.append({
                    "combined_code":              code,
                    "taibah_course_code":         r.get("taibah_course_code"),
                    "similarity_percentage":      r.get("similarity_percentage"),
                    "trs_suggestion":             r.get("trs_suggestion"),
                    "trs_rejection_reason":       r.get("trs_rejection_reason"),
                    "committee_decision":         r.get("committee_decision"),
                    "committee_rejection_reason": r.get("committee_rejection_reason"),
                    "form_id":                    form_id,
                })

    return {
        "form":             form.data,
        "student":          student.data or {},
        "courses":          courses,
        "combined_results": combined_results,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
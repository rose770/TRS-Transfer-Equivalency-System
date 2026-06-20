"""
test_similarity.py
==================
Local testing script — runs the similarity comparison without starting the web server.
Reads a student course spec JSON file and compares it against all Taibah courses in the DB.

Flow:
  1. Grade check         — student must have 70%+ or passing letter grade
  2. Domain filtering    — fetch only same-domain Taibah courses
  3. Title pre-check     — rank Taibah courses by title similarity (CrossEncoder)
  4. Credit hours check  — skip if student credit hours < Taibah credit hours
  5. Cache check         — if pair already compared, reuse the score from DB
  6. Similarity scoring  — compare content, title, and description
  7. Domain boost        — add +3% if both courses are in the same domain
  8. Cache result        — save the score so it's reused next time

Usage:
    python test_similarity.py --student path/to/student_course_spec.json
"""

import json
import re
import argparse
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
import os

load_dotenv()  # load environment variables from .env

# A course pair is considered equivalent if similarity reaches 70% or above
SIMILARITY_THRESHOLD = 0.70

# Letter grades that disqualify a course from equivalency consideration
FAILING_GRADES = ["F", "D", "D+", "ع", "راسب", "محروم"]

# Groups of keywords that define subject domains used to filter comparison candidates
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

# Connect to the Supabase database using credentials from .env
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Load the NLP models — these take a few seconds on first run
print(" Loading sentence transformer models...")
from sentence_transformers import SentenceTransformer, CrossEncoder
# all-mpnet-base-v2 gives strong semantic similarity for academic text
MODEL = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
# ms-marco CrossEncoder is good at scoring relevance between two short texts (titles)
TITLE_CROSS_ENCODER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print(" Models loaded.\n")


# ── Text normalization helpers ─────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Basic text normalization — lowercase and remove punctuation."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_academic_title(text: str) -> str:
    """
    Normalizes a course title for comparison by:
    - Replacing common phrase variants with a standard form ("Introduction to" -> "intro")
    - Stripping course code prefixes ("CS101" or "1312CCS-3")
    - Converting Roman numerals to digits (II -> 2, III -> 3)
    This ensures "Introduction to Programming" and "Intro to Programming" are treated as the same.
    """
    text = str(text).lower()
    # Map common phrase variants to a single standard form
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
    # Strip course code prefixes like "cs101", "csce 102", or "1312ccs-3"
    text = re.sub(r'^[a-z]{2,6}\s?\d{3,4}[a-z]?(-\d)?\s*', '', text)
    text = re.sub(r'^\d{4}[a-z]{2,4}-?\d?\s*', '', text)
    # Convert Roman numerals to digits for consistent level comparison
    text = re.sub(r'\biii\b', '3', text)
    text = re.sub(r'\bii\b',  '2', text)
    text = re.sub(r'\biv\b',  '4', text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def semantic_sim(t1: str, t2: str) -> float:
    """Computes semantic similarity between two texts using sentence embeddings.
    Encodes both texts as vectors and returns their dot product (cosine similarity)."""
    t1, t2 = normalize(t1), normalize(t2)
    if not t1 or not t2:
        return 0.0
    vecs = MODEL.encode([t1, t2], convert_to_numpy=True, normalize_embeddings=True)
    return float(np.dot(vecs[0], vecs[1]))

def lexical_sim(t1: str, t2: str) -> float:
    """Computes lexical (word-overlap) similarity using TF-IDF vectors.
    Useful for catching technical terms that appear verbatim in both texts."""
    t1, t2 = normalize(t1), normalize(t2)
    if not t1 or not t2:
        return 0.0
    try:
        mat = TfidfVectorizer().fit_transform([t1, t2])
        return float(sk_cosine(mat[0:1], mat[1:2])[0][0])
    except Exception:
        return 0.0

def hybrid(t1: str, t2: str) -> float:
    """Combines semantic (75%) and lexical (25%) similarity.
    Semantic similarity gets more weight because meaning matters more than exact words."""
    return round(semantic_sim(t1, t2) * 0.75 + lexical_sim(t1, t2) * 0.25, 4)

def strip_level(text: str) -> str:
    """Removes trailing level indicators from a course title.
    For example "Programming II" becomes "Programming" , used for subject-level comparison."""
    text = str(text).lower().strip()
    text = re.sub(r'\s*\bii\b\s*$', '', text)
    text = re.sub(r'\s*\bi\b\s*$',  '', text)
    text = re.sub(r'\s*\b[12]\b\s*$', '', text)
    return text.strip()

def get_title_score(s_title: str, t_title: str) -> float:
    """Scores the similarity between two course titles.
    Tries exact normalized match first, then subject-only match, then CrossEncoder.
    Returns a score between 0.0 (no match) and 1.0 (perfect match)."""
    # Exact match after full normalization
    if normalize_academic_title(s_title) == normalize_academic_title(t_title):
        return 1.0
    # Strip level suffix and compare subjects only
    s_subj = strip_level(s_title)
    t_subj = strip_level(t_title)
    if normalize_academic_title(s_subj) == normalize_academic_title(t_subj):
        return 0.90  # same subject, different level ,strong but not perfect
    # Fall back to CrossEncoder model for relevance scoring
    score = TITLE_CROSS_ENCODER.predict([s_subj, t_subj])
    return float(1 / (1 + np.exp(-score)))  # sigmoid to convert to 0-1 range


# ── Domain and level detection ─────────────────────────────────────────────────

def same_domain(title1: str, title2: str) -> bool:
    """Returns True if both course titles belong to the same subject domain."""
    t1, t2 = title1.lower(), title2.lower()
    for domain in DOMAIN_KEYWORDS:
        if any(k in t1 for k in domain) and any(k in t2 for k in domain):
            return True
    return False

def detect_level(title: str, code: str) -> str:
    """Detects whether a course is level 1 or level 2 based on its title.
    Returns '1', '2', or None if the level is not clear from the title.
    We intentionally ignore the course code number,catalog numbers don't indicate level."""
    t = str(title).lower()
    if re.search(r'\bii\b', t) or re.search(r'\b2\b', t):
        return "2"
    if re.search(r'\bi\b', t) or re.search(r'\b1\b', t):
        return "1"
    # Level is ambiguous ,return None so callers don't apply wrong penalties
    return None


# ── Core similarity scoring ────────────────────────────────────────────────────

def weighted_similarity(t_content, s_content, t_desc, s_desc, t_title="", s_title="", t_code="", s_code="") -> tuple:
    """
    Computes the final similarity score between a student course and a Taibah course.
    Returns a tuple of (final_score, content_sim, desc_sim) for transparency.

    Scoring logic:
    - Content similarity is the main signal (computed with hybrid semantic+lexical)
    - Title similarity adjusts the weighting strong title match means title carries more weight
    - Description similarity is added as a blend if content similarity passes a threshold
    - A -0.15 penalty is applied if both courses are programming but at different levels
    """
    content_sim = hybrid(t_content, s_content)
    title_sim   = get_title_score(s_title, t_title)

    # Lower the description gate for courses with a strong title match
    desc_gate = 0.20 if title_sim >= 0.85 else 0.40

    # Weighting depends on how confident the title match is
    if title_sim >= 1.0:
        # Perfect title match , give title score 60% weight
        base = round(content_sim * 0.40 + title_sim * 0.60, 4)
    elif title_sim >= 0.85:
        # Strong title match , give content 70%
        base = round(content_sim * 0.70 + title_sim * 0.30, 4)
    elif len(t_content.strip()) < 1000:
        # Short course content ,rely more on title since content is thin
        base = round(content_sim * 0.50 + title_sim * 0.50, 4)
    else:
        # Normal case , content carries 85% of the score
        base = round(content_sim * 0.85 + title_sim * 0.15, 4)

    # Blend in description similarity if the content similarity is high enough
    if content_sim >= desc_gate and t_desc and s_desc:
        desc_sim = hybrid(t_desc, s_desc)
        blended  = round(base * 0.70 + desc_sim * 0.30, 4)
        final    = max(base, blended)  # take the better of base and blended score
    else:
        desc_sim = 0.0
        final    = base

    # Apply programming level penalty , prevents Prog I from matching Prog II
    # Only fires when BOTH courses have an explicit level in their title
    if "program" in s_title.lower() or "program" in t_title.lower():
        s_level = detect_level(s_title, s_code)
        t_level = detect_level(t_title, t_code)
        if s_level is not None and t_level is not None and s_level != t_level:
            final = round(final - 0.15, 4)  # subtract 15% for level mismatch

    return final, content_sim, desc_sim


# ── Grade eligibility check ────────────────────────────────────────────────────

def check_grade(spec: dict) -> tuple:
    """Checks whether the student's grade meets the minimum requirement (70%).
    Returns (True, "Grade OK") if eligible, or (False, reason) if not."""
    grade_letter  = spec.get("grade_letter", "")
    grade_numeric = spec.get("grade_numeric")

    if grade_letter in FAILING_GRADES:
        return False, f"Failing grade ({grade_letter})"
    if grade_numeric is not None:
        try:
            if float(grade_numeric) < 70:
                return False, f"Grade {grade_numeric}% is below required 70%"
        except Exception:
            pass
    return True, "Grade OK"


# ── Similarity cache helpers ───────────────────────────────────────────────────

def check_cache(taibah_code: str, ext_code: str, institution: str):
    """Looks up a previously computed similarity score from the cache table.
    Returns the cached row if found, or None if not cached yet."""
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

def save_cache(taibah_code: str, ext_code: str, institution: str, sim_pct: float, decision: str):
    """Saves a computed similarity score to the cache table so it can be reused."""
    try:
        supabase.table("similarity_cache").upsert({
            "taibah_course_code":    taibah_code,
            "external_course_code":  ext_code,
            "external_institution":  institution,
            "similarity_percentage": sim_pct,
            "trs_decision":          decision,
        }).execute()
    except Exception:
        pass


# ── Database helpers ───────────────────────────────────────────────────────────

def get_taibah_courses(student_title: str = None):
    """Fetches Taibah courses from the database, filtered by domain if a title is given.
    If domain filtering finds matches, only those are returned (saves comparison time).
    If no domain match, falls back to returning all CS-related courses."""
    all_c = supabase.table("taibah_courses").select("*").eq("is_cs_related", True).execute().data or []

    if student_title:
        domain_courses = [c for c in all_c if same_domain(student_title, c.get("course_title_en", ""))]
        if domain_courses:
            print(f" Domain match — comparing against {len(domain_courses)} related course(s) only\n")
            courses = domain_courses
        else:
            print(f"  No domain match — comparing against all {len(all_c)} courses\n")
            courses = all_c
    else:
        courses = all_c

    # Fetch and attach content text for each course
    result = []
    for c in courses:
        rows = supabase.table("taibah_course_content").select("content_text").eq("course_code", c["course_code"]).execute().data or []
        content = "\n".join(r["content_text"] for r in rows if r.get("content_text"))
        if content:  # skip courses with no content — they can't be compared
            result.append({
                "code":         c["course_code"],
                "title":        c.get("course_title_en", ""),
                "desc":         c.get("general_description", ""),
                "content":      content,
                "credit_hours": c.get("credit_hours"),
            })

    print(f" Loaded {len(result)} Taibah course(s) for comparison\n")
    return result


# ── Main script ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student",   required=True, help="Path to student course spec JSON")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    args = parser.parse_args()

    # Read the student course spec JSON from disk
    spec           = json.loads(Path(args.student).read_text(encoding="utf-8"))
    s_code         = spec.get("course_code", "").strip().upper()
    s_title        = spec.get("course_title", "")
    s_desc         = spec.get("general_description", "")
    s_credit_hours = spec.get("credit_hours")
    s_institution  = spec.get("institution", "")

    # Build the content string from sections , same logic as main.py
    # This ensures local test scores match what the web app produces
    content_parts = []
    for sec in spec.get("content_sections", []):
        section_text = sec.get("content_text", "").strip()
        if not section_text:
            # Build from structured fields if content_text is missing
            parts     = []
            heading   = sec.get("heading", "")
            topics    = sec.get("topics", [])
            practical = sec.get("practical_topics", [])
            if heading:  parts.append(heading)
            if topics:   parts.append(", ".join(str(t) for t in topics))
            if practical: parts.append("Practical: " + ", ".join(str(t) for t in practical))
            section_text = ". ".join(filter(None, parts))
        else:
            # Append topics to existing content_text for richer comparison
            topics    = sec.get("topics", [])
            practical = sec.get("practical_topics", [])
            extra     = []
            if topics:    extra.append(", ".join(str(t) for t in topics))
            if practical: extra.append("Practical: " + ", ".join(str(t) for t in practical))
            if extra:
                section_text = section_text + "\n" + ". ".join(extra)
        if section_text.strip():
            content_parts.append(section_text)
    s_content = "\n".join(content_parts)

    print(f" Student course: {s_code} — {s_title}")
    print(f"   Credit hours: {s_credit_hours or 'N/A'}")
    print(f"   Institution:  {s_institution or 'N/A'}")
    print(f"   Content length: {len(s_content)} chars")
    print(f"   Threshold:    {args.threshold * 100}%")
    print()

    # Step 1: Make sure the student actually passed this course
    grade_ok, grade_msg = check_grade(spec)
    if not grade_ok:
        print(f" Grade check failed: {grade_msg}")
        print(f" Course is not eligible for equivalency.\n")
        return
    print(f" Grade check passed\n")

    # Step 2: Load Taibah courses , filtered by domain if possible
    taibah_courses = get_taibah_courses(student_title=s_title)
    if not taibah_courses:
        print("  No Taibah courses found for comparison.")
        return

    # Step 3: Rank Taibah courses by title similarity ,compare most likely matches first
    print(" Title similarity pre-ranking:")
    title_ranked = sorted(
        taibah_courses,
        key=lambda t: get_title_score(s_title, t["title"]),
        reverse=True
    )
    for t in title_ranked:
        tscore = get_title_score(s_title, t["title"])
        print(f"   {t['code']:<12} {t['title']:<40} Title sim: {round(tscore*100,1)}%")
    print()

    # Step 4-6: Credit check, cache lookup, and full similarity scoring
    print(f"{'Code':<12} {'Title':<40} {'Final':>7} {'Content':>9} {'Desc':>7} {'Credits':>9}  Decision")
    print("-" * 105)

    results = []
    for t in title_ranked:
        t_credit = t.get("credit_hours")

        # Skip if student has fewer credit hours than the Taibah course requires
        if s_credit_hours and t_credit:
            try:
                if float(s_credit_hours) < float(t_credit):
                    print(f"{t['code']:<12} {t['title'][:40]:<40}   SKIP (student {s_credit_hours}cr < Taibah {t_credit}cr)")
                    continue
            except Exception:
                pass

        # Check cache , reuse stored score if this pair was already compared
        cached = check_cache(t["code"], s_code, s_institution)
        if cached:
            sim_pct    = float(cached["similarity_percentage"])
            decision   = cached["trs_decision"]
            cr_display = f"{t_credit}cr" if t_credit else "N/A"
            icon       = "[+]" if decision == "Equivalent" else "[-]"
            print(f"{t['code']:<12} {t['title'][:40]:<40} "
                  f"{sim_pct:>6.1f}%  {'(cached)':>9}  {'':>5}  {cr_display:>7}   {icon} {decision}")
            results.append({
                "taibah_code":  t["code"],
                "taibah_title": t["title"],
                "final_%":      sim_pct,
                "content_%":    0,
                "desc_%":       0,
                "decision":     decision,
                "cached":       True,
            })
            continue

        # Run the full similarity computation for this course pair
        final, content_sim, desc_sim = weighted_similarity(
            t["content"], s_content,
            t["desc"],    s_desc,
            t["title"],   s_title,
            t["code"],    s_code,
        )

        # Apply domain boost (+3%) if both courses are in the same subject area
        if same_domain(s_title, t["title"]):
            final = min(round(final + 0.03, 4), 1.0)  # cap at 100%

        sim_pct    = round(final * 100, 2)
        decision   = "Equivalent" if final >= args.threshold else "Not Equivalent"
        cr_display = f"{t_credit}cr" if t_credit else "N/A"
        icon       = "[+]" if decision == "Equivalent" else "[-]"

        print(f"{t['code']:<12} {t['title'][:40]:<40} "
              f"{round(final*100,1):>6}%  "
              f"{round(content_sim*100,1):>7}%  "
              f"{round(desc_sim*100,1):>5}%  "
              f"{cr_display:>7}   "
              f"{icon} {decision}")

        # Save to cache so this pair doesn't get recomputed next time
        save_cache(t["code"], s_code, s_institution, sim_pct, decision)

        results.append({
            "taibah_code":  t["code"],
            "taibah_title": t["title"],
            "final_%":      sim_pct,
            "content_%":    round(content_sim * 100, 2),
            "desc_%":       round(desc_sim * 100, 2),
            "decision":     decision,
            "cached":       False,
        })

    if not results:
        print("\n  All courses were skipped due to credit hours mismatch.")
        return

    # Print the best matching course at the end
    best = max(results, key=lambda x: x["final_%"])
    cached_note = " (from cache)" if best.get("cached") else ""
    print(f"\n Best match: {best['taibah_code']} — {best['taibah_title']} ({best['final_%']}%){cached_note} -> {best['decision']}")


if __name__ == "__main__":
    main()

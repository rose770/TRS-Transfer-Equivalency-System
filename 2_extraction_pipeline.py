"""
AI Extraction Pipeline — Step 2
================================
Reads raw OCR text from ./ocr_outputs/ and uses OpenAI to extract
structured data into JSON files in ./json_outputs/.

Automatically detects document type from page 1:
  - "Course Specification" keywords  -> course_specification JSON
  - Arabic transcript keywords       -> transcript JSON

Input:  ./ocr_outputs/<pdf_name>/full_text.txt
Output: ./json_outputs/<pdf_name>_course_specification.json  OR
        ./json_outputs/<pdf_name>_transcript.json

Usage:
    python 2_extraction_pipeline.py
    python 2_extraction_pipeline.py --name specific_pdf_name
"""

import os
import sys
import json
import re
import time
import argparse
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

# Where the OCR output folders are stored (one folder per PDF)
OCR_DIR    = Path("./ocr_outputs")
# Where the final JSON files will be saved
OUTPUT_DIR = Path("./json_outputs")
# Using GPT-4o for the best extraction accuracy on academic documents
MODEL      = "gpt-4o"
# Skip non-CS course specs to save API credits , set to False to process everything
SKIP_NON_CS = True

# ── Keywords used for document type detection ─────────────────────────────────

# These words only appear in course specification documents, not transcripts
# We removed "course code" and "course title" because transcripts use those
# as table column headers which caused false positives before
COURSE_SPEC_KEYWORDS = [
    "course specification",
    "course content",
    "learning outcomes",
    "course learning outcomes",
    "clos",
    "توصيف المقرر",
    "مخرجات التعلم",
    "محتوى المقرر",
]

# These words are strong signals that the document is a student transcript.
# We check transcripts FIRST because transcripts have "course code" as a column
# header, which used to confuse the old detection logic into thinking it was a spec.
TRANSCRIPT_KEYWORDS = [
    # Arabic transcript phrases
    "السجل الأكاديمي",
    "سجل أكاديمي",
    "السجل الاكاديمي",
    "للطالب",
    # English transcript phrases , added to support English transcripts
    "academic transcript",
    "academic record",
    "end of academic record",
    "deanship of admission and registration",
    "cumulative",
    "quality point",
    "academic standing",
    "credit hours",
    "gpa",
    "attempted",
    "earned",
    "transcript",
]

# ── JSON Schemas sent to GPT-4o ───────────────────────────────────────────────

# This is the exact structure we want GPT-4o to fill in for course specs
COURSE_SPEC_SCHEMA = """
Extract into this exact JSON structure. Use null for missing fields.
Return a single object (not an array).

{
  "file_code": "string (e.g. TP-153, from header/cover page)",
  "college": "string",
  "department": "string",
  "institution": "string",
  "course_code": "string (e.g. PHYS 101)",
  "course_title": "string",
  "is_cs_related": boolean (true if the course is related to Computer Science, IT, Software Engineering, Programming, Networks, Databases, AI, Cybersecurity, or any computing field. false otherwise),
  "general_description": "string — full paragraph description of course",
  "content_sections": [
    {
      "heading": "string (chapter/topic/unit title)",
      "topics": ["list of theoretical sub-topics, or empty array"],
      "practical_topics": ["list of practical/lab topics, or empty array"],
      "content_text": "string — ALWAYS populated, used for similarity comparison. Build it as follows: start with the heading, then append theoretical topics if any, then append practical topics prefixed with 'Practical:'. Examples: (1) heading only → 'Introduction to Computers and Programming'. (2) topics only → 'Elementary Programming: Identifiers, Variables, Assignment Statements'. (3) both → 'Elementary Programming: Identifiers, Variables. Practical: Writing simple programs'. (4) same content in both theoretical and practical → 'Elementary Programming: Identifiers, Variables. Practical: Identifiers, Variables' — include both even if identical."
    }
  ]
}

Important:
- is_cs_related must be true or false (boolean), not a string
- content_text must ALWAYS be populated — never leave it null or empty
- There are two types of course content layouts:
  TYPE 1: Practical topics are inline under each week as "Lab: ..." — extract them directly into practical_topics of that section
  TYPE 2: There is a separate "List of Practical Topics" table at the bottom of the page — you MUST extract ALL items from this separate table into practical_topics. Match them to the closest theoretical section by topic name. If a practical topic does not match any theoretical section, add it as a new section with empty topics and the practical item in practical_topics.
- For TYPE 2 documents, do NOT ignore the separate practical topics table — it is just as important as the theoretical topics table
- content_text must include BOTH theoretical and practical content prefixed with 'Practical:' even if they are identical
- Only extract the fields listed above — nothing else
- Return a single object, not wrapped in an array
"""

# This is the exact structure we want GPT-4o to fill in for student transcripts
TRANSCRIPT_SCHEMA = """
Extract into this exact JSON structure. Use null for missing fields.
Return a single object (not an array).

{
  "student_info": {
    "student_name": "string — full name exactly as written",
    "student_id": "string — the university/academic ID number",
    "national_id": "string or null — the civil/national ID number if present",
    "institution": "string or null — ALWAYS in English even if the transcript is in Arabic (e.g. 'University of Tabuk' not 'جامعة تبوك'). Translate to English if needed.",
    "college": "string or null — ALWAYS in English even if the transcript is in Arabic. Translate to English if needed.",
    "major": "string or null — ALWAYS in English even if the transcript is in Arabic. Translate to English if needed.",
    "degree": "string or null",
    "student_status": "string or null",
    "print_date": "string or null"
  },
  "summary": {
    "cumulative_gpa": "string or null — the final cumulative GPA (e.g. 3.090)",
    "total_credit_hours": "string or null — total attempted/earned hours",
    "total_points": "string or null"
  },
  "courses": [
    {
      "semester": "string — semester name exactly as written (e.g. First Semester 2024/2025)",
      "course_code": "string — exact course code (e.g. CSCE 101, MATH 101)",
      "course_name": "string — course name ALWAYS in English. Translate from Arabic if needed (e.g. 'مقدمة في البرمجة' → 'Introduction to Programming')",
      "credit_hours": "string — individual course credit hours (e.g. 2, 3, 4) NOT the semester total",
      "grade_letter": "string — letter grade (e.g. A+, B, C+)",
      "grade_numeric": "string or null — numeric grade percentage if present (e.g. 95.00)",
      "grade_points": "string or null — weighted quality points (e.g. 12.00, 9.00)"
    }
  ]
}

Critical extraction rules:
- Extract ALL courses from ALL semesters — do not skip any semester or course
- credit_hours for each course is the INDIVIDUAL course hours (2, 3, or 4)
  NOT the semester total row (e.g. 18.00 total attempted)
- grade_points is the quality/weighted points like 12.00, 9.00 — NOT the percentage
- grade_numeric is the percentage like 85.00 — use null if not present
- institution, college, and major MUST always be in English — translate from Arabic if the transcript is in Arabic
- course_name must ALWAYS be in English — translate from Arabic if needed
- course_code must be preserved exactly as written — do NOT correct or translate
- Do NOT include semester summary rows as courses
"""

# ── System prompt for GPT-4o ──────────────────────────────────────────────────

# This is sent as the "system" role to set GPT-4o's behavior for extraction
EXTRACTION_SYSTEM = """You are a precise data extraction engine. You read raw OCR text
from scanned academic documents and extract structured information into JSON format.

CRITICAL RULES — never violate these:
- Extract ONLY what is explicitly present in the OCR text
- NEVER invent, guess, or fill in data from your training knowledge
- If a field is unclear, garbled, or missing → use null
- If a course appears garbled in the OCR → use null for that field, do NOT guess the course name
- institution, college, major, and course_name must ALWAYS be returned in English — translate from Arabic if needed
- Course codes must be preserved exactly as they appear — do not translate them
- Course codes must be copied exactly (e.g. PHYS 101, CSCE 102) — do NOT correct typos
- Return ONLY valid JSON — no preamble, no explanation, no markdown code fences
- If the OCR text is too blurry or garbled to read a value reliably → use null"""

# Used as a fallback when heuristic detection returns "unknown"
DETECTION_PROMPT = """Look at this text from page 1 of a scanned PDF document.
Identify whether this is:
1. A "course_specification" — a course specification document describing ONE course's content, learning outcomes, and topics
2. A "transcript" — a student academic transcript listing all courses and grades
3. "unknown" — neither of the above

Reply with ONLY one of these three words: course_specification, transcript, unknown

Page 1 text:
{page1_text}"""

# ── Helper functions ──────────────────────────────────────────────────────────

# CS keywords used to decide whether a course spec is relevant to our system
CS_KEYWORDS = [
    "computer", "computing", "programming", "software", "network",
    "database", "algorithm", "data structure", "artificial intelligence",
    "machine learning", "cybersecurity", "information technology",
    "information system", "web", "mobile", "cloud", "operating system",
    "compiler", "digital logic", "computer architecture", "csc", "cis",
    "ceng", "it ", "cs ", "swe", "sec", "net",
    "حاسب", "حاسوب", "برمجة", "شبكة", "قاعدة بيانات", "خوارزمية",
    "ذكاء اصطناعي", "أمن معلومات", "تقنية معلومات", "نظم معلومات",
    "هندسة برمجيات", "تطوير", "ويب",
]


def is_cs_related(pdf_name: str, page1_text: str) -> bool:
    """
    Checks if a course specification document is related to Computer Science.
    We look for CS keywords in the page text and also check for known
    CS department course code patterns (e.g. CSC, CENG, SWE).
    Does not rely on the filename — reads the actual content.
    """
    text_lower = page1_text.lower()
    # Check if any CS keyword appears in the text
    for kw in CS_KEYWORDS:
        if kw.lower() in text_lower:
            return True
    # Also check for CS course code patterns using a regex
    cs_code_pattern = re.compile(
        r"\b(CSC|CIS|CENG|SWE|NET|SEC|CID|CSCI|INFS|ITS|ICT|COMP|INFO|CYS|IS|IT)\s*\d+",
        re.IGNORECASE
    )
    if cs_code_pattern.search(page1_text):
        return True
    return False


def detect_doc_type_heuristic(pdf_name: str, page1_text: str) -> str:
    """
    Tries to figure out the document type using simple keyword matching.
    We check transcript keywords FIRST because transcripts contain phrases
    like "course code" and "course title" as table headers, which used to
    falsely match the course specification keywords in older versions.

    Also checks the filename prefix first — main.py saves transcripts as
    "transcript_<uuid>.pdf" so we can skip detection entirely in that case.

    Returns "transcript", "course_specification", or "unknown".
    """
    # Fastest check —,use filename prefix set by the web app upload handler
    if pdf_name.lower().startswith("transcript_"):
        return "transcript"

    text_lower = page1_text.lower()

    # Check for transcript keywords first , they are strong, unambiguous signals
    tr_score = sum(1 for kw in TRANSCRIPT_KEYWORDS if kw.lower() in text_lower)
    if tr_score >= 1:
        return "transcript"

    # Only check course spec keywords if transcript detection failed
    cs_score = sum(1 for kw in COURSE_SPEC_KEYWORDS if kw.lower() in text_lower)
    if cs_score >= 2:
        return "course_specification"

    # Not enough evidence either way ,fall back to LLM detection
    return "unknown"


def detect_doc_type_with_llm(client, page1_text: str) -> str:
    """
    Uses GPT-4o as a fallback when keyword-based detection returns "unknown".
    Sends the first 2000 characters of page 1 to the model and asks it to
    classify the document as course_specification, transcript, or unknown.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=20,  # we only need one word back
            messages=[{
                "role": "user",
                "content": DETECTION_PROMPT.format(page1_text=page1_text[:2000])
            }]
        )
        result = response.choices[0].message.content.strip().lower()
        if result in ("course_specification", "transcript"):
            return result
        return "unknown"
    except Exception as e:
        print(f"     LLM detection failed: {e}")
        return "unknown"


def extract_structured_data(client, full_text: str, doc_type: str) -> dict:
    """
    Sends the full OCR text to GPT-4o with a detailed schema and instructions.
    GPT-4o fills in the schema fields by reading the OCR text carefully.
    We retry up to 3 times if the response is not valid JSON.
    Returns the parsed dict, or an error dict if all retries fail.
    """
    # Choose the right schema and instructions based on document type
    if doc_type == "course_specification":
        schema = COURSE_SPEC_SCHEMA
        instruction = """Extract all course specification data from this OCR text into the JSON schema provided.

CRITICAL RULES FOR content_sections:
- Read the ENTIRE OCR text carefully — do not stop early
- Look for TWO types of content tables:
  TYPE 1: A single table where each row has a topic AND a "Lab:" practical inline — extract directly
  TYPE 2: TWO separate tables — one for theoretical topics, one titled "List of Topics (Practical Aspects)" or similar
- For TYPE 2 documents:
  * Extract ALL rows from the theoretical table as headings
  * Extract ALL rows from the practical table
  * Try to match each practical to the closest theoretical topic by subject
  * If a practical item does NOT clearly match any theoretical topic, add it as a NEW section with empty topics[] and the practical item in practical_topics[]
  * Do NOT skip any practical item — every row from the practical table must appear somewhere
- content_text MUST always be populated — combine heading + topics + "Practical: " + practical_topics
- If practical_topics exist, content_text MUST include "Practical: ..." at the end
- Do NOT leave content_sections empty or incomplete — extract ALL rows from ALL tables"""
    else:
        schema = TRANSCRIPT_SCHEMA
        instruction = """Extract all student transcript data from this OCR text into the JSON schema provided.

CRITICAL RULES:
- Extract ALL courses from ALL semesters — never skip any
- Each course row has: course_code, course_name, grade_letter, credit_hours, quality_points
- credit_hours is the INDIVIDUAL course value (2, 3, or 4) — NOT the semester total row
- Do NOT include the semester summary row (Attempted / Earned / GPA row) as a course
- For each semester, the semester name goes in the "semester" field of each course in that semester
- grade_numeric may be null for English transcripts that only show letter grades
- institution, college, and major MUST be in English — translate from Arabic if needed
- course_name must ALWAYS be in English — translate from Arabic if needed
- Preserve the exact course code as written"""

    # Build the full prompt combining instructions, schema, and OCR text
    prompt = f"""{instruction}

JSON Schema to populate:
{schema}

--- RAW OCR TEXT START ---
{full_text}
--- RAW OCR TEXT END ---

Return ONLY the populated JSON object. No markdown, no explanation."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=16000,
                response_format={"type": "json_object"},  # force JSON output mode
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw).strip()
            return json.loads(raw)

        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"     JSON parse error attempt {attempt+1}: {e}. Retrying...")
                time.sleep(3)
            else:
                print(f"    Could not parse JSON after {max_retries} attempts")
                return {"_extraction_error": str(e), "_raw_response": raw}

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"     API error attempt {attempt+1}: {e}. Retrying in 5s...")
                time.sleep(5)
            else:
                return {"_api_error": str(e)}


def check_if_cs_related(client, page1_text: str) -> bool:
    """
    Quick check to decide if a course spec is CS-related.
    First tries keyword matching for speed, then falls back to asking GPT-4o
    if the keywords aren't conclusive.
    """
    cs_keywords = [
        "csc", "cis", "cs ", "software", "programming", "network",
        "database", "artificial intelligence", "cybersecurity", "computing",
        "information technology", "computer", "data structure", "algorithm",
        "operating system", "web ", "mobile", "cloud", "machine learning",
    ]
    text_lower = page1_text.lower()
    # Fast path , keyword match
    if any(kw in text_lower for kw in cs_keywords):
        return True
    # Slow path ,ask GPT-4o if keywords didn't match
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=5,
            messages=[{
                "role": "user",
                "content": f"""Is this course related to Computer Science, IT, Programming, Networks, Databases, AI, or Cybersecurity?
Reply with only: yes or no

Course info:
{page1_text[:500]}"""
            }]
        )
        answer = response.choices[0].message.content.strip().lower()
        return answer.startswith("yes")
    except Exception:
        return True  # default to True if the check fails ,better to process than miss


def process_ocr_output(ocr_dir: Path, client) -> bool:
    """
    Processes one OCR output folder — reads the full text, detects the document
    type, sends it to GPT-4o for structured extraction, and saves the result as JSON.
    Returns True on success, False on failure.
    """
    pdf_name       = ocr_dir.name
    full_text_file = ocr_dir / "full_text.txt"
    page1_file     = ocr_dir / "page_001.txt"

    if not full_text_file.exists():
        print(f"     No full_text.txt in {ocr_dir}")
        return False

    print(f"\n Processing: {pdf_name}")

    full_text  = full_text_file.read_text(encoding="utf-8")
    # Use just page 1 for detection —,faster than reading the whole document
    page1_text = page1_file.read_text(encoding="utf-8") if page1_file.exists() else full_text[:2000]

    # Detect document type , heuristic first, LLM as fallback
    print("    Detecting document type...", end="", flush=True)
    doc_type = detect_doc_type_heuristic(pdf_name, page1_text)
    if doc_type == "unknown":
        # Heuristic wasn't confident enough , ask GPT-4o
        doc_type = detect_doc_type_with_llm(client, page1_text)
    print(f" -> {doc_type}")

    if doc_type == "unknown":
        print("     Could not determine document type. Saving as 'unknown'.")

    # Skip non-CS course specs to avoid wasting API credits
    if SKIP_NON_CS and doc_type == "course_specification":
        if not is_cs_related(pdf_name, page1_text):
            print("     Skipping — not a CS-related course spec")
            return True

    # Send the full OCR text to GPT-4o for structured data extraction
    print("    Extracting structured data...", end="", flush=True)
    data = extract_structured_data(client, full_text, doc_type)
    print(f"  ({len(json.dumps(data))} chars)")

    # Add metadata so we can trace where this JSON came from
    data["_metadata"] = {
        "source_pdf_name": pdf_name,
        "document_type":   doc_type,
        "ocr_output_dir":  str(ocr_dir),
        "extracted_by":    "2_extraction_pipeline.py",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"{pdf_name}_{doc_type}.json"
    out_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"    Saved: {out_file}")
    return True


# ── Main entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI Extraction Pipeline — OCR text -> structured JSON via OpenAI")
    parser.add_argument("--dir",  type=str, default=str(OCR_DIR))
    parser.add_argument("--name", type=str, help="Process only this OCR folder name")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(" No API key found.")
        print("   Add OPENAI_API_KEY=sk-... to your .env file")
        sys.exit(1)

    client   = OpenAI(api_key=api_key)
    ocr_root = Path(args.dir)

    if not ocr_root.exists():
        print(f" OCR directory not found: {ocr_root}")
        sys.exit(1)

    if args.name:
        # Single folder mode — process just one specific OCR output
        dirs = [ocr_root / args.name]
        if not dirs[0].exists():
            print(f" Not found: {dirs[0]}")
            sys.exit(1)
    else:
        # Batch mode — process all folders in the OCR output directory
        dirs = [d for d in sorted(ocr_root.iterdir()) if d.is_dir()]
        if not dirs:
            print(f"  No OCR output directories found in {ocr_root}")
            print("   Run step 1 first: python 1_ocr_pipeline.py")
            sys.exit(0)

    print(f" Extraction Pipeline — {len(dirs)} document(s) to process")

    success          = 0
    seen_course_codes = set()  # track course codes to avoid re-extracting duplicates

    for d in dirs:
        meta_file = d / "metadata.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if meta.get("doc_type") == "course_spec":
                    # Check if we already extracted this course code before
                    json_outputs = list(OUTPUT_DIR.glob(f"{d.name}_*.json"))
                    for jf in json_outputs:
                        try:
                            jdata       = json.loads(jf.read_text(encoding="utf-8"))
                            course_code = jdata.get("course_code", "").strip().upper().replace(" ", "")
                            if course_code and course_code in seen_course_codes:
                                print(f"\n {d.name}")
                                print(f"     Duplicate course spec ({course_code}) — skipping extraction")
                                success += 1
                                break
                            if course_code:
                                seen_course_codes.add(course_code)
                        except Exception:
                            pass
                    else:
                        if process_ocr_output(d, client):
                            success += 1
                    continue
            except Exception:
                pass

        if process_ocr_output(d, client):
            success += 1

    print(f"\n Done! {success}/{len(dirs)} documents extracted.")
    print(f"   JSON outputs -> {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

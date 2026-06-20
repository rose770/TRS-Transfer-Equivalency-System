"""
OCR Pipeline — Step 1
=====================
Converts scanned PDF pages to raw text.
Automatically routes documents to the best OCR engine:

  Transcript  → GPT-4o Vision (handles RTL Arabic tables perfectly)
  Course spec → PaddleOCR (fast, local, works well for English docs)

Install:
    pip install paddlepaddle paddleocr numpy pdf2image Pillow opencv-python openai python-dotenv
    brew install poppler          # macOS
    apt-get install poppler-utils # Ubuntu/Debian

Usage:
    python 1_ocr_pipeline.py                        # all PDFs in ./input_pdfs/
    python 1_ocr_pipeline.py --file path/to/doc.pdf
    python 1_ocr_pipeline.py --dpi 200
    python 1_ocr_pipeline.py --lang ar              # force language
    python 1_ocr_pipeline.py --lang en
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image

INPUT_DIR = Path("./input_pdfs")
OUTPUT_DIR = Path("./ocr_outputs")
DPI = 100
COOLDOWN = 15
FILE_COOLDOWN = 120
SKIP_NON_CS = True
BG_VARIANCE_THRESHOLD = 15

ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')

TRANSCRIPT_KEYWORDS = [
    "السجل الأكاديمي",
    "سجل أكاديمي",
    "السجل الاكاديمي",
    "للطالب",
]

COURSE_SPEC_KEYWORDS = [
    "course specification",
    "course title",
    "course code",
    "learning outcomes",
    "bachelor",
]

CS_KEYWORDS = [
    "computer", "computing", "programming", "software", "network",
    "database", "algorithm", "data structure", "artificial intelligence",
    "machine learning", "cybersecurity", "information technology",
    "information system", "web", "mobile", "cloud", "operating system",
    "compiler", "digital logic", "computer architecture", "csc", "cis",
    "ceng", "it ", "cs ", "swe", "sec", "net",
    "حاسب", "حاسوب", "برمجة", "شبكة", "قاعدة بيانات", "خوارزمية",
    "ذكاء اصطناعي", "أمن معلومات", "تقنية معلومات", "نظم معلومات",
    "هندسة برمجيات", "تطوير", "ويب", "خدمة", "معالج",
]


def has_complex_background(pil_image) -> bool:
    try:
        import cv2
        import numpy as np
        img = np.array(pil_image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        sample = gray[h-300:h-100, 50:300]
        variance = sample.std()
        return variance > BG_VARIANCE_THRESHOLD
    except Exception:
        return False


def preprocess_image(pil_image):
    try:
        import cv2
        import numpy as np
        from PIL import Image

        img = np.array(pil_image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cleaned = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 15
        )
        cleaned = cv2.fastNlMeansDenoising(cleaned, h=10)
        cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(cleaned_rgb)
    except ImportError:
        print("     opencv-python not installed — skipping preprocessing.")
        return pil_image
    except Exception as e:
        print(f"     Preprocessing failed: {e} — using original image")
        return pil_image


def preprocess_for_gpt4o(pil_image, dpi: int, is_first_page: bool = False):
    try:
        import cv2
        import numpy as np
        
        img = np.array(pil_image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        if is_first_page:
            h, w = sharpened.shape
            header_region = sharpened[0:int(h*0.25), :]
            header_enhanced = cv2.equalizeHist(header_region)
            sharpened[0:int(h*0.25), :] = header_enhanced
        
        processed = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
        scale = 1.5 if is_first_page else 1.2
        h, w = processed.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        return Image.fromarray(processed)
    except ImportError:
        print("     opencv-python not installed — skipping preprocessing.")
        return pil_image
    except Exception as e:
        print(f"     Preprocessing warning: {e} — using original")
        return pil_image


def extract_header_with_gpt4o(pil_image, api_key: str) -> dict:
    import base64
    import io
    from openai import OpenAI
    
    width, height = pil_image.size
    header_crop = pil_image.crop((0, 0, width, int(height * 0.3)))
    
    buf = io.BytesIO()
    header_crop.save(buf, format="JPEG", quality=98)
    img_b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    
    client = OpenAI(api_key=api_key)
    
    prompt = """Extract ONLY the following header information from this academic transcript (السجل الأكاديمي للطالب).

Look for these Arabic labels and extract the values that follow them:

IMPORTANT: The university name (الجامعة) and college name (الكلية) appear at the TOP of the page, before the student information.

Return as JSON format with these exact keys (use null if not found):
{
    "university": "extracted university name in Arabic",
    "college": "extracted college name in Arabic",
    "student_name": "extracted student name",
    "academic_id": "الرقم الأكاديمي value",
    "national_id": "السجل المدني value",
    "major": "التخصص value",
    "degree": "الدرجة العلمية value"
}

Output ONLY the JSON, no explanatory text."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=500,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                            "detail": "high"
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except json.JSONDecodeError:
        return {"error": response.choices[0].message.content, "raw": True}
    except Exception as e:
        return {"error": str(e)}


def ocr_page_with_gpt4o(pil_image, api_key: str, is_first_page: bool = False, header_info: dict = None) -> str:
    import base64
    import io
    from openai import OpenAI

    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=92)
    img_b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    client = OpenAI(api_key=api_key)

    if is_first_page:
        prompt = """This is the FIRST PAGE of a student academic transcript (السجل الأكاديمي للطالب).

**CRITICAL: Extract ALL text from the TOP of the page FIRST, then extract the tables.**

Step 1 - HEADER INFORMATION (appears at the very top of the page, BEFORE any tables):
- University name (الجامعة) — usually a single line like "جامعة تبوك"
- College name (الكلية) — usually a single line like "الكلية الجامعية بأملح"
- Student name (الاسم)
- Academic ID (الرقم الأكاديمي)
- National ID (السجل المدني)
- Major (التخصص)
- Degree (الدرجة العلمية)
- Study type (نوع الدراسة)
- Classification code (رمز التصنيف / المستوى)

Step 2 - TABLE DATA (course information):
For each course, extract:
- Course code (رمز المقرر)
- Course name (اسم المقرر)
- Credit hours (الساعات)
- Grade/points
- Any other numbers

Step 3 - SEMESTER SUMMARY:
- Semester GPA (المعدل الفصلي)
- Cumulative GPA (المعدل التراكمي)

RULES:
- Preserve Arabic text exactly in Arabic script
- Preserve English text exactly
- For tables, read RIGHT TO LEFT for Arabic content
- Use | to separate table columns
- Extract EVERY row from EVERY table
- DO NOT skip header information just because it looks like formatting

Output ONLY the transcribed text, nothing else."""
    else:
        prompt = """This is a student academic transcript page (not first page).
Extract ALL text exactly as it appears including:
- Course codes (رمز المقرر)
- Course names (اسم المقرر)
- Credit hours (الساعات)
- Grades and points
- Semester information

Preserve Arabic text exactly in Arabic script.
Use | to separate table columns.
Output ONLY the transcribed text, nothing else."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=4096,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"GPT-4o Vision failed: {e}")


def detect_doc(pdf_path: Path):
    if pdf_path.stem.startswith("transcript_"):
        print("   Detected as transcript (filename prefix)")
        return "transcript", "ar", None

    if pdf_path.stem.startswith("spec_"):
        print("   Detected as course_spec (filename prefix)")
        return "course_spec", "en", True

    print("    Detecting document language from page 1...", end="", flush=True)

    try:
        pages = convert_from_path(str(pdf_path), dpi=100, first_page=1, last_page=1)
    except Exception as e:
        print(f"   Could not render page 1: {e}. Defaulting to 'arabic'.")
        return "ar"

    page1 = pages[0]

    try:
        from paddleocr import PaddleOCR
        import numpy as np
        ocr_detect = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        det_results = ocr_detect.ocr(np.array(page1), cls=False)
        sample_text = ""
        if det_results and det_results[0]:
            for line in det_results[0]:
                if line and len(line) >= 2 and line[1]:
                    sample_text += line[1][0] + " "
    except Exception:
        sample_text = ""

    for kw in TRANSCRIPT_KEYWORDS:
        if kw in sample_text:
            print(" → transcript")
            return "transcript", "ar", None

    arabic_chars = len(ARABIC_PATTERN.findall(sample_text))
    total_chars = len(sample_text.strip())

    if total_chars == 0:
        print(" → transcript (no text detected, defaulting)")
        return "transcript", "ar", None

    arabic_ratio = arabic_chars / max(total_chars, 1)
    text_lower = sample_text.lower()
    is_english_course_spec = any(k in text_lower for k in COURSE_SPEC_KEYWORDS + ["department of"])

    if is_english_course_spec and arabic_ratio < 0.3:
        cs = is_cs_course_spec(pdf_path, sample_text)
        print(f" → course_spec (English) | CS: {cs}")
        return "course_spec", "en", cs
    elif arabic_ratio >= 0.3:
        cs = is_cs_course_spec(pdf_path, sample_text)
        print(f" → course_spec (Arabic) | CS: {cs}")
        return "course_spec", "ar", cs
    else:
        if any(c in pdf_path.name for c in "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"):
            print(" → transcript (Arabic filename)")
            return "transcript", "ar", None
        cs = is_cs_course_spec(pdf_path, sample_text)
        print(f" → course_spec (default) | CS: {cs}")
        return "course_spec", "en", cs


def is_cs_course_spec(pdf_path: Path, sample_text: str) -> bool:
    text_lower = sample_text.lower()
    for kw in CS_KEYWORDS:
        if kw.lower() in text_lower:
            return True
    import re
    cs_code_pattern = re.compile(
        r"\b(CSC|CIS|CENG|SWE|NET|SEC|CID|CSCI|INFS|ITS|ICT|COMP|INFO|CYS|IS|IT)\s*\d+",
        re.IGNORECASE
    )
    if cs_code_pattern.search(sample_text):
        return True
    return False


def format_tables(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        cells = re.split(r" {2,}", line.strip())
        if len(cells) >= 3:
            lines.append(" | ".join(c.strip() for c in cells if c.strip()))
        else:
            lines.append(line)
    return "\n".join(lines)


def _paddle_ocr_page(ocr, img, do_preprocess: bool = False) -> str:
    import numpy as np
    if do_preprocess:
        img = preprocess_image(img)
    result = ocr.ocr(np.array(img), cls=True)
    lines = []
    if result and result[0]:
        for line in result[0]:
            if line and len(line) >= 2 and line[1]:
                text = line[1][0]
                score = line[1][1]
                if score > 0.5:
                    lines.append(text)
    return "\n".join(lines)


def extract_course_code(pdf_path: Path):
    try:
        pages = convert_from_path(str(pdf_path), dpi=72, first_page=1, last_page=1)
        from paddleocr import PaddleOCR
        import numpy as np
        ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        result = ocr.ocr(np.array(pages[0]), cls=False)
        text = ""
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2 and line[1]:
                    text += line[1][0] + " "
        import re
        match = re.search(r"\b([A-Z]{2,4}\s*\d{3,4})\b", text)
        if match:
            return match.group(1).strip().upper().replace(" ", "")
    except Exception:
        pass
    return None


def process_pdf(pdf_path: Path, forced_lang, dpi: int) -> bool:
    pdf_name = pdf_path.stem
    out_dir = OUTPUT_DIR / pdf_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n Processing: {pdf_path.name}")

    if forced_lang:
        doc_type = "transcript" if forced_lang == "ar" else "course_spec"
        lang = forced_lang
        is_cs = None
    else:
        doc_type, lang, is_cs = detect_doc(pdf_path)

    if SKIP_NON_CS and doc_type == "course_spec" and is_cs is False:
        print(f"     Skipping — not a CS-related course spec")
        return True

    print(f"    Doc type: {doc_type} | Lang: {lang}")

    try:
        import subprocess
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)], capture_output=True, text=True
        )
        page_count = 1
        for line in result.stdout.splitlines():
            if line.startswith("Pages:"):
                page_count = int(line.split(":")[1].strip())
                break
    except Exception:
        try:
            pages = convert_from_path(str(pdf_path), dpi=72)
            page_count = len(pages)
            del pages
        except Exception as e:
            print(f"    Failed to read PDF: {e}")
            return False

    print(f"    {page_count} pages found")
    import time

    if doc_type == "transcript":
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print(" OPENAI_API_KEY not set in .env file")
            return False

        transcript_dpi = max(dpi, 200)
        print(f"    Using DPI {transcript_dpi} for transcript")
        print("    Using GPT-4o Vision for transcript OCR...")

        header_info = None
        skipped = 0

        for i in range(page_count):
            page_file = out_dir / f"page_{i+1:03d}.txt"

            if page_file.exists():
                skipped += 1
                continue

            if skipped:
                print(f"     Skipped {skipped} already-done page(s)")
                skipped = 0

            print(f"     Rendering page {i+1}/{page_count}...", end="", flush=True)

            if i == 0:
                page_img = convert_from_path(
                    str(pdf_path), dpi=300,
                    first_page=1, last_page=1
                )[0]
            else:
                page_img = convert_from_path(
                    str(pdf_path), dpi=transcript_dpi,
                    first_page=i+1, last_page=i+1
                )[0]

            print(f" rendered", end="", flush=True)

            is_first_page = (i == 0)
            print(f" preprocessing...", end="", flush=True)
            page_img = preprocess_for_gpt4o(page_img, transcript_dpi, is_first_page)

            if is_first_page and api_key:
                print(f" header-extract...", end="", flush=True)
                try:
                    header_info = extract_header_with_gpt4o(page_img, api_key)
                    (out_dir / "header_info.json").write_text(
                        json.dumps(header_info, indent=2, ensure_ascii=False),
                        encoding="utf-8"
                    )
                    if header_info.get("university"):
                        print(f"\n     ✓ Found university: {header_info['university']}")
                    if header_info.get("college"):
                        print(f"     ✓ Found college: {header_info['college']}")
                except Exception as e:
                    print(f" header-extract warning: {e}", end="", flush=True)

            print(f" GPT-4o Vision...", end="", flush=True)

            try:
                page_text = ocr_page_with_gpt4o(
                    page_img, api_key,
                    is_first_page=is_first_page,
                    header_info=header_info
                )

                if is_first_page and header_info and header_info.get("university"):
                    header_block = f"الجامعة : {header_info.get('university', '')}\n"
                    header_block += f"الكلية : {header_info.get('college', '')}\n"
                    header_block += f"الاسم : {header_info.get('student_name', '')}\n"
                    header_block += f"الرقم الأكاديمي : {header_info.get('academic_id', '')}\n"
                    header_block += f"السجل المدني : {header_info.get('national_id', '')}\n"
                    header_block += f"التخصص : {header_info.get('major', '')}\n"
                    header_block += f"الدرجة العلمية : {header_info.get('degree', '')}\n\n"

                    if header_info.get('university') not in page_text:
                        page_text = header_block + page_text

            except Exception as e:
                print(f"\n    GPT-4o Vision failed: {e}")
                return False

            page_file.write_text(page_text, encoding="utf-8")
            del page_img
            print(f"  ({len(page_text)} chars)")
            time.sleep(2)

        if skipped:
            print(f"     Skipped {skipped} already-done page(s)")

    else:
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            print(" PaddleOCR not installed. pip install paddlepaddle paddleocr")
            return False

        print(f"    Loading PaddleOCR model [lang={lang}]...")
        ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        do_preprocess = False

        skipped = 0
        for i in range(page_count):
            page_file = out_dir / f"page_{i+1:03d}.txt"
            if page_file.exists():
                skipped += 1
                continue

            if skipped:
                print(f"     Skipped {skipped} already-done page(s)")
                skipped = 0

            print(f"     Rendering page {i+1}/{page_count}...", end="", flush=True)
            page_img = convert_from_path(str(pdf_path), dpi=dpi,
                                         first_page=i+1, last_page=i+1)[0]
            print(f" rendered   PaddleOCR...", end="", flush=True)

            page_text = _paddle_ocr_page(ocr, page_img, do_preprocess=do_preprocess)
            page_text = format_tables(page_text)
            page_file.write_text(page_text, encoding="utf-8")
            del page_img
            print(f"  ({len(page_text)} chars)")
            time.sleep(COOLDOWN)

        if skipped:
            print(f"     Skipped {skipped} already-done page(s)")

    all_texts = []
    for i in range(page_count):
        pf = out_dir / f"page_{i+1:03d}.txt"
        all_texts.append(pf.read_text(encoding="utf-8") if pf.exists() else "")

    sep = "\n\n" + "=" * 60 + "\n\n"
    full_text = sep.join(f"PAGE {i+1}\n{'='*60}\n{t}" for i, t in enumerate(all_texts))
    (out_dir / "full_text.txt").write_text(full_text, encoding="utf-8")

    meta = {
        "pdf_name": pdf_path.name,
        "pdf_path": str(pdf_path),
        "page_count": page_count,
        "dpi": dpi,
        "doc_type": doc_type,
        "lang": lang,
        "forced_lang": forced_lang,
        "output_dir": str(out_dir),
        "pages": [str(out_dir / f"page_{i+1:03d}.txt") for i in range(page_count)],
        "full_text": str(out_dir / "full_text.txt"),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"    Saved to: {out_dir}  [lang={lang}]")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="OCR Pipeline — auto-detects Arabic vs English per document"
    )
    parser.add_argument("--file", type=str,
                        help="Process a single PDF file")
    parser.add_argument("--dpi", type=int, default=DPI,
                        help=f"Render DPI (default: {DPI})")
    parser.add_argument("--lang", choices=["ar", "en"],
                        help="Force OCR language (skip auto-detection)")
    parser.add_argument("--input-dir", type=str, default=str(INPUT_DIR))
    args = parser.parse_args()

    if args.file:
        pdf_paths = [Path(args.file)]
        if not pdf_paths[0].exists():
            print(f" File not found: {args.file}")
            sys.exit(1)
    else:
        input_dir = Path(args.input_dir)
        input_dir.mkdir(parents=True, exist_ok=True)
        pdf_paths = sorted(input_dir.rglob("*.pdf"))
        if not pdf_paths:
            print(f"  No PDFs found in {input_dir.resolve()}")
            sys.exit(0)

    print(f" OCR Pipeline (Modified - Header Extraction Fix)  |  {len(pdf_paths)} PDF(s)")
    if args.lang:
        print(f"     Language forced to: {args.lang}")

    import time
    success = 0
    seen_course_codes = set()

    for idx, p in enumerate(pdf_paths):
        print(f"\n{'='*60}")
        print(f" File {idx+1}/{len(pdf_paths)}: {p.name}")
        print(f"{'='*60}")

        if not args.lang or args.lang != "ar":
            course_code = extract_course_code(p)
            if course_code and course_code in seen_course_codes:
                print(f"     Duplicate course spec detected ({course_code}) — skipping")
                continue
            if course_code:
                seen_course_codes.add(course_code)

        if process_pdf(p, args.lang, args.dpi):
            success += 1

        if idx < len(pdf_paths) - 1:
            print(f"\n  Cooling down for {FILE_COOLDOWN}s before next file...")
            for remaining in range(FILE_COOLDOWN, 0, -5):
                print(f"   {remaining}s remaining...", end="\r")
                time.sleep(min(5, remaining))
            print("    Ready for next file        ")

    print(f"\n Done! {success}/{len(pdf_paths)} PDFs processed.")
    print(f"   OCR outputs -> {OUTPUT_DIR.resolve()}")
    print(f"\n Next: python 2_extraction_pipeline.py")


if __name__ == "__main__":
    main()

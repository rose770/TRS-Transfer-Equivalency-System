"""
run_pipeline.py — Convenience wrapper
======================================
Runs step 1 (OCR) and step 2 (AI extraction) back-to-back.

Usage:
    python run_pipeline.py path/to/document.pdf          # single file, auto-detect language
    python run_pipeline.py path/to/document.pdf --lang arabic  # force language
    python run_pipeline.py path/to/document.pdf --lang en
    python run_pipeline.py --batch                       # all PDFs in ./input_pdfs/
    python run_pipeline.py --batch --dpi 250             # higher DPI
"""

import sys
import shutil
import argparse
import subprocess
from pathlib import Path


def run(cmd, cwd):
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        print(f"❌ Step failed: {' '.join(str(c) for c in cmd)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run full OCR + Extraction pipeline")
    parser.add_argument("pdf",    nargs="?", help="Path to a single PDF file")
    parser.add_argument("--batch", action="store_true", help="Process all PDFs in ./input_pdfs/")
    parser.add_argument("--dpi",   type=int, default=200, help="OCR render DPI (default: 200)")
    parser.add_argument("--lang",  choices=["arabic", "en"], help="Force OCR language (skip auto-detect)")
    args = parser.parse_args()

    if not args.pdf and not args.batch:
        parser.print_help()
        sys.exit(1)

    base = Path(__file__).parent

    if args.pdf:
        pdf_path = Path(args.pdf).resolve()
        if not pdf_path.exists():
            print(f"❌ File not found: {pdf_path}")
            sys.exit(1)

        # Copy into input_pdfs/ so step 1 can find it
        input_dir = base / "input_pdfs"
        input_dir.mkdir(exist_ok=True)
        dest = input_dir / pdf_path.name
        if dest != pdf_path:
            shutil.copy2(pdf_path, dest)
            print(f"📥 Copied to: {dest}")

        # Build step 1 command
        step1_cmd = [sys.executable, str(base / "1_ocr_pipeline.py"),
                     "--file", str(dest), "--dpi", str(args.dpi)]
        if args.lang:
            step1_cmd += ["--lang", args.lang]

        # Build step 2 command
        step2_cmd = [sys.executable, str(base / "2_extraction_pipeline.py"),
                     "--name", dest.stem]

        print("\n" + "=" * 60)
        print("STEP 1 — OCR")
        print("=" * 60)
        run(step1_cmd, base)

        print("\n" + "=" * 60)
        print("STEP 2 — AI EXTRACTION")
        print("=" * 60)
        run(step2_cmd, base)

    elif args.batch:
        step1_cmd = [sys.executable, str(base / "1_ocr_pipeline.py"),
                     "--dpi", str(args.dpi)]
        if args.lang:
            step1_cmd += ["--lang", args.lang]

        step2_cmd = [sys.executable, str(base / "2_extraction_pipeline.py")]

        print("\n" + "=" * 60)
        print("STEP 1 — OCR (batch)")
        print("=" * 60)
        run(step1_cmd, base)

        print("\n" + "=" * 60)
        print("STEP 2 — AI EXTRACTION (batch)")
        print("=" * 60)
        run(step2_cmd, base)

    print("\n🎉 Pipeline complete!")
    print(f"   JSON results → {(base / 'json_outputs').resolve()}")


if __name__ == "__main__":
    main()


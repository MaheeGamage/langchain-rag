#!/usr/bin/env python3
"""Recursively convert PDFs to Markdown while preserving folder structure."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from pymupdf4llm import to_markdown


@dataclass
class ConversionResult:
    converted: int = 0
    skipped: int = 0
    failed: int = 0


def find_pdf_files(root_dir: Path) -> list[Path]:
    """Return all PDF files below root_dir."""
    return [p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]


def build_output_path(pdf_path: Path, input_root: Path, output_root: Path) -> Path:
    """Map an input PDF path to its mirrored Markdown output path."""
    relative_pdf = pdf_path.relative_to(input_root)
    return (output_root / relative_pdf).with_suffix(".md")


def convert_pdf(pdf_path: Path, output_md_path: Path, overwrite: bool, dry_run: bool) -> str:
    """Convert one PDF to Markdown and write to output_md_path."""
    if output_md_path.exists() and not overwrite:
        return "skipped"

    if dry_run:
        return "converted"

    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep conversion portable by disabling OCR; this avoids hard dependency on
    # system Tesseract language packs while still extracting embedded PDF text.
    markdown = to_markdown(str(pdf_path), use_ocr=False)
    output_md_path.write_text(markdown, encoding="utf-8")
    return "converted"


def run_conversion(input_dir: Path, output_dir: Path, overwrite: bool, dry_run: bool) -> ConversionResult:
    """Convert all PDFs below input_dir and mirror output structure under output_dir."""
    pdf_files = find_pdf_files(input_dir)
    result = ConversionResult()

    if not pdf_files:
        print(f"No PDF files found under: {input_dir}")
        return result

    print(f"Found {len(pdf_files)} PDF file(s) under: {input_dir}")
    print(f"Writing Markdown output under: {output_dir}")

    for pdf_path in pdf_files:
        output_md_path = build_output_path(pdf_path, input_dir, output_dir)
        try:
            status = convert_pdf(
                pdf_path=pdf_path,
                output_md_path=output_md_path,
                overwrite=overwrite,
                dry_run=dry_run,
            )
            if status == "converted":
                result.converted += 1
                print(f"[ok] {pdf_path} -> {output_md_path}")
            else:
                result.skipped += 1
                print(f"[skip] {pdf_path} (already exists)")
        except Exception as exc:
            result.failed += 1
            print(f"[error] {pdf_path}: {exc}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively convert every PDF under an input directory to Markdown "
            "while preserving folder structure."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory to scan recursively for PDF files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where mirrored Markdown files will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Markdown files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")

    result = run_conversion(
        input_dir=input_dir,
        output_dir=output_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    print("\nSummary")
    print(f"- converted: {result.converted}")
    print(f"- skipped:   {result.skipped}")
    print(f"- failed:    {result.failed}")

    if result.failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

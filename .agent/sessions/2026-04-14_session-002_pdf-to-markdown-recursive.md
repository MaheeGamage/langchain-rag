# Session: 2026-04-14 #002

## Goal
Add a script to recursively convert scientific-paper PDFs into Markdown files while preserving the folder structure under a chosen output root.

## Prompts Summary
- User asked for a Python script in `experimentation/pdf-to-markdown`.
- User requested support for any input path with recursive traversal and mirrored output structure.
- User provided a target source directory: `knowledge_ingestion/content/v2/content/orig_paper`.

## Actions Taken
- Added `pymupdf4llm` to project dependencies using Poetry.
- Created `experimentation/pdf-to-markdown/convert_pdfs_to_markdown.py`:
  - Recursively discovers `.pdf` files below the input directory.
  - Maps each PDF path to a mirrored `.md` path below the output directory.
  - Supports `--overwrite` and `--dry-run` flags.
  - Continues on per-file failures and prints a conversion summary.
- Updated conversion call to disable OCR (`use_ocr=False`) so conversion does not require system Tesseract language packs.
- Added `experimentation/pdf-to-markdown/README.md` with usage examples.
- Ran the converter against `knowledge_ingestion/content/v2/content/orig_paper` and generated 4 Markdown files under `knowledge_ingestion/content/v2/content_md/orig_paper`.

## Outcome
A reusable CLI converter is in place and ready to run against the provided source directory or any other directory path.

## Agent
GitHub Copilot (GPT-5.3-Codex)

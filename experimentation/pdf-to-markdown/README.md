# PDF to Markdown Converter

This utility recursively scans an input directory for PDF files and writes
Markdown files to an output directory with the same relative folder structure.

## Script

- `convert_pdfs_to_markdown.py`

## Usage

```bash
poetry run python experimentation/pdf-to-markdown/convert_pdfs_to_markdown.py \
  knowledge_ingestion/content/v2/content/orig_paper \
  knowledge_ingestion/content/v2/content_md/orig_paper
```

Optional flags:

- `--overwrite` overwrite existing `.md` files
- `--dry-run` print conversions without writing files

## Example: Convert your current paper set

```bash
poetry run python experimentation/pdf-to-markdown/convert_pdfs_to_markdown.py \
  knowledge_ingestion/content/v2/content/orig_paper \
  knowledge_ingestion/content/v2/content_md/orig_paper
```

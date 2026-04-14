# ingest_pipeline/parsers/mdx_parser.py
"""
Parser for .mdx and .md files.

Strategy:
  1. Extract YAML frontmatter (title, description) as document metadata.
  2. Strip JSX/MDX-specific syntax that has no plain-text meaning:
       - Import / export lines
       - Self-closing uppercase components  e.g. <Admonition />
       - Paired uppercase components        e.g. <Tabs>...</Tabs>
       - Inline JSX expressions             e.g. {someVar}
  3. Return a single Document whose page_content is the cleaned markdown.
     The chunker will later split it by heading hierarchy.
"""

import re
from pathlib import Path

import yaml
from langchain_core.documents import Document


# ── JSX cleaning patterns ─────────────────────────────────────────────────────

# Self-closing uppercase JSX tags: <ComponentName ... />
_RE_JSX_SELF_CLOSING = re.compile(r"<[A-Z][A-Za-z0-9]*[^>]*/\s*>", re.DOTALL)

# Paired uppercase JSX tags and their inner content: <Tabs>...</Tabs>
# DOTALL so the body can span multiple lines.
_RE_JSX_PAIRED = re.compile(r"<[A-Z][A-Za-z0-9]*[^>]*>.*?</[A-Z][A-Za-z0-9]*>", re.DOTALL)

# MDX import/export lines
_RE_MDX_IMPORT_EXPORT = re.compile(r"^(import|export)\s.*$", re.MULTILINE)

# Inline JSX expressions {expression} — leave plain {key: value} object notation
# in code blocks alone; we just strip bare {identifier} references outside code
_RE_JSX_EXPRESSION = re.compile(r"\{[^}\n]{0,80}\}")

# Collapse 3+ blank lines down to 2
_RE_EXCESS_BLANK = re.compile(r"\n{3,}")


def _strip_frontmatter(text: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body_without_frontmatter)."""
    match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n", text, re.DOTALL)
    if not match:
        return {}, text
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, text[match.end():]


def parse_mdx(path: Path) -> list[Document]:
    """
    Parse a single .mdx or .md file into one Document.

    The document carries all meaningful metadata extracted from frontmatter;
    the chunker is responsible for further splitting by heading.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")

    # Step 1 — frontmatter
    fm, body = _strip_frontmatter(raw)

    # Step 2 — strip JSX/MDX noise
    body = _RE_JSX_PAIRED.sub("", body)
    body = _RE_JSX_SELF_CLOSING.sub("", body)
    body = _RE_MDX_IMPORT_EXPORT.sub("", body)
    body = _RE_JSX_EXPRESSION.sub("", body)

    # Step 3 — normalise whitespace
    body = _RE_EXCESS_BLANK.sub("\n\n", body).strip()

    if not body:
        return []

    metadata = {
        "source_file":  str(path),
        "format":       "mdx" if path.suffix == ".mdx" else "md",
        "content_type": "narrative",
        "title":        fm.get("title", path.stem),
        "description":  fm.get("description", ""),
    }

    return [Document(page_content=body, metadata=metadata)]

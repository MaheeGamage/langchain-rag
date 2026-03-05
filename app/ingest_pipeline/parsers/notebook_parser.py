# ingest_pipeline/parsers/notebook_parser.py
"""
Parser for Jupyter Notebook (.ipynb) files.

Strategy:
  - Markdown cells → content_type "narrative"
  - Code cells     → content_type "code"
  - Cell outputs   → only plain-text outputs are kept (images/widgets discarded)
  - First markdown cell is inspected for YAML frontmatter to extract the title.

Each cell becomes its own Document so the chunker can handle them
independently (code cells are ideally kept whole; long markdown cells get
split by heading).
"""

import json
import re
from pathlib import Path

import yaml
from langchain_core.documents import Document

_RE_FRONTMATTER = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)


def _extract_title(source: str, fallback: str) -> tuple[str, str]:
    """
    Try to extract a title from YAML frontmatter or the first H1 heading.
    Returns (title, source_without_frontmatter).
    """
    fm_match = _RE_FRONTMATTER.match(source)
    if fm_match:
        try:
            fm = yaml.safe_load(fm_match.group(1)) or {}
            title = fm.get("title", fallback)
        except yaml.YAMLError:
            title = fallback
        return title, source[fm_match.end():]

    # Fallback: first # heading
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip(), source

    return fallback, source


def _plain_text_outputs(cell: dict) -> str:
    """Return text/plain output lines from a code cell, joined as a string."""
    lines: list[str] = []
    for output in cell.get("outputs", []):
        output_type = output.get("output_type", "")
        if output_type in ("stream",):
            lines.extend(output.get("text", []))
        elif output_type in ("execute_result", "display_data"):
            data = output.get("data", {})
            if "text/plain" in data:
                plain = data["text/plain"]
                lines.extend(plain if isinstance(plain, list) else [plain])
            # Explicitly skip image/png, image/svg+xml, application/json outputs
    return "".join(lines).strip()


def parse_notebook(path: Path) -> list[Document]:
    """
    Parse a .ipynb file.  Returns one Document per non-empty cell.
    """
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return []

    cells = nb.get("cells", [])
    if not cells:
        return []

    # Resolve notebook-level title from first markdown cell
    notebook_title = path.stem
    for cell in cells:
        if cell.get("cell_type") == "markdown":
            src = "".join(cell.get("source", []))
            notebook_title, _ = _extract_title(src, path.stem)
            break

    docs: list[Document] = []

    for idx, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "")
        source = "".join(cell.get("source", []))

        if not source.strip():
            continue

        base_meta = {
            "source_file":  str(path),
            "format":       "ipynb",
            "title":        notebook_title,
            "cell_index":   idx,
        }

        if cell_type == "markdown":
            # Strip frontmatter from the first cell but keep body
            _, clean_source = _extract_title(source, notebook_title) if idx == 0 else ("", source)
            if not clean_source.strip():
                continue
            docs.append(Document(
                page_content=clean_source.strip(),
                metadata={**base_meta, "content_type": "narrative"},
            ))

        elif cell_type == "code":
            # Build content: source code + optional plain-text output
            plain_output = _plain_text_outputs(cell)
            content = source.strip()
            if plain_output:
                content += f"\n\n# --- output ---\n{plain_output}"
            docs.append(Document(
                page_content=content,
                metadata={**base_meta, "content_type": "code"},
            ))

    return docs

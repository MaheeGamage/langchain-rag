# ingest_pipeline/parsers/python_parser.py
"""
Parser for .py files.

Strategy:
  1. Walk the AST to find every module / class / function with a docstring.
     Each docstring becomes a Document tagged content_type "narrative" so it
     is chunked and embedded like prose.
  2. Each definition's source code (including the docstring) is also emitted
     as a separate content_type "code" Document so code-search queries can
     match on implementation details.
  3. If the file cannot be parsed (syntax error) the entire source is stored
     as a single code Document as a fallback.
"""

import ast
import textwrap
from pathlib import Path

from langchain_core.documents import Document


def _source_segment(source_lines: list[str], node: ast.AST) -> str:
    """Extract the verbatim source lines for an AST node."""
    start = node.lineno - 1          # type: ignore[attr-defined]
    end   = node.end_lineno          # type: ignore[attr-defined]
    return "\n".join(source_lines[start:end])


def parse_python(path: Path) -> list[Document]:
    """
    Parse a .py file into docstring (narrative) and source-code Documents.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    source_lines = raw.splitlines()
    docs: list[Document] = []

    try:
        tree = ast.parse(raw, filename=str(path))
    except SyntaxError:
        # Cannot parse → store the whole file as a code chunk
        docs.append(Document(
            page_content=raw,
            metadata={
                "source_file":  str(path),
                "format":       "py",
                "content_type": "code",
                "title":        path.stem,
                "symbol":       "<module>",
            },
        ))
        return docs

    # Module-level docstring
    module_ds = ast.get_docstring(tree)
    if module_ds:
        docs.append(Document(
            page_content=textwrap.dedent(module_ds).strip(),
            metadata={
                "source_file":  str(path),
                "format":       "py",
                "content_type": "narrative",
                "title":        path.stem,
                "symbol":       "<module>",
            },
        ))

    # Per-definition docstrings + source code
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        symbol = node.name  # type: ignore[attr-defined]
        ds = ast.get_docstring(node)

        if ds:
            docs.append(Document(
                page_content=f"{symbol}:\n{textwrap.dedent(ds).strip()}",
                metadata={
                    "source_file":  str(path),
                    "format":       "py",
                    "content_type": "narrative",
                    "title":        path.stem,
                    "symbol":       symbol,
                },
            ))

        # Source code of the definition
        code_segment = _source_segment(source_lines, node)
        if code_segment.strip():
            docs.append(Document(
                page_content=code_segment,
                metadata={
                    "source_file":  str(path),
                    "format":       "py",
                    "content_type": "code",
                    "title":        path.stem,
                    "symbol":       symbol,
                },
            ))

    # Fallback: if nothing was extracted (e.g. a script with no functions)
    if not docs:
        docs.append(Document(
            page_content=raw,
            metadata={
                "source_file":  str(path),
                "format":       "py",
                "content_type": "code",
                "title":        path.stem,
                "symbol":       "<module>",
            },
        ))

    return docs

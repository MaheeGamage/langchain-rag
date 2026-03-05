# ingest_pipeline/router.py
"""
File router — maps each file path to the correct parser.

Skipped extensions
──────────────────
  .rst    Sphinx API-reference stubs (e.g. ".. automodule:: mlflow.pmdarima").
          These are too terse and too directive-heavy to embed meaningfully as
          prose.  They will be skipped until a proper Sphinx-resolve pass is
          added.

  .json   Navigation / TOC metadata (_toc.json, _package.json).  No knowledge
          content for a vector store.

  .csv    Raw numerical dataset files.  Not directly embeddable as text.  A
          future "dataset descriptor" pass could generate a natural-language
          summary and re-feed it here.

  .pack / .idx / .rev / .sample
          Git object store and sample config artefacts — not content.
"""

from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

from .parsers.mdx_parser      import parse_mdx
from .parsers.notebook_parser import parse_notebook
from .parsers.pdf_parser      import parse_pdf
from .parsers.python_parser   import parse_python

# ── Corpus detection ──────────────────────────────────────────────────────────
# Derived from the path so every chunk knows which knowledge domain it came from.
# Useful for Chroma `where` filters without a dedicated per-corpus collection.
_CORPUS_KEYWORDS: dict[str, str] = {
    "mlflow":                             "mlflow",
    "qiskit":                             "qiskit",
    "qprov":                              "qprov",
    "sample-quantum-circuit":             "sample",
}


def detect_corpus(path: Path) -> str:
    path_str = str(path).lower()
    for keyword, corpus in _CORPUS_KEYWORDS.items():
        if keyword in path_str:
            return corpus
    return "unknown"


# ── Extension → parser mapping ────────────────────────────────────────────────
_PARSER_MAP: dict[str, Callable[[Path], list[Document]]] = {
    ".mdx":  parse_mdx,
    ".md":   parse_mdx,
    ".ipynb": parse_notebook,
    ".pdf":  parse_pdf,
    ".py":   parse_python,
}

# Extensions that are explicitly skipped (not an error, just not useful content)
_SKIP_EXTENSIONS: frozenset[str] = frozenset({
    ".rst",     # Sphinx directive stubs — too terse without docstring resolution
    ".json",    # TOC / navigation metadata
    ".csv",     # Raw tabular data — no prose
    ".pack",    # Git object pack
    ".idx",     # Git index
    ".rev",     # Git revision
    ".sample",  # Git hook samples
})


def route_file(path: Path) -> list[Document]:
    """
    Dispatch a single file to its parser and inject corpus metadata.

    Returns an empty list for skipped / unsupported files.
    """
    ext = path.suffix.lower()

    if ext in _SKIP_EXTENSIONS:
        return []

    parser = _PARSER_MAP.get(ext)
    if parser is None:
        return []

    docs = parser(path)

    # Stamp every document with corpus so the RAG layer can filter by domain
    corpus = detect_corpus(path)
    for doc in docs:
        doc.metadata["source_corpus"] = corpus

    return docs


def walk_data_root(data_root: Path) -> list[Path]:
    """
    Recursively collect all files under data_root, sorted for determinism.
    """
    return sorted(p for p in data_root.rglob("*") if p.is_file())

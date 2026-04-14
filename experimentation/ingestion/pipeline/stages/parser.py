# pipeline/stages/parser.py
"""
Stage 2 — Route each file to a parser and return raw Documents.

A parser is any callable: (Path) -> list[Document].
Register new file types by adding an entry to the parser_map — no other
changes needed.

Corpus detection is also handled here: every Document gets a
`source_corpus` metadata field derived from path keywords so the retrieval
layer can filter by domain without separate collections.
"""

from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

# Re-use the existing parsers from app/ingest_pipeline — no duplication.
from knowledge_ingestion.ingest_pipeline.parsers.mdx_parser      import parse_mdx
from knowledge_ingestion.ingest_pipeline.parsers.notebook_parser import parse_notebook
from knowledge_ingestion.ingest_pipeline.parsers.pdf_parser      import parse_pdf
from knowledge_ingestion.ingest_pipeline.parsers.python_parser   import parse_python

ParserFn = Callable[[Path], list[Document]]

# ── Default corpus keyword map ────────────────────────────────────────────────
# Extend this dict to recognise new knowledge domains.
DEFAULT_CORPUS_KEYWORDS: dict[str, str] = {
    "mlflow":                 "mlflow",
    "qiskit":                 "qiskit",
    "qprov":                  "qprov",
    "sample-quantum-circuit": "sample",
    "orig_paper":             "paper",   # academic PDFs under orig_paper/
    "original-content":       "paper",   # alternate path convention
}

# ── Default parser map ────────────────────────────────────────────────────────
DEFAULT_PARSER_MAP: dict[str, ParserFn] = {
    ".mdx":  parse_mdx,
    ".md":   parse_mdx,
    ".ipynb": parse_notebook,
    ".pdf":  parse_pdf,
    ".py":   parse_python,
}


class ParserStage:
    """
    Dispatch files to parsers and stamp corpus metadata.

    Parameters
    ----------
    parser_map:
        Maps file extension (lower-case, with dot) to a parser callable.
    corpus_keywords:
        Maps path substring to corpus label.  First match wins.
    """

    def __init__(
        self,
        parser_map: dict[str, ParserFn] | None = None,
        corpus_keywords: dict[str, str] | None = None,
    ) -> None:
        self.parser_map = parser_map or DEFAULT_PARSER_MAP
        self.corpus_keywords = corpus_keywords or DEFAULT_CORPUS_KEYWORDS

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self, paths: list[Path]) -> tuple[list[Document], int]:
        """
        Parse all paths.

        Returns
        -------
        docs:
            All successfully parsed Documents.
        skipped:
            Number of files with no registered parser (not an error).
        """
        docs: list[Document] = []
        skipped = 0

        for path in paths:
            parsed = self._parse_one(path)
            if not parsed:
                skipped += 1
                continue
            corpus = self._detect_corpus(path)
            for doc in parsed:
                doc.metadata["source_corpus"] = corpus
            docs.extend(parsed)

        return docs, skipped

    # ── Private ───────────────────────────────────────────────────────────────

    def _parse_one(self, path: Path) -> list[Document]:
        parser = self.parser_map.get(path.suffix.lower())
        if parser is None:
            return []
        try:
            return parser(path)
        except Exception as exc:
            # Log and continue — one bad file should not abort the run.
            import logging
            logging.getLogger("ingest_pipeline").warning(
                f"Parser failed for {path}: {exc!r}"
            )
            return []

    def _detect_corpus(self, path: Path) -> str:
        path_str = str(path).lower()
        for keyword, corpus in self.corpus_keywords.items():
            if keyword in path_str:
                return corpus
        return "unknown"

# ingest_pipeline/chunker.py
"""
Chunking strategy for the heterogeneous document set.

Narrative documents (MDX, MD, PDF, notebook markdown cells)
────────────────────────────────────────────────────────────
  1. MarkdownHeaderTextSplitter splits by H1/H2/H3 boundaries into "sections".
     Each section inherits heading breadcrumb metadata  (h1, h2, h3) so a
     retrieved chunk always carries its structural location.
  2. Sections that exceed CHUNK_SIZE are further split by
     RecursiveCharacterTextSplitter, which prefers paragraph and sentence
     boundaries, with CHUNK_OVERLAP tokens of context carry-over.

Code documents (notebook code cells, Python source)
────────────────────────────────────────────────────
  Code cells and extracted function/class bodies are typically short and
  self-contained; we keep them whole and only split if they exceed CHUNK_SIZE.
  Separators respect Python / Jupyter conventions (class, def boundaries).

Chunk size rationale
────────────────────
  CHUNK_SIZE = 2 000 chars ≈ 500 tokens (at ~4 chars/token average)
  CHUNK_OVERLAP = 200 chars ≈ 50 tokens

  This is the standard "middle ground":
    - Fine enough for precise retrieval over 10k+ document corpus.
    - Coarse enough to carry enough surrounding context for the LLM answer step.
  Increase CHUNK_SIZE if answers are missing context; decrease if retrieval
  precision drops (too many irrelevant chunks bubble up).
"""

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


# ── Splitter instances (created once, reused) ─────────────────────────────────

_HEADERS_TO_SPLIT: list[tuple[str, str]] = [
    ("#",   "h1"),
    ("##",  "h2"),
    ("###", "h3"),
]

_header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=_HEADERS_TO_SPLIT,
    strip_headers=False,  # keep the heading in the chunk text for context
)

_narrative_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    # Prefer splitting at paragraph / sentence boundaries before hard character cut
    separators=["\n\n", "\n", ". ", " ", ""],
)

_code_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    # Prefer splitting at top-level definition boundaries
    separators=["\n\ndef ", "\n\nclass ", "\ndef ", "\nclass ", "\n\n", "\n"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_section_path(header_meta: dict) -> str:
    """
    Combine h1/h2/h3 header metadata into a human-readable breadcrumb.

    Example: "Guides > Construct circuits > What is a quantum circuit?"
    """
    parts = [
        header_meta.get("h1", ""),
        header_meta.get("h2", ""),
        header_meta.get("h3", ""),
    ]
    return " > ".join(p for p in parts if p)


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_document(doc: Document) -> list[Document]:
    """
    Split one Document into retrieval-ready chunks.

    The output chunks inherit all metadata from the source document and gain
    an additional `section` field with the heading breadcrumb (narrative only).
    """
    content_type = doc.metadata.get("content_type", "narrative")

    if content_type == "code":
        return _chunk_code(doc)
    else:
        return _chunk_narrative(doc)


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Convenience wrapper: chunk a list of Documents."""
    result: list[Document] = []
    for doc in docs:
        result.extend(chunk_document(doc))
    return result


# ── Private split logic ───────────────────────────────────────────────────────

def _chunk_narrative(doc: Document) -> list[Document]:
    """
    Two-pass split for prose content:
      Pass 1 — section boundaries via MarkdownHeaderTextSplitter.
      Pass 2 — size enforcement via RecursiveCharacterTextSplitter.
    """
    sections = _header_splitter.split_text(doc.page_content)
    chunks: list[Document] = []

    for section in sections:
        # section.metadata carries {h1, h2, h3} keys set by the splitter
        section_path = _build_section_path(section.metadata)

        # Merge parent metadata with section-level heading metadata
        merged_meta = {
            **doc.metadata,
            **section.metadata,          # h1 / h2 / h3 fields
            "section": section_path,
        }

        if len(section.page_content) <= CHUNK_SIZE:
            # Small enough — keep as one chunk
            chunks.append(Document(
                page_content=section.page_content,
                metadata=merged_meta,
            ))
        else:
            # Too large — split further while preserving metadata
            sub_chunks = _narrative_splitter.split_text(section.page_content)
            for sub in sub_chunks:
                chunks.append(Document(
                    page_content=sub,
                    metadata=merged_meta,
                ))

    # Fallback: if the header splitter produced nothing (no headings in doc)
    if not chunks:
        for text in _narrative_splitter.split_text(doc.page_content):
            chunks.append(Document(
                page_content=text,
                metadata={**doc.metadata, "section": ""},
            ))

    return chunks


def _chunk_code(doc: Document) -> list[Document]:
    """
    Simple size-based split for code content.

    Code units (functions, classes, notebook cells) are usually short and
    should not be split across heading boundaries.  We only apply the size
    splitter, preserving the original metadata exactly.
    """
    if len(doc.page_content) <= CHUNK_SIZE:
        return [doc]

    texts = _code_splitter.split_text(doc.page_content)
    return [
        Document(page_content=text, metadata=doc.metadata)
        for text in texts
    ]

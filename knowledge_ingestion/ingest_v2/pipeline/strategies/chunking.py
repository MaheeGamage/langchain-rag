# pipeline/strategies/chunking.py
"""
Per-document-type chunking strategies.

Each strategy implements one method:
    chunk(doc: Document) -> list[Document]

The ChunkingStage selects a strategy based on doc.metadata["content_type"]
(and optionally "doc_type" for finer routing).

Adding a new strategy
─────────────────────
1. Subclass ChunkingStrategy and implement chunk().
2. Register it in ChunkingStage's strategy_map.
That's it — no other files change.

Strategy rationale (from rag_embedding_strategy.md)
────────────────────────────────────────────────────
  narrative   → section-level split (H2/H3) + size enforcement
  code        → size-only split respecting def/class boundaries
  synthetic   → sliding window with generous overlap (cross-domain docs)
  paper       → abstract-prefixed section chunks (parent-doc pattern)
  simple_text → flat sliding-window split by chunk size and overlap only
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


# ── Base ──────────────────────────────────────────────────────────────────────

class ChunkingStrategy(ABC):
    """All strategies implement this single method."""

    @abstractmethod
    def chunk(self, doc: Document) -> list[Document]:
        ...


# ── Shared splitter instances (created once) ──────────────────────────────────

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]

_header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=_HEADERS,
    strip_headers=False,
)

_narrative_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

_code_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\ndef ", "\n\nclass ", "\ndef ", "\nclass ", "\n\n", "\n"],
)


def _section_path(meta: dict) -> str:
    parts = [meta.get("h1", ""), meta.get("h2", ""), meta.get("h3", "")]
    return " > ".join(p for p in parts if p)


# ── Concrete strategies ───────────────────────────────────────────────────────

class NarrativeChunkingStrategy(ChunkingStrategy):
    """
    Two-pass split for prose (MDX, MD, PDF, notebook markdown cells).

    Pass 1 — MarkdownHeaderTextSplitter splits at H1/H2/H3 boundaries.
             Each section inherits a heading breadcrumb in metadata.
    Pass 2 — RecursiveCharacterTextSplitter enforces CHUNK_SIZE.

    Heading-only sections (no body text beyond the heading line itself) are
    dropped — they are navigation artefacts from index pages and carry no
    retrievable knowledge.
    """

    # Minimum body length (chars) after stripping the heading line.
    # Sections shorter than this are discarded as heading-only stubs.
    MIN_BODY_CHARS: int = 50

    def chunk(self, doc: Document) -> list[Document]:
        sections = _header_splitter.split_text(doc.page_content)
        chunks: list[Document] = []

        for section in sections:
            body = self._body_text(section.page_content)
            if len(body) < self.MIN_BODY_CHARS:
                # Heading-only stub — no useful content to embed
                continue

            merged_meta = {
                **doc.metadata,
                **section.metadata,
                "section": _section_path(section.metadata),
            }
            if len(section.page_content) <= CHUNK_SIZE:
                chunks.append(Document(page_content=section.page_content, metadata=merged_meta))
            else:
                for text in _narrative_splitter.split_text(section.page_content):
                    chunks.append(Document(page_content=text, metadata=merged_meta))

        # Fallback: no headings found
        if not chunks:
            for text in _narrative_splitter.split_text(doc.page_content):
                chunks.append(Document(page_content=text, metadata={**doc.metadata, "section": ""}))

        return chunks

    @staticmethod
    def _body_text(content: str) -> str:
        """Return content with leading heading lines stripped."""
        lines = content.splitlines()
        body_lines = [l for l in lines if not l.strip().startswith("#")]
        return "\n".join(body_lines).strip()


class CodeChunkingStrategy(ChunkingStrategy):
    """
    Size-only split for code (notebook code cells, Python source).

    Code units are kept whole when possible; only split at def/class
    boundaries when they exceed CHUNK_SIZE.
    """

    def chunk(self, doc: Document) -> list[Document]:
        if len(doc.page_content) <= CHUNK_SIZE:
            return [doc]
        return [
            Document(page_content=text, metadata=doc.metadata)
            for text in _code_splitter.split_text(doc.page_content)
        ]


class SyntheticDocChunkingStrategy(ChunkingStrategy):
    """
    Sliding window with generous overlap for cross-domain synthetic documents
    (e.g. qiskit_experiment_tracking_bridge.md, qprov_taxonomy.md).

    These docs synthesise knowledge from multiple domains; larger overlap
    preserves cross-referential context at chunk boundaries.
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap_ratio: float = 0.20) -> None:
        overlap = int(chunk_size * overlap_ratio)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, doc: Document) -> list[Document]:
        if len(doc.page_content) <= self._splitter._chunk_size:
            return [doc]
        return [
            Document(page_content=text, metadata=doc.metadata)
            for text in self._splitter.split_text(doc.page_content)
        ]


class PlainTextChunkingStrategy(ChunkingStrategy):
    """
    Simple size-and-overlap split with no structural awareness.

    Splits the raw text purely by character count using CHUNK_SIZE and
    CHUNK_OVERLAP from config.  No heading detection, no separator
    hierarchy — just a flat sliding window over the full document text.

    Use this for plain .txt files, unstructured blobs, or any document
    where markdown/code structure should be ignored.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk(self, doc: Document) -> list[Document]:
        if len(doc.page_content) <= self._splitter._chunk_size:
            return [doc]
        return [
            Document(page_content=text, metadata=doc.metadata)
            for text in self._splitter.split_text(doc.page_content)
        ]


class PaperChunkingStrategy(ChunkingStrategy):
    """
    Abstract-prefixed section chunking for academic papers.

    The abstract is extracted as a standalone chunk AND prepended as a
    prefix to every other section chunk from the same paper.  This is the
    'parent document retriever' pattern: any retrieved passage carries
    enough framing context for the generator.

    For markdown papers: looks for a section whose heading contains 'abstract'.
    For PDFs (no headings): treats the first chunk (page 0) as the abstract.
    Falls back to NarrativeChunkingStrategy if no abstract can be identified.
    """

    def __init__(self) -> None:
        self._narrative = NarrativeChunkingStrategy()

    def chunk(self, doc: Document) -> list[Document]:
        all_chunks = self._narrative.chunk(doc)

        abstract_text = self._find_abstract(all_chunks, doc)
        if not abstract_text:
            return all_chunks

        result: list[Document] = []
        for chunk in all_chunks:
            if chunk.page_content == abstract_text:
                result.append(chunk)
            else:
                prefixed = f"[Abstract]\n{abstract_text}\n\n[Section]\n{chunk.page_content}"
                result.append(Document(
                    page_content=prefixed,
                    metadata={**chunk.metadata, "abstract_prefixed": True},
                ))
        return result

    def chunk_group(self, docs: list[Document]) -> list[Document]:
        """
        Chunk all pages/sections of one paper together so the abstract
        from page 0 can be prepended to every other chunk in the paper.
        """
        # Chunk each doc individually first
        all_chunks: list[Document] = []
        for doc in docs:
            all_chunks.extend(self._narrative.chunk(doc))

        # Find abstract across the whole paper (page 0 for PDFs)
        abstract_text = self._find_abstract(all_chunks, docs[0])
        if not abstract_text:
            return all_chunks

        result: list[Document] = []
        for chunk in all_chunks:
            if chunk.page_content == abstract_text:
                result.append(chunk)
            else:
                prefixed = f"[Abstract]\n{abstract_text}\n\n[Section]\n{chunk.page_content}"
                result.append(Document(
                    page_content=prefixed,
                    metadata={**chunk.metadata, "abstract_prefixed": True},
                ))
        return result

    @staticmethod
    def _find_abstract(chunks: list[Document], source_doc: Document) -> str:
        # Strategy 1: heading-based (markdown papers)
        for chunk in chunks:
            meta = chunk.metadata
            haystack = " ".join([
                meta.get("section", ""),
                meta.get("h1", ""),
                meta.get("h2", ""),
                meta.get("h3", ""),
            ]).lower()
            if "abstract" in haystack:
                return chunk.page_content

        # Strategy 2: PDF convention — page 0 is the abstract/intro
        if source_doc.metadata.get("format") == "pdf":
            for chunk in chunks:
                if chunk.metadata.get("page", -1) == 0:
                    return chunk.page_content

        return ""

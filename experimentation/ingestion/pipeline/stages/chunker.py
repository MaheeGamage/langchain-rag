# pipeline/stages/chunker.py
"""
Stage 3 — Split raw Documents into retrieval-ready chunks.

The stage itself is just a router: it picks the right ChunkingStrategy
for each document based on metadata, then delegates.

Routing logic (in priority order)
──────────────────────────────────
1. doc_type field  (most specific — set by parsers that know the doc type)
2. content_type field  (narrative / code)
3. source_corpus field  (e.g. "paper" corpus → PaperChunkingStrategy)
4. Default fallback

To change how a document type is chunked, swap its strategy in the
strategy_map — nothing else changes.
"""

from langchain_core.documents import Document

from ..strategies.chunking import (
    ChunkingStrategy,
    NarrativeChunkingStrategy,
    CodeChunkingStrategy,
    SyntheticDocChunkingStrategy,
    PaperChunkingStrategy,
    PlainTextChunkingStrategy,
)

# ── Corpus names that should use the paper strategy ───────────────────────────
PAPER_CORPORA: frozenset[str] = frozenset({"paper", "orig_paper"})

# ── Source file substrings that indicate synthetic cross-domain docs ──────────
SYNTHETIC_FILENAMES: tuple[str, ...] = (
    "qiskit_experiment_tracking_bridge",
    "qprov_taxonomy",
)


class ChunkingStage:
    """
    Route each Document to the appropriate ChunkingStrategy and chunk it.

    Parameters
    ----------
    strategy_map:
        Maps a strategy key string to a ChunkingStrategy instance.
        Keys used internally: "narrative", "code", "synthetic", "paper".
        Override any key to swap that strategy without touching others.
    """

    def __init__(self, strategy_map: dict[str, ChunkingStrategy] | None = None) -> None:
        self.strategy_map: dict[str, ChunkingStrategy] = {
            "narrative": NarrativeChunkingStrategy(),
            "code":      CodeChunkingStrategy(),
            "synthetic": SyntheticDocChunkingStrategy(),
            "paper":     PaperChunkingStrategy(),
            "plain":     PlainTextChunkingStrategy(),
            **(strategy_map or {}),
        }

    def run(self, docs: list[Document]) -> list[Document]:
        chunks: list[Document] = []

        # Group docs by source_file so PaperChunkingStrategy can see all pages
        # of a paper together (needed for abstract-prefix across pages).
        from collections import defaultdict
        groups: dict[str, list[Document]] = defaultdict(list)
        for doc in docs:
            key = doc.metadata.get("source_file", id(doc))
            groups[key].append(doc)

        for source_file, group_docs in groups.items():
            strategy = self._select_strategy(group_docs[0])
            if hasattr(strategy, "chunk_group"):
                # Strategy supports multi-doc grouping (e.g. PaperChunkingStrategy)
                chunks.extend(strategy.chunk_group(group_docs))
            else:
                for doc in group_docs:
                    chunks.extend(strategy.chunk(doc))

        return chunks

    # ── Private ───────────────────────────────────────────────────────────────

    def _select_strategy(self, doc: Document) -> ChunkingStrategy:
        meta = doc.metadata

        # 1. Explicit doc_type override (parsers can set this)
        doc_type = meta.get("doc_type", "")
        if doc_type in self.strategy_map:
            return self.strategy_map[doc_type]

        # 2. Code content — always use code strategy regardless of corpus
        if meta.get("content_type") == "code":
            return self.strategy_map["code"]

        # 3. Corpus-based routing
        corpus = meta.get("source_corpus", "")
        if corpus in PAPER_CORPORA:
            return self.strategy_map["paper"]

        # 4. Filename-based routing for known synthetic docs
        source_file = meta.get("source_file", "").lower()
        if any(name in source_file for name in SYNTHETIC_FILENAMES):
            return self.strategy_map["synthetic"]

        # 5. Default: narrative
        return self.strategy_map["narrative"]

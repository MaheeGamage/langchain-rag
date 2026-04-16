# app/retriever.py

import logging
from dataclasses import dataclass

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .vectorstore import get_vectorstore

log = logging.getLogger(__name__)


# ── Per-retriever config dataclasses ─────────────────────────────────────────
# Each retriever type owns its own parameters.  Add new config classes here
# as new retriever types are introduced.

@dataclass
class SemanticConfig:
    """Configuration for vector-similarity (semantic) retrieval."""
    k: int = 4


@dataclass
class BM25Config:
    """Configuration for BM25 keyword retrieval."""
    k: int = 4


# ── Individual retriever factories ────────────────────────────────────────────

def get_semantic_retriever(config: SemanticConfig = SemanticConfig()) -> BaseRetriever:
    """Standard top-k vector similarity search."""
    return get_vectorstore().as_retriever(search_kwargs={"k": config.k})


def get_bm25_retriever(config: BM25Config = BM25Config()) -> BaseRetriever:
    """BM25 keyword retriever built from all indexed documents.

    Handles exact token matching for domain-specific terms (e.g. NISQ, VQE,
    QAOA) that embedding models may represent poorly.

    Loads the full corpus from ChromaDB at call time — call once at graph
    initialisation rather than per-query.
    """
    vectorstore = get_vectorstore()
    docs = _fetch_all_documents(vectorstore)
    log.info("Building BM25 index over %d documents", len(docs))
    return BM25Retriever.from_documents(docs, k=config.k)


# ── Hybrid combinator ─────────────────────────────────────────────────────────

def get_hybrid_retriever(
    retrievers_with_weights: list[tuple[BaseRetriever, float]],
) -> BaseRetriever:
    """Combine any number of retrievers via reciprocal rank fusion.

    Each entry in *retrievers_with_weights* is a (retriever, weight) pair.
    Weights are passed directly to EnsembleRetriever — they need not sum to 1
    but are normalised internally by LangChain.

    Example — two retrievers::

        semantic = get_semantic_retriever(SemanticConfig(k=4))
        bm25     = get_bm25_retriever(BM25Config(k=6))
        hybrid   = get_hybrid_retriever([(semantic, 0.6), (bm25, 0.4)])

    Example — three retrievers (future)::

        mmr    = get_mmr_retriever(MMRConfig(...))
        hybrid = get_hybrid_retriever([(semantic, 0.5), (bm25, 0.3), (mmr, 0.2)])
    """
    active = [(r, w) for r, w in retrievers_with_weights if w > 0]
    if not active:
        raise ValueError("get_hybrid_retriever: all retrievers have weight <= 0")
    if len(active) == 1:
        return active[0][0]
    retrievers = [r for r, _ in active]
    weights    = [w for _, w in active]
    return EnsembleRetriever(retrievers=retrievers, weights=weights)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_all_documents(vectorstore) -> list[Document]:
    result = vectorstore._collection.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(result["documents"], result["metadatas"])
    ]

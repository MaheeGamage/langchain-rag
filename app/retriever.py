# app/retriever.py

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from .vectorstore import get_vectorstore

log = logging.getLogger(__name__)

Reranker = Callable[[str, list[Document]], list[Document]]
Tokenizer = Callable[[str], list[str]]


# ── BM25 tokenizers ──────────────────────────────────────────────────────────

_PUNCT_STRIP = ".,;:!?()[]{}\"'`"
_SPLIT_COMPOUND_RE = re.compile(r"[._]")


def api_aware_tokenize(text: str) -> list[str]:
    """BM25 tokenizer that handles API-style identifiers.

    Whitespace-splits, lowercases, strips trailing punctuation, and — for
    tokens containing ``.`` or ``_`` — also emits the sub-parts while
    *keeping* the original compound token.  So ``mlflow.log_param`` produces
    ``["mlflow.log_param", "mlflow", "log", "param"]``; exact API-name
    queries still hit the compound, partial queries (e.g. ``log_param``,
    ``start_run``) hit the sub-parts.

    Use case: the default whitespace tokenizer in BM25Retriever treats
    ``mlflow.log_param`` as one opaque token, so a user query for
    ``log_param`` never matches.
    """
    tokens: list[str] = []
    for word in text.lower().split():
        word = word.strip(_PUNCT_STRIP)
        if not word:
            continue
        tokens.append(word)
        if "." in word or "_" in word:
            parts = [p for p in _SPLIT_COMPOUND_RE.split(word) if p]
            if len(parts) > 1:
                tokens.extend(parts)
    return tokens


# ── Per-retriever config dataclasses ─────────────────────────────────────────
# Each retriever type owns its own parameters.  Add new config classes here
# as new retriever types are introduced.

@dataclass
class SemanticConfig:
    """Configuration for vector-similarity (semantic) retrieval.

    score_threshold: relevance score lower bound (0–1, higher = more similar).
    Documents scoring BELOW this value are dropped before RRF merging.
    Note: this is a *similarity* score (higher = better), NOT a distance
    (where lower = better).  LangChain converts ChromaDB's raw L2 distance
    into this 0–1 relevance scale via similarity_search_with_relevance_scores.

    Use case: for acronym queries (NISQ, VQE, QAOA) the embedding model
    returns irrelevant MLflow docs at 0.43–0.53.  Setting threshold=0.55
    drops all semantic results for those queries so BM25 handles them alone.
    Verify good queries (e.g. "quantum circuit") still score above the
    threshold before committing a value.  Set to None to disable filtering.
    """
    k: int = 4
    score_threshold: float | None = None


@dataclass
class BM25Config:
    """Configuration for BM25 keyword retrieval.

    preprocess_func: optional tokenizer.  When None, BM25Retriever uses its
    default whitespace split.  Set to ``api_aware_tokenize`` (or any custom
    tokenizer) to control how text is split into tokens.  The same function
    is applied to both indexed documents AND incoming queries, so tokens
    must match on both sides.
    """
    k: int = 4
    preprocess_func: Optional[Tokenizer] = None


# ── Individual retriever factories ────────────────────────────────────────────

def get_semantic_retriever(config: SemanticConfig = SemanticConfig()) -> BaseRetriever:
    """Standard top-k vector similarity search.

    When config.score_threshold is set, uses similarity_score_threshold
    search so low-relevance documents are dropped before RRF merging.
    """
    if config.score_threshold is not None:
        return get_vectorstore().as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": config.k, "score_threshold": config.score_threshold},
        )
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
    kwargs: dict = {"k": config.k}
    if config.preprocess_func is not None:
        kwargs["preprocess_func"] = config.preprocess_func
    return BM25Retriever.from_documents(docs, **kwargs)


# ── Hybrid combinator ─────────────────────────────────────────────────────────

@dataclass
class HybridConfig:
    """Top-level hybrid retriever configuration.

    Combines semantic + BM25 with weighted RRF, then optionally applies a
    reranker and trims to ``final_k``.  All previous call-site parameters
    (k, score_threshold, weights) are expressible here — defaults match the
    historic behaviour (semantic/bm25 k=4, weights 0.5/0.5, no threshold,
    no reranker, no final cap).

    Fields:
        semantic:        SemanticConfig for the dense retriever.
        bm25:            BM25Config for the keyword retriever.
        semantic_weight: RRF weight for the semantic retriever.
        bm25_weight:     RRF weight for the BM25 retriever.
        rrf_c:           RRF constant (matches LangChain default of 60;
                         higher values flatten rank differences).
        final_k:         If set, trim the fused/reranked result to this many.
        reranker:        Optional callable (query, docs) -> reranked docs.
                         Applied after RRF fusion, before final_k trimming.
    """
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    semantic_weight: float = 0.5
    bm25_weight: float = 0.5
    rrf_c: int = 60
    final_k: Optional[int] = None
    reranker: Optional[Reranker] = None


class _PostProcessingRetriever(BaseRetriever):
    """Wraps a base retriever with optional rerank + final_k trimming.

    Kept private: use ``build_hybrid_retriever`` to construct.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    base: BaseRetriever
    reranker: Optional[Reranker] = None
    final_k: Optional[int] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        docs = list(self.base.invoke(query))
        if self.reranker is not None:
            docs = list(self.reranker(query, docs))
        if self.final_k is not None:
            docs = docs[: self.final_k]
        return docs


def build_hybrid_retriever(config: HybridConfig = HybridConfig()) -> BaseRetriever:
    """Construct a hybrid retriever from a single config object.

    Pipeline: semantic(k) + bm25(k) → RRF fusion (weighted) →
    [optional reranker] → [optional final_k cap].

    Retrievers with weight <= 0 are skipped entirely (no index build).
    With the default ``HybridConfig()`` the behaviour is identical to the
    previous call-site pattern of ``get_hybrid_retriever([(sem, 0.5), (bm25, 0.5)])``.
    """
    retrievers_with_weights: list[tuple[BaseRetriever, float]] = []
    if config.semantic_weight > 0:
        retrievers_with_weights.append(
            (get_semantic_retriever(config.semantic), config.semantic_weight)
        )
    if config.bm25_weight > 0:
        retrievers_with_weights.append(
            (get_bm25_retriever(config.bm25), config.bm25_weight)
        )
    base = get_hybrid_retriever(retrievers_with_weights, rrf_c=config.rrf_c)
    if config.reranker is None and config.final_k is None:
        return base
    return _PostProcessingRetriever(
        base=base, reranker=config.reranker, final_k=config.final_k
    )


def get_hybrid_retriever(
    retrievers_with_weights: list[tuple[BaseRetriever, float]],
    *,
    rrf_c: int = 60,
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
    return EnsembleRetriever(retrievers=retrievers, weights=weights, c=rrf_c)


# ── Named retrieval profiles ──────────────────────────────────────────────────

# Preset HybridConfig variants for different query shapes.  Used by rag_agent
# (and any other caller) so the LLM can pick a strategy by name rather than
# tune weights/k directly.  The "default" profile reproduces the historic
# call-site configuration exactly.
#
# All profiles use ``api_aware_tokenize`` for BM25 so compound API names
# (e.g. "mlflow.log_param") are searchable by their parts.
PROFILES: dict[str, "HybridConfig"] = {
    "default": HybridConfig(
        semantic=SemanticConfig(k=4, score_threshold=0.55),
        bm25=BM25Config(k=4, preprocess_func=api_aware_tokenize),
        semantic_weight=0.5,
        bm25_weight=0.5,
    ),
    "acronym": HybridConfig(
        semantic=SemanticConfig(k=6, score_threshold=None),
        bm25=BM25Config(k=6, preprocess_func=api_aware_tokenize),
        semantic_weight=0.2,
        bm25_weight=0.8,
    ),
    "conceptual": HybridConfig(
        semantic=SemanticConfig(k=4, score_threshold=0.45),
        bm25=BM25Config(k=4, preprocess_func=api_aware_tokenize),
        semantic_weight=0.7,
        bm25_weight=0.3,
    ),
    "overview": HybridConfig(
        semantic=SemanticConfig(k=10, score_threshold=None),
        bm25=BM25Config(k=10, preprocess_func=api_aware_tokenize),
        semantic_weight=0.5,
        bm25_weight=0.5,
    ),
    "reasoning": HybridConfig(
        semantic=SemanticConfig(k=8, score_threshold=None),
        bm25=BM25Config(k=8, preprocess_func=api_aware_tokenize),
        semantic_weight=0.5,
        bm25_weight=0.5,
    ),
    "semantic": HybridConfig(
        semantic=SemanticConfig(k=6, score_threshold=None),
        bm25=BM25Config(k=0),
        semantic_weight=1.0,
        bm25_weight=0.0,
    ),
}

# Profiles whose HybridConfig depends on a heavy dependency (e.g. the
# sentence-transformers cross-encoder) are held as nullary factories instead
# of module-level HybridConfig instances, so the model isn't loaded until the
# profile is actually selected.  Factories are consulted BEFORE PROFILES in
# get_profile_retriever, so a name present in both would prefer the factory.
_PROFILE_FACTORIES: dict[str, Callable[[], "HybridConfig"]] = {
    # Wide retrieve (k=10 each → up to 20 candidates) then cross-encoder
    # rerank and trim to 4.  Higher latency and requires sentence-transformers
    # but typically improves precision@k on long-tail queries.
    "reranked": lambda: HybridConfig(
        semantic=SemanticConfig(k=10, score_threshold=None),
        bm25=BM25Config(k=10, preprocess_func=api_aware_tokenize),
        semantic_weight=0.5,
        bm25_weight=0.5,
        reranker=make_cross_encoder_reranker(),
        final_k=4,
    ),
}

PROFILE_DESCRIPTIONS: dict[str, str] = {
    "default":    "Balanced 50/50 hybrid. Use when unsure.",
    "acronym":    "BM25-heavy (0.2/0.8), k=6. For acronyms or exact API names.",
    "conceptual": "Semantic-heavy (0.7/0.3), k=4, threshold=0.45. For definitions and explanations.",
    "overview":   "Balanced, k=10. For summary/listing/taxonomy questions needing broad context.",
    "reasoning":  "Balanced, k=8. For multi-hop reasoning needing several supporting facts.",
    "reranked":   "Wide retrieve (k=10 each) + cross-encoder rerank, trimmed to 4. Higher latency; best precision.",
    "semantic":   "Semantic-only (k=6, no BM25). For queries where keyword matching adds noise.",
}

# All valid profile names — union of eager configs and lazy factories.  Callers
# should check membership against this rather than PROFILES alone.
PROFILE_NAMES: frozenset[str] = frozenset(PROFILES) | frozenset(_PROFILE_FACTORIES)

# Lazy per-profile cache so BM25 indexing (and reranker model load) happens at
# most once per profile per process.
_profile_retriever_cache: dict[str, BaseRetriever] = {}


def get_profile_retriever(profile: str) -> BaseRetriever:
    """Return a lazily-constructed retriever for a named profile.

    Unknown names fall back to "default" with a warning.  Profiles are cached
    at module level so BM25 index build (and reranker model load) happens once
    per profile per process.
    """
    if profile not in PROFILE_NAMES:
        log.warning("Unknown retrieval profile %r; falling back to 'default'", profile)
        profile = "default"
    if profile not in _profile_retriever_cache:
        if profile in _PROFILE_FACTORIES:
            cfg = _PROFILE_FACTORIES[profile]()
        else:
            cfg = PROFILES[profile]
        _profile_retriever_cache[profile] = build_hybrid_retriever(cfg)
    return _profile_retriever_cache[profile]


# ── Rerankers ─────────────────────────────────────────────────────────────────

def make_cross_encoder_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> Reranker:
    """Build a cross-encoder reranker for use with HybridConfig.reranker.

    Requires ``sentence-transformers`` at call time (lazy import so the rest
    of the retriever module works without it).  Loads the model once and
    returns a callable (query, docs) -> docs sorted by cross-encoder score.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "make_cross_encoder_reranker requires sentence-transformers. "
            "Install with: pip install sentence-transformers"
        ) from exc

    model = CrossEncoder(model_name)
    log.info("Loaded cross-encoder reranker: %s", model_name)

    def _rerank(query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return docs
        pairs = [(query, d.page_content) for d in docs]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda pair: pair[0], reverse=True)
        return [doc for _, doc in ranked]

    return _rerank


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_all_documents(vectorstore) -> list[Document]:
    result = vectorstore._collection.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(result["documents"], result["metadatas"])
    ]

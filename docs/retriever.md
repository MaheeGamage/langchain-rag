# Retriever

Hybrid semantic + BM25 retrieval with named profiles and optional cross-encoder reranking.

Lives in [`app/retriever.py`](../app/retriever.py). This doc explains **what it does**, **how to use it**, and **how it's put together**.

---

## TL;DR

```python
from app.retriever import get_profile_retriever

retriever = get_profile_retriever("default")      # or "acronym", "conceptual", ...
docs = retriever.invoke("What is VQE?")
```

That's it. The profile name picks a preset of retrieval parameters tuned for different question shapes.

---

## Concepts

### Two retrievers, one merged result

- **Semantic retriever** — embeds the query, finds chunks with similar meaning. Good at paraphrases, weak on rare tokens like `NISQ` or `mlflow.log_param`.
- **BM25 retriever** — classic keyword frequency scoring. Good at exact tokens, blind to synonyms.
- **Hybrid** — run both, merge ranked lists via **Reciprocal Rank Fusion (RRF)** with weights. Gets the best of both.

### Optional post-processing

After fusion, the result can go through:

1. **Reranker** (optional) — a cross-encoder re-scores every (query, doc) pair and reorders.
2. **final_k trim** (optional) — cut to the top N.

### Profiles

Retrieval has many knobs (k, weights, thresholds). Instead of tuning them at every call site, **profiles** are named presets. Callers — the agent, the query-rewriter, eval scripts — just say `"acronym"` or `"reranked"` and get the right config.

---

## Quick start

### 1. Use a named profile (recommended)

```python
from app.retriever import get_profile_retriever

retriever = get_profile_retriever("default")
docs = retriever.invoke("How does mlflow.log_param work?")
```

Profiles are cached per process — the BM25 index (and reranker model, if applicable) is built once on first use.

### 2. Build a custom hybrid retriever

```python
from app.retriever import (
    BM25Config, HybridConfig, SemanticConfig,
    api_aware_tokenize, build_hybrid_retriever,
)

cfg = HybridConfig(
    semantic=SemanticConfig(k=6, score_threshold=0.5),
    bm25=BM25Config(k=6, preprocess_func=api_aware_tokenize),
    semantic_weight=0.4,
    bm25_weight=0.6,
)
retriever = build_hybrid_retriever(cfg)
```

### 3. Add a cross-encoder reranker

```python
from app.retriever import (
    BM25Config, HybridConfig, SemanticConfig,
    api_aware_tokenize, build_hybrid_retriever, make_cross_encoder_reranker,
)

cfg = HybridConfig(
    semantic=SemanticConfig(k=10),
    bm25=BM25Config(k=10, preprocess_func=api_aware_tokenize),
    reranker=make_cross_encoder_reranker(),
    final_k=4,
)
retriever = build_hybrid_retriever(cfg)
```

Requires `sentence-transformers` to be installed. The `"reranked"` profile does exactly this.

### 4. Low-level: arbitrary ensemble

```python
from app.retriever import (
    BM25Config, SemanticConfig,
    get_bm25_retriever, get_semantic_retriever, get_hybrid_retriever,
)

sem = get_semantic_retriever(SemanticConfig(k=4))
bm  = get_bm25_retriever(BM25Config(k=6))
retriever = get_hybrid_retriever([(sem, 0.6), (bm, 0.4)])
```

`get_hybrid_retriever` takes any number of `(retriever, weight)` pairs — room to add MMR, a second semantic retriever, etc., without touching the hybrid code.

---

## Profile catalogue

| Profile      | Semantic k | BM25 k | Weights (sem/bm25) | Threshold | Reranker | Best for |
|--------------|:---:|:---:|:---:|:---:|:---:|---|
| `default`    | 4   | 4   | 0.5 / 0.5 | 0.55 | — | Unsure, balanced queries |
| `acronym`    | 6   | 6   | 0.2 / 0.8 | —    | — | `NISQ`, `VQE`, `mlflow.log_param` |
| `conceptual` | 4   | 4   | 0.7 / 0.3 | 0.45 | — | "What is X?", definitions |
| `overview`   | 10  | 10  | 0.5 / 0.5 | —    | — | "List all X", taxonomies |
| `reasoning`  | 8   | 8   | 0.5 / 0.5 | —    | — | Multi-hop "why", "how does X affect Y" |
| `reranked`   | 10  | 10  | 0.5 / 0.5 | —    | ✓ + final_k=4 | Hard/ambiguous queries; precision > latency |

All profiles use [`api_aware_tokenize`](#api-aware-tokenizer) for BM25.

---

## How callers pick a profile

Profile selection happens at two points in the system:

- **`rag_agent` graph** — the agent LLM picks via a tool argument. See [`app/graphs/rag_agent.py`](../app/graphs/rag_agent.py). The profile menu is injected into the system prompt from [`PROFILE_DESCRIPTIONS`](../app/retriever.py).
- **`query_rewriting` graph** — the helper LLM emits a `PROFILE:` line alongside the rewritten query. See [`app/prompts/query_rewrite.py`](../app/prompts/query_rewrite.py) and [`app/graphs/query_rewriting.py`](../app/graphs/query_rewriting.py).

Unknown names fall back to `"default"` with a warning.

---

## Architecture

```
                ┌──────────────────────────────────────┐
                │  get_profile_retriever("reranked")   │
                └──────────────────────────────────────┘
                                  │  (lookup + lazy cache)
                                  ▼
                   ┌───────────────────────────────┐
                   │  build_hybrid_retriever(cfg)  │
                   └───────────────────────────────┘
                                  │
                  ┌───────────────┼───────────────┐
                  ▼                               ▼
      ┌───────────────────────┐       ┌───────────────────────┐
      │ get_semantic_retriever│       │ get_bm25_retriever    │
      │   (Chroma top-k)      │       │ (BM25 over all docs,  │
      │                       │       │  optional tokenizer)  │
      └───────────────────────┘       └───────────────────────┘
                  │                               │
                  └───────────────┬───────────────┘
                                  ▼
                   ┌───────────────────────────────┐
                   │  get_hybrid_retriever         │
                   │  (EnsembleRetriever, RRF)     │
                   └───────────────────────────────┘
                                  │
                                  ▼  (only if reranker or final_k set)
                   ┌───────────────────────────────┐
                   │  _PostProcessingRetriever     │
                   │   rerank → trim to final_k    │
                   └───────────────────────────────┘
                                  │
                                  ▼
                              list[Document]
```

Layers, bottom-up:

1. **Config dataclasses** — `SemanticConfig`, `BM25Config`, `HybridConfig`. Each retriever type owns its own config.
2. **Factory functions** — `get_semantic_retriever(cfg)`, `get_bm25_retriever(cfg)`. One per retriever type.
3. **Combinator** — `get_hybrid_retriever([(r, w), ...])` wraps any number of retrievers in LangChain's `EnsembleRetriever` (RRF).
4. **Post-processing** — `_PostProcessingRetriever` wraps the base with reranker + `final_k`. Skipped when neither is set.
5. **One-shot builder** — `build_hybrid_retriever(cfg)` chains (2–4) from a single `HybridConfig`.
6. **Profiles** — `PROFILES` (eager) + `_PROFILE_FACTORIES` (lazy) + `get_profile_retriever(name)` with per-profile cache.

---

## API reference

### Config dataclasses

#### `SemanticConfig`
```python
@dataclass
class SemanticConfig:
    k: int = 4
    score_threshold: float | None = None   # 0–1 relevance; None = no filter
```

`score_threshold` is a **similarity** score (higher = better). Docs below it are dropped before RRF merging. Useful for acronym queries where embeddings return weakly-related MLflow docs at 0.43–0.53.

#### `BM25Config`
```python
@dataclass
class BM25Config:
    k: int = 4
    preprocess_func: Optional[Callable[[str], list[str]]] = None
```

`preprocess_func` must be applied to **both** indexed docs and queries. Leave `None` for default whitespace split; pass `api_aware_tokenize` for API-style tokens.

#### `HybridConfig`
```python
@dataclass
class HybridConfig:
    semantic: SemanticConfig         = SemanticConfig()
    bm25: BM25Config                 = BM25Config()
    semantic_weight: float           = 0.5
    bm25_weight: float               = 0.5
    rrf_c: int                       = 60          # RRF constant
    final_k: Optional[int]           = None        # trim after rerank
    reranker: Optional[Reranker]     = None        # (query, docs) -> docs
```

Default values exactly reproduce the pre-profile retriever (k=4, 0.5/0.5, no threshold, no rerank).

### Top-level functions

| Function | Returns | Purpose |
|---|---|---|
| `get_semantic_retriever(cfg)` | `BaseRetriever` | Chroma top-k (optional threshold) |
| `get_bm25_retriever(cfg)` | `BaseRetriever` | BM25 over entire corpus |
| `get_hybrid_retriever(pairs, *, rrf_c=60)` | `BaseRetriever` | RRF-fuse N retrievers |
| `build_hybrid_retriever(cfg)` | `BaseRetriever` | One-shot semantic+bm25+postproc |
| `get_profile_retriever(name)` | `BaseRetriever` | Named preset, lazy-cached |
| `api_aware_tokenize(text)` | `list[str]` | Tokenizer for API-style identifiers |
| `make_cross_encoder_reranker(model_name)` | `Reranker` | Build reranker callable (lazy ST import) |

### Constants

| Name | Type | Purpose |
|---|---|---|
| `PROFILES` | `dict[str, HybridConfig]` | Eager profile configs |
| `_PROFILE_FACTORIES` | `dict[str, Callable[[], HybridConfig]]` | Lazy profile factories (for expensive configs) |
| `PROFILE_NAMES` | `frozenset[str]` | All valid profile names (both dicts) |
| `PROFILE_DESCRIPTIONS` | `dict[str, str]` | Human-readable descriptions for LLM menus |

---

## API-aware tokenizer

`BM25Retriever`'s default tokenizer splits only on whitespace, so `"mlflow.log_param"` stays as one opaque token — a query for `log_param` matches nothing.

`api_aware_tokenize` also splits on `.` and `_`, **keeping the compound**:

```python
api_aware_tokenize("How do I call mlflow.log_param in Python?")
# → ["how", "do", "i", "call",
#    "mlflow.log_param", "mlflow", "log", "param",
#    "in", "python"]
```

So queries hit *either* the exact API name (`mlflow.log_param`) *or* the parts (`log_param`, `start_run`).

All shipped profiles use this tokenizer.

---

## Reranker

`make_cross_encoder_reranker()` returns a callable matching the `Reranker` signature:

```python
Reranker = Callable[[str, list[Document]], list[Document]]
```

It loads a cross-encoder model (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`) once, then re-scores every `(query, doc)` pair on each invocation. Slower than the base retrievers but much more accurate at ordering.

Typical pattern — **over-fetch then rerank**:

```python
HybridConfig(
    semantic=SemanticConfig(k=10),   # get more candidates...
    bm25=BM25Config(k=10),
    reranker=make_cross_encoder_reranker(),
    final_k=4,                        # ...trim to the best 4
)
```

---

## Why two profile dicts? (`PROFILES` vs `_PROFILE_FACTORIES`)

Python evaluates `PROFILES = { "default": HybridConfig(...), ... }` **at import time** — every HybridConfig in the dict is built the moment anything imports `retriever.py`.

That's fine for cheap configs (just numbers). It's a problem when a config needs to load a ~90 MB cross-encoder model: every startup, every test run, every `python -c "from app.retriever import ..."` would pay that cost even if nobody uses the profile.

**Solution:** wrap the expensive construction in a nullary function (a *factory*):

```python
_PROFILE_FACTORIES = {
    "reranked": lambda: HybridConfig(..., reranker=make_cross_encoder_reranker()),
}
```

The lambda is a cheap function object. The model only loads when `get_profile_retriever("reranked")` is actually called. The result is cached, so the cost is paid **at most once per process**.

`PROFILE_NAMES = frozenset(PROFILES) | frozenset(_PROFILE_FACTORIES)` is the union — callers validate against this, not against `PROFILES` alone.

Rule of thumb:
- Profile just sets numbers/weights? → put it in `PROFILES`.
- Profile loads a model, opens a connection, or does anything slow? → put a factory in `_PROFILE_FACTORIES`.

---

## Extending

### Add a new profile

Cheap profile — add to `PROFILES`:

```python
PROFILES["mycustom"] = HybridConfig(
    semantic=SemanticConfig(k=5, score_threshold=0.5),
    bm25=BM25Config(k=5, preprocess_func=api_aware_tokenize),
    semantic_weight=0.6,
    bm25_weight=0.4,
)
PROFILE_DESCRIPTIONS["mycustom"] = "Short description for the LLM menu."
```

Expensive profile (loads a model, etc.) — add to `_PROFILE_FACTORIES`:

```python
_PROFILE_FACTORIES["myheavy"] = lambda: HybridConfig(
    ...,
    reranker=make_cross_encoder_reranker("some/other-model"),
)
PROFILE_DESCRIPTIONS["myheavy"] = "..."
```

Then update consumers that declare a fixed set of names:
- [`app/graphs/rag_agent.py`](../app/graphs/rag_agent.py) — `RetrievalProfile` Literal and tool docstring.
- [`app/prompts/query_rewrite.py`](../app/prompts/query_rewrite.py) — `REWRITE_PROMPT_V3` menu.

### Add a new retriever type (e.g. MMR)

1. Define `MMRConfig` dataclass and `get_mmr_retriever(cfg)` factory.
2. Either add an `mmr: Optional[MMRConfig]` field to `HybridConfig` (if it should plug into existing profiles), or compose directly via `get_hybrid_retriever`:

```python
get_hybrid_retriever([
    (get_semantic_retriever(SemanticConfig(k=4)), 0.5),
    (get_bm25_retriever(BM25Config(k=4)), 0.3),
    (get_mmr_retriever(MMRConfig(...)), 0.2),
])
```

### Write a custom tokenizer

Any `Callable[[str], list[str]]` works. Apply the same function to docs and queries automatically by passing it via `BM25Config.preprocess_func`.

### Write a custom reranker

Any `Callable[[str, list[Document]], list[Document]]` works. Pass via `HybridConfig.reranker`. Common patterns: LLM-as-judge, MMR diversification, metadata-aware boosting.

---

## Gotchas

- **BM25 rebuild cost.** `get_bm25_retriever` loads the entire corpus from Chroma. Call it once at graph initialisation, not per-query. Profiles cache this automatically.
- **Threshold is similarity, not distance.** `score_threshold=0.55` keeps docs **scoring ≥ 0.55** (higher = better). LangChain converts Chroma's raw L2 distance internally.
- **Tokenizer symmetry.** If you pass `preprocess_func`, it applies to both the BM25 index and incoming queries. Different functions on either side will silently miss matches.
- **Reranker is opt-in.** None of the default 5 profiles use one. Only `"reranked"` does, and only when selected.
- **Import-time side effects.** Keep `PROFILES` cheap. Anything slow goes in `_PROFILE_FACTORIES`.

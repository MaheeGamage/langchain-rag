# RAG Embedding Strategy for AI-Assisted Experiment Tracking in Quantum Software Development

**Thesis:** AI-Assisted Experiment Tracking for Quantum Software Development
**Author:** Mahee Hewa Gamage
**Degree Program:** Master's Degree Program in Artificial Intelligence

---

## Overview

The system serves two primary use cases with distinct retrieval demands:

- **Use Case 1 (Advisory):** Semantic similarity — answering conceptual questions about MLflow and QProv in natural language.
- **Use Case 2 (Code Injection):** Syntactic precision — retrieving exact API signatures and usage patterns to inject tracking code into quantum programs.

This asymmetry motivates a hybrid retrieval architecture with document-type-aware chunking.

---

## 1. Heterogeneous Chunking by Document Type

Rather than applying a uniform fixed-size token window, chunks follow the natural structure of each source type. This aligns with the *structure-aware chunking* approach described in Gao et al. (2023).

### MLflow API Reference (`.rst` files)

- **Unit:** One autodoc function or class directive per chunk.
- **Content:** Function signature + docstring + parameter table + return type + inline code example.
- **Target size:** 150–400 tokens.
- **Rationale:** Keeps the API contract atomic. Code examples are retained in the same chunk to support UC2 code generation.

### Conceptual Documentation (`.mdx` files)

- **Unit:** One H2 or H3 section per chunk.
- **Content:** Full section prose with 50-token overlap at boundaries.
- **Target size:** 300–500 tokens.
- **Rationale:** Preserves the reasoning behind the API, which is the primary retrieval target for UC1 advisory queries.

### Synthetic Documents (`qiskit_experiment_tracking_bridge.md`, `qprov_taxonomy.md`)

- **Unit:** Sliding window over paragraphs.
- **Content:** ~400-token chunks with 20% overlap.
- **Rationale:** These cross-domain documents synthesize MLflow and quantum computing knowledge. Generous overlap preserves cross-referential context across chunk boundaries.

### Academic Papers (`orig_paper/`)

- **Unit:** Section-level chunks (Abstract, Introduction, Methodology, etc.).
- **Special rule:** The abstract is stored as a standalone chunk and prepended as a prefix to every other chunk from the same paper (*parent document retriever* pattern, Shi et al., 2023).
- **Rationale:** Ensures any retrieved passage carries sufficient framing context for generation.

---

## 2. Hybrid Embedding: Dense + Sparse

A purely dense approach underweights low-frequency API tokens (e.g., `mlflow.log_metric`, `mlflow.MlflowClient.create_experiment`). A hybrid strategy combines the strengths of both paradigms.

### Dense Embeddings (Semantic Similarity)

- **Recommended models:**
  - `text-embedding-3-large` (OpenAI) — strong on mixed prose and code.
  - `nomic-embed-text-v1.5` (open-weight) — competitive MTEB scores, runs locally on Qubernetes.
- **Primary use:** UC1 advisory queries requiring semantic closeness.

### Sparse Retrieval (BM25)

- **Implementation:** Standard inverted index over raw chunk text (e.g., `rank_bm25` or Elasticsearch).
- **Primary use:** UC2 queries containing exact API names where precise token matching outperforms dense search.
- **Basis:** Robertson & Zaragoza (2009).

### Fusion: Reciprocal Rank Fusion (RRF)

- Merge BM25 and dense ranked lists using RRF (Cormack et al., 2009).
- RRF is parameter-free and empirically outperforms weighted linear combination for heterogeneous corpora.
- The query router can increase the BM25 weight for code-centric UC2 queries.

---

## 3. Metadata-Enriched Index

Each chunk carries structured metadata to enable pre-filtering before embedding retrieval, reducing noise and improving latency.

| Field | Values | Purpose |
|---|---|---|
| `doc_type` | `api_ref`, `conceptual`, `synthetic`, `paper` | Route UC1 toward conceptual/synthetic; UC2 toward `api_ref` |
| `source_lib` | `mlflow`, `qiskit`, `qprov`, `general` | Filter by domain relevance |
| `content_form` | `prose`, `code`, `mixed` | Upweight code chunks for UC2 |
| `section_depth` | `module`, `function`, `class`, `section` | Fine-grained retrieval granularity |

**UC1 filter:** `doc_type IN (conceptual, synthetic, paper)`

**UC2 filter:** Include `api_ref`; bias toward `content_form = code`

---

## 4. Hierarchical (Parent-Child) Retrieval for UC2

For code injection, the system needs not just a function signature but the surrounding module context (required imports, object ownership, related methods).

- **Child chunks (~200 tokens):** Individual function/method chunks used for retrieval.
- **Parent chunks (~1000 tokens):** Full module-level section containing the retrieved function, returned at generation time.

This *small-to-big retrieval* pattern (Shi et al., 2023) ensures the retriever operates on precise units while the generator receives richer context — critical for MLflow's object-oriented API where `mlflow.start_run()`, `mlflow.log_param()`, and `mlflow.end_run()` form a single usage pattern.

---

## Strategy Summary Table

| Strategy | Applies To | Primary Benefit |
|---|---|---|
| Function-level chunking | API `.rst` files | Atomic API contracts for UC2 |
| Section-level chunking + overlap | Conceptual `.mdx` | Preserves explanatory context for UC1 |
| Sliding window chunking | Synthetic `.md` | Retains cross-domain synthesis |
| Abstract-prefixed section chunking | Academic papers | Contextual framing in retrieved passages |
| Dense embeddings | All docs | Semantic similarity for UC1 |
| BM25 sparse index | All docs | Exact API token match for UC2 |
| RRF fusion | Query time | Combines ranked lists without tuning |
| Metadata pre-filtering | Query time | Reduces retrieval noise per use case |
| Parent-child index | API `.rst` files | Module context for code injection (UC2) |

---

## DSR Evaluation Note

Within the Design Science Research methodology, the chunking boundaries and BM25/dense weight balance are artefact design decisions that can be evaluated iteratively. A set of hand-crafted Q&A pairs — one set per use case — provides ground-truth queries for measuring retrieval precision across DSR design cycles.

---

## References

- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
- Gao, Y., et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey*. arXiv:2312.10997.
- Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP 2020.
- Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval.
- Cormack, G., et al. (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*. SIGIR 2009.
- Shi, W., et al. (2023). *REPLUG: Retrieval-Augmented Language Model Pre-Training*. arXiv:2301.12652. *(cited for parent-document retrieval pattern)*

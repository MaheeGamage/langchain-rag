# Retrieval Quality Investigation — 2026-04-16

## Motivation

The RAG system was returning answers that did not reflect the content of the indexed knowledge base. Semantic search queries against ChromaDB were consistently surfacing documents that were not meaningfully relevant to the user's question — even for queries closely related to the domain the system was built for (quantum software experiment tracking). This indicated a problem at the retrieval layer: either the documents were not indexed correctly, the embeddings were misaligned, or the corpus itself was not representative of the intended domain.

The investigation was triggered by observing that the system could not answer quantum-specific questions accurately despite the knowledge base containing academic papers and synthetic documents on the topic. The suspicion was that the issue could be in the ingestion pipeline (bad chunking, wrong metadata, failed embedding) or in the retrieval step (wrong distance metric, missing documents, poor ranking). Both were examined.

---

## Summary

The embedding and retrieval pipeline is functionally correct — ChromaDB is reachable, 1,550 chunks are indexed, embeddings are normalized (cosine-ready at dim=768 via `nomic-embed-text`), and direct queries return semantically relevant results when matching content exists. The root causes of poor retrieval quality are corpus imbalance, a corpus mislabeling bug, and the absence of retrieval guardrails.

---

## Environment

| Setting | Value |
|---------|-------|
| Embedding provider | Ollama (`nomic-embed-text`) |
| Embedding dimension | 768 |
| ChromaDB collection | `nomic_embed_text` |
| Distance metric | L2 (embeddings are normalized so ranking is equivalent to cosine) |
| Chunks indexed | 1,550 |
| Retriever k | 4 |

---

## Corpus Breakdown (at time of investigation)

| Source corpus | Chunks | % of total |
|---------------|--------|-----------|
| `mlflow`      | 1,388  | 89.5%     |
| `unknown`*    | 84     | 5.4%      |
| `qprov`       | 58     | 3.7%      |
| `qiskit`      | 20     | 1.3%      |

\* The `unknown` corpus contains the 4 academic papers (see Root Cause 2 below).

---

## Root Cause 1: Corpus imbalance (most impactful)

89.5% of indexed content is MLflow documentation. The quantum-specific content is:
- 4 academic papers → ~127 chunks
- 2 synthetic bridge documents → ~35 chunks
- **No Qiskit SDK or quantum circuit documentation**

The content root (`knowledge_ingestion/content/v3/content/`) contains:

```
original paper/   — 4 .md files  (academic papers)
synth_docs/       — 2 .md files  (synthetic cross-domain docs)
tech_docs/mlflow/ — 65 .mdx + 20 .ipynb files (MLflow docs)
```

No Qiskit technical documentation exists in the corpus. When asking quantum-specific questions, MLflow chunks dominate the top-k results purely by volume — the "closest" 4 results can all be MLflow content even when they are semantically distant from the query.

**Fix:** Add Qiskit SDK documentation (or other quantum-specific technical docs) under `tech_docs/`.

---

## Root Cause 2: Academic papers are mislabeled as `source_corpus: 'unknown'`

`ParserStage._detect_corpus()` matches path substrings against `DEFAULT_CORPUS_KEYWORDS`:

```python
# knowledge_ingestion/ingest_v2/pipeline/stages/parser.py
DEFAULT_CORPUS_KEYWORDS: dict[str, str] = {
    "mlflow":                 "mlflow",
    "qiskit":                 "qiskit",
    "qprov":                  "qprov",
    "sample-quantum-circuit": "sample",
    "orig_paper":             "paper",      # <-- expects hyphen
    "original-content":       "paper",      # <-- expects hyphen
}
```

The actual folder name is `original paper` (with a **space**, not a hyphen). Neither `"orig_paper"` nor `"original-content"` matches, so all 4 academic papers receive `source_corpus: 'unknown'`.

### Cascading effect on chunking

`ChunkingStage._select_strategy()` routes to `PaperChunkingStrategy` only if the corpus is in `PAPER_CORPORA`:

```python
# knowledge_ingestion/ingest_v2/pipeline/stages/chunker.py
PAPER_CORPORA: frozenset[str] = frozenset({"paper", "orig_paper"})

if corpus in PAPER_CORPORA:
    return self.strategy_map["paper"]  # never reached for these papers
```

Because the corpus is `'unknown'`, papers fall through to `NarrativeChunkingStrategy`. They are chunked without abstract-prefixing, losing the "parent document retriever" pattern that `PaperChunkingStrategy` provides.

**Fix:** Add `"original paper": "paper"` to `DEFAULT_CORPUS_KEYWORDS` in `parser.py`.

---

## Root Cause 3: 63 RST files silently skipped

`WalkerStage` explicitly excludes `.rst` files:

```python
# knowledge_ingestion/ingest_v2/pipeline/stages/walker.py
SKIP_EXTENSIONS: frozenset[str] = frozenset({
    ".rst",     # Sphinx directive stubs — too terse without docstring resolution
    ...
})
```

There are 63 `.rst` files under `tech_docs/mlflow/docs/api_reference/` containing MLflow Python API reference documentation. These are intentionally excluded (Sphinx stubs without resolved docstrings), but they represent a gap in API-level coverage.

---

## Root Cause 4: No retrieval quality guardrails

`app/retriever.py` uses a bare top-k retriever:

```python
return vectorstore.as_retriever(search_kwargs={"k": 4})
```

No score threshold is applied, so all 4 results are returned regardless of relevance. With a heavily skewed corpus, this means low-relevance MLflow chunks can be passed to the LLM as context for quantum-specific queries.

**Observed distances** (L2, lower = more similar):
- Highly relevant query (e.g. "QProv quantum provenance"): ~0.30–0.35
- Moderately relevant: ~0.50–0.55
- Irrelevant but returned: ~0.65–0.70

**Fix options:**
- Add `score_threshold` to `search_kwargs` (e.g. `{"k": 4, "score_threshold": 0.55}`)
- Filter by `source_corpus` metadata at query time when the domain is known
- Increase `k` to retrieve more candidates, then re-rank or filter

---

## Recommended fixes (priority order)

| Priority | Issue | Change |
|----------|-------|--------|
| 1 | MLflow docs dominate corpus | Add Qiskit SDK docs to `knowledge_ingestion/content/v3/content/tech_docs/` and re-ingest |
| 2 | Papers labeled `unknown` | Add `"original paper": "paper"` to `DEFAULT_CORPUS_KEYWORDS` in `parser.py:31` |
| 3 | Papers skip abstract-prefix chunking | Follows automatically from fix #2 |
| 4 | No score threshold | Add `score_threshold` to `app/retriever.py` or implement post-retrieval filtering |

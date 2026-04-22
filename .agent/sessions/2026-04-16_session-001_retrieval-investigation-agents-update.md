# Session: 2026-04-16 #001

## Goal
Investigate poor retrieval quality in the RAG system (semantic search returning irrelevant results), document findings, and update `AGENTS.md` to reflect the current ingest pipeline architecture.

## Prompts Summary
- User reported that retrieval was not returning relevant results and asked for an investigation of both the ingestion pipeline (`knowledge_ingestion/ingest_v2`) and retrieval logic (`invoke_retriever_with_retry`), with permission to query ChromaDB directly at `localhost:8001`.
- User asked for the analysis to be saved to a `docs/` subfolder with the date in the filename.
- User asked for a Motivation section to be added to the analysis document.
- User asked to update `AGENTS.md` and any related files as directed by `AGENTS.md` conventions.
- User asked agents to treat `docs/analysis/` files as read-only after completion, editable only on explicit human instruction.

## Actions Taken

### Investigation
- Queried ChromaDB HTTP API at `localhost:8001` to confirm connectivity and inspect the `nomic_embed_text` collection (1,550 chunks).
- Verified embedding normalization (norms ≈ 1.0) — confirmed L2 metric is equivalent to cosine for this collection.
- Ran `similarity_search_with_score` for several quantum-specific queries; confirmed that MLflow content dominated results due to corpus imbalance.
- Inspected `WalkerStage` — found 63 `.rst` files explicitly skipped.
- Inspected `ParserStage._detect_corpus()` — found that the `original paper/` folder (with a space) matches no keyword in `DEFAULT_CORPUS_KEYWORDS`, causing all academic papers to receive `source_corpus: 'unknown'` instead of `'paper'`.
- Traced cascading effect: `ChunkingStage` routes `'unknown'` corpus to `NarrativeChunkingStrategy` instead of `PaperChunkingStrategy`, so papers never receive abstract-prefixed chunks.
- Confirmed corpus breakdown: 1,388/1,550 chunks (89.5%) are MLflow docs; only 162 chunks cover quantum topics.
- Confirmed no Qiskit SDK documentation exists under `knowledge_ingestion/content/v3/content/`.

### Documentation
- Created `docs/analysis/` directory.
- Created `docs/analysis/2026-04-16-retrieval-quality-investigation.md` with: motivation, environment table, corpus breakdown, four root causes with code references, and a prioritized fix table.

### AGENTS.md updates
- **Section 2 (Build & Run)**: Updated ingest command from `python -m app.ingest` to `python -m knowledge_ingestion.ingest_v2.ingest`; updated Docker compose ingest command to match; updated log file name from `ingest.log` to `ingest_pipeline.log`.
- **Section 3 (Project Structure)**: Removed non-existent `app/ingest.py`; expanded `app/` entries with `factory.py`, `models.py`, `schemas.py`; added full `knowledge_ingestion/` subtree (v1, v2 stages and strategies, content layout); added `docs/` and `experimentation/` top-level entries; updated log file name.
- **Section 4 (Development Patterns)**: Replaced "Ingest batching/dedup/error-tolerance" paragraphs (which referenced the deleted `app/ingest.py`) with an accurate description of the v2 pipeline stages, corpus detection, and chunking strategies. Updated `BATCH_SIZE` reference to point to `app/config.py` instead of `ingest.py`. Updated `EmbedderStage` and `_upsert_batch()` references for dedup and error-tolerance patterns.
- **Section 7 (What Not to Do)**: Updated the "Don't run ingest directly" rule to reference the correct module path.
- **`docs/analysis/` read-only convention** added in three places: Section 3 project structure comment, Section 5 code style rules, and Section 7 what-not-to-do list.

## Outcome
Retrieval quality issues fully documented with root causes and recommended fixes. `AGENTS.md` now accurately reflects the v2 ingest pipeline architecture and enforces the `docs/analysis/` read-only convention in three places (Sections 3, 5, and 7). No code was changed; all fixes identified are tracked in the analysis document.

## Agent
Claude Sonnet 4.6 (Claude Code)

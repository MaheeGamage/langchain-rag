# Agent Guidance

General instructions for any AI agent (GitHub Copilot, Claude, Cursor, etc.)
working in this repository. Read this file before making any changes.

> **Keeping this file current:** If you add a feature, change a pattern, or
> discover a new pitfall, update the relevant section(s) of this file as part
> of the same change. Do **not** log edits to `AGENTS.md` here — record them
> in a session log under `.agent/sessions/` as you would for any other code
> change.

---

## 1. Project Overview

This is a **local RAG (Retrieval-Augmented Generation) system**. The core loop is:

```
PDFs → ingest → ChromaDB (embeddings) → retriever → LangGraph → answer
```

Supported providers — `LLM_PROVIDER` and `EMBEDDING_PROVIDER` are set **independently**
in `.env`, so any combination works:

| Provider | LLM default | Embedding default |
|---|---|---|
| `ollama` | `tinyllama` | `nomic-embed-text` |
| `openai` | `gpt-4o-mini` | `text-embedding-3-small` |
| `gemini` | `gemini-2.5-flash` | `gemini-embedding-001` |

Example mixed config (local embeddings + cloud LLM):
```
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=ollama
```
`COLLECTION_NAME` is derived from the embedding model automatically, so the
ChromaDB collection always matches the active embedding provider.

---

## 2. Build & Run Commands

### Environment

The project uses **Poetry** with a local `.venv/` at the repo root.

```bash
# Install / sync all dependencies
poetry install

# If pip-installing directly (e.g. a quick one-off package):
source .venv/bin/activate && pip install <package>
# Then also add it to pyproject.toml manually.
```

> Always use the `.venv` interpreter. Never use a system-level `python`.

### Ingest content into ChromaDB

```bash
poetry run python -m knowledge_ingestion.ingest_v2.ingest
# or
source .venv/bin/activate && python -m knowledge_ingestion.ingest_v2.ingest
```

- Content root is controlled by `DATA_ROOT` in `.env` (default: `./knowledge_ingestion/content/v3/content`).
- Logs go to `ingest_pipeline.log` (auto-created at repo root). Tail with `tail -f ingest_pipeline.log`.
- Embeddings are written to ChromaDB over HTTP (`CHROMA_HOST:CHROMA_PORT`).
- `ingest_v1/` is the older single-file pipeline — kept for reference. **Use `ingest_v2` for all new ingestion.**

### Run both servers (recommended)

```bash
python run.py
# FastAPI   → http://localhost:8000
# Streamlit → http://localhost:8501
# Ctrl+C stops both
```

`run.py` launches Uvicorn and Streamlit as subprocesses, pipes their output
with `[api]` / `[ui ]` prefixes, and shuts both down cleanly on Ctrl+C.

### Run the FastAPI server standalone

```bash
source .venv/bin/activate && uvicorn app.api:app --port 8000 --reload
```

### Run the Streamlit UI standalone

```bash
source .venv/bin/activate && streamlit run ui/streamlit_app.py
# Opens in browser at http://localhost:8501
```

### Check Ollama

```bash
ollama list           # confirm models are available
ollama serve          # start if not running (usually auto-started)
```

### Run with Docker

```bash
# Build and start all services (Ollama + ChromaDB + API + UI)
docker compose up --build

# Ingest content inside Docker (set DATA_ROOT in .env first)
docker compose run --rm api python -m knowledge_ingestion.ingest_v2.ingest

# Full rebuild from scratch (re-downloads Ollama models)
docker compose down -v && docker compose up --build
```

Image layout:
- The `api` service builds and tags the shared image as `langchain-rag`.
- The `ui` service reuses `langchain-rag` via `image: langchain-rag` (no `build:`).
- Chroma vectors are stored in the `chroma_data` named volume (via the `chroma` service).
- `data/` is bind-mounted from the host — never baked into the image.
- Ollama model weights are stored in the `ollama_data` named volume.
- `OLLAMA_BASE_URL` is overridden to `http://ollama:11434` inside containers so they
  resolve the Ollama service by name rather than `localhost`.
- Chroma env vars are overridden to `CHROMA_HOST=chroma`, `CHROMA_PORT=8000`
  inside containers so the API resolves the Chroma service by name.

---

## 3. Project Structure

```
app/
  config.py       — Central config: model names, paths, chunk sizes. Change models here.
  vectorstore.py  — Builds Chroma HTTP vectorstore client
  retriever.py    — Wraps ChromaDB as a LangChain retriever (k=4)
  factory.py      — Only place that imports provider-specific LLM/embedding packages
  graph.py        — Single graph export point (selects implementation via RAG_GRAPH_IMPLEMENTATION)
  graphs/
    baseline.py   — Baseline retrieve → generate graph
    agentic.py    — Agentic/iterative retrieval graph
    common.py     — Shared graph utilities (BASE_PROMPT, invoke_retriever_with_retry, etc.)
    types.py      — Shared graph state type (GraphState)
  api.py          — FastAPI app exposing POST /query and GET /config
  models.py       — Shared data models (ContextEntry, etc.)
  schemas.py      — API request/response schemas
  __init__.py

knowledge_ingestion/
  ingest_v2/               — Active modular ingestion pipeline (use this)
    ingest.py              — Entry point: poetry run python -m knowledge_ingestion.ingest_v2.ingest
    pipeline/
      pipeline.py          — IngestPipeline: wires Walker → Parser → Chunker → Embedder
      stages/
        walker.py          — Stage 1: discover files, skip .rst/.json/.csv etc.
        parser.py          — Stage 2: route files to parsers, stamp source_corpus metadata
        chunker.py         — Stage 3: route each doc to the right ChunkingStrategy
        embedder.py        — Stage 4: batch-embed chunks into Chroma with dedup + retry
      strategies/
        chunking.py        — NarrativeChunkingStrategy, CodeChunkingStrategy,
                             SyntheticDocChunkingStrategy, PaperChunkingStrategy,
                             PlainTextChunkingStrategy
  ingest_v1/               — Legacy single-file pipeline (kept for reference only)
  content/
    v3/content/            — Active knowledge base (DATA_ROOT points here)
      original paper/      — Academic papers (Markdown)
      synth_docs/          — Synthetic cross-domain documents
      tech_docs/           — Technical documentation (MLflow, etc.)

ui/
  streamlit_app.py  — Streamlit UI (chat interface, pure HTTP client)

docs/
  analysis/         — Investigation and analysis reports (one .md per investigation)
                      Files here are READ-ONLY once written. Do not edit without
                      explicit human instruction. Add a new dated file instead.
  *.md              — Other project documentation

experimentation/
  evaluation/       — RAGAS evaluation scripts and notebooks
  testset-generation/ — Testset generation scripts
  pdf-to-markdown/  — PDF pre-processing utilities

run.py            — Starts both servers locally (no Docker)
Dockerfile        — Two-stage build; produces the shared `langchain-rag` image
docker-compose.yml — Five services: ollama, ollama-init, chroma, api, ui
.dockerignore     — Excludes .venv/, chroma_db/, data/, .env, etc.
pyproject.toml    — Dependencies (Poetry / PEP 621 hybrid)

ingest_pipeline.log — Last ingest run log (not committed)

.agent/
  README.md       — Logging conventions (read before writing a session log)
  sessions/       — One .md file per agent session
```

---

## 4. Development Patterns

### Configuration

All tuneable values live in `app/config.py`. **Do not hard-code model names or
paths anywhere else.**

```python
# app/config.py — provider switches + graph switch
LLM_PROVIDER       = os.getenv("LLM_PROVIDER",       "ollama")  # ollama | openai | gemini
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")  # ollama | openai | gemini
RAG_GRAPH_IMPLEMENTATION = os.getenv("RAG_GRAPH_IMPLEMENTATION", "baseline")  # baseline | agentic

# Resolved independently
LLM_MODEL,    LLM_API_KEY,    LLM_BASE_URL       = ...  # from _LLM_DEFAULTS[LLM_PROVIDER]
EMBEDDING_MODEL, EMBEDDING_API_KEY, EMBEDDING_BASE_URL = ...  # from _EMBEDDING_DEFAULTS[EMBEDDING_PROVIDER]
COLLECTION_NAME = ...  # derived from EMBEDDING_MODEL — keeps collections separate
CHROMA_HOST = ...      # default localhost for host-run API/ingest in development
CHROMA_PORT = ...      # default 8001 (published Chroma port)
CHROMA_TARGET = ...    # http(s)://host:port
```

To switch providers, set `LLM_PROVIDER` and/or `EMBEDDING_PROVIDER` in `.env`
(copy `.env.example`). They are fully independent.  
To override a specific model within a provider, set e.g. `OLLAMA_LLM_MODEL=llama3` in `.env`.

> **Note:** each embedding model gets its own ChromaDB collection (`COLLECTION_NAME`).
> Switching `EMBEDDING_PROVIDER` requires a re-ingest. Switching only `LLM_PROVIDER`
> does **not** require re-ingest.
> Switching Chroma instances requires a re-ingest to populate the target backend.

### Provider / LLM factory

`app/factory.py` is the **only** place that imports provider-specific packages
(`langchain_ollama`, `langchain_openai`, `langchain_google_genai`).  
Everywhere else calls:

```python
from .factory import get_llm, get_embeddings

llm        = get_llm()        # dispatches on LLM_PROVIDER
embeddings = get_embeddings() # dispatches on EMBEDDING_PROVIDER
```

To add a new provider (e.g. Anthropic):
1. Add its block to `_LLM_DEFAULTS` and/or `_EMBEDDING_DEFAULTS` in `config.py`.
2. Add an `if LLM_PROVIDER == "anthropic":` branch in `get_llm()` in `factory.py`.
3. Add an `if EMBEDDING_PROVIDER == "anthropic":` branch in `get_embeddings()` if needed.
4. `poetry add langchain-anthropic` and update `pyproject.toml`.
5. No changes needed in `ingest.py`, `retriever.py`, or `graph.py`.

### Graph implementation switch

`app/graph.py` is the only module that chooses which graph implementation is active.
It reads `RAG_GRAPH_IMPLEMENTATION` from `app/config.py` and exports the selected `graph`.

- `baseline` → `app/graphs/baseline.py`
- `agentic` → `app/graphs/agentic.py`

Do not import `app.graphs.*` directly from API/UI or evaluators; import from `app.graph`.

### LangGraph state

The shared graph state is defined in `app/graphs/types.py`:

```python
class GraphState(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]
  context: list[ContextEntry]
  retrieved: list[ContextEntry]
```

- Add new fields here when extending the pipeline (e.g. `rewritten_question`, `citations`).
- Each node receives and returns a **partial state dict** — only return the keys you changed.

### Adding a new graph node

1. Define a function `def my_node(state: GraphState) -> dict:` in the target implementation file under `app/graphs/`.
2. Register it: `builder.add_node("my_node", my_node)`.
3. Wire edges: `builder.add_edge("previous_node", "my_node")`.
4. Return only the state keys the node modifies.

### Retriever

`get_retriever()` in `retriever.py` returns a LangChain `VectorStoreRetriever`.
`search_kwargs={"k": 4}` controls how many chunks are returned. Increase `k`
for more context; decrease for speed.

### Ingest pipeline (v2)

The active ingest pipeline lives in `knowledge_ingestion/ingest_v2/` and has four composable stages:

```
WalkerStage → ParserStage → ChunkingStage → EmbedderStage
```

Each stage is a plain object with a `run()` method. Swap any stage by passing a replacement to `IngestPipeline` — nothing else changes. Entry point:

```bash
poetry run python -m knowledge_ingestion.ingest_v2.ingest
```

**Corpus detection** — `ParserStage` stamps every document with a `source_corpus` field by matching path substrings against `DEFAULT_CORPUS_KEYWORDS` in `parser.py`. First match wins; unmatched files get `"unknown"`. The corpus label drives chunking strategy selection in `ChunkingStage`.

**Chunking strategies** — defined in `pipeline/strategies/chunking.py`:
- `NarrativeChunkingStrategy` — two-pass: H1/H2/H3 header split then size enforcement
- `CodeChunkingStrategy` — size-only split respecting `def`/`class` boundaries
- `SyntheticDocChunkingStrategy` — sliding window with generous overlap for cross-domain docs
- `PaperChunkingStrategy` — abstract-prefixed section chunks (parent-doc pattern for academic papers)
- `PlainTextChunkingStrategy` — flat sliding-window with no structural awareness

### Ingest batching

Embeddings are sent to the provider in batches of `BATCH_SIZE = 25` chunks (configured in `app/config.py`). Lower this if the embedding provider OOMs; raise it to speed up ingest on machines with more VRAM.

### Ingest deduplication

`EmbedderStage` assigns a deterministic MD5 ID from chunk content and skips duplicate IDs within a single ingest run. This prevents Chroma upsert errors from duplicate IDs. The strategy is implemented in `knowledge_ingestion/ingest_v2/pipeline/stages/embedder.py` (`_doc_id()`). Note: this is a within-run guard only — it does not check what is already in Chroma from prior runs.

### Ingest error tolerance

If a batch upsert fails, `EmbedderStage._upsert_batch()` retries per document and skips only the failing chunks, logging their IDs and source paths. This keeps ingestion running even when some documents are malformed or trigger provider errors.

### Streamlit UI

`ui/streamlit_app.py` is a **pure presentation layer** — it does **not** import
`app.graph` or any backend modules. All RAG work (retrieval, generation) happens
via HTTP calls to the FastAPI `/query` endpoint.

- `API_URL` env var controls the backend address (default: `http://localhost:8000`,
  overridden to `http://api:8000` in Docker).
- Model info is fetched from `GET /config` for display in the sidebar.
- Chat history is stored in `st.session_state.messages`.
- Each message dict: `{"role": "user"|"assistant", "content": str, "sources": list}`.
- Source documents are shown in a collapsible expander below each answer.

---

## 5. Code Style & Structure Rules

- **One concern per file.** `graph.py` = graph selection/export only; `app/graphs/*.py` = implementation logic; `retriever.py` = retriever only; `vectorstore.py` = Chroma client construction only.
- **No hard-coded strings** for model names or paths — always import from `config.py`.
- **Never import provider-specific LLM/embedding classes outside `factory.py`** — use `get_llm()` / `get_embeddings()` instead.
- **No `vectorstore.persist()`** — Chroma ≥ 0.4 auto-persists on write.
- **Import from `langchain_chroma`**, not `langchain_community.vectorstores.Chroma`
  (the community import is deprecated).
- **Import from `langchain_text_splitters`**, not `langchain.text_splitter`
  (the old path was removed in LangChain 1.x).
- Keep `streamlit_app.py` inside `ui/`, not at the repo root; it is a
  presentation layer, not part of the core pipeline.
- When adding new dependencies, update **both** the `.venv` (via pip/poetry)
  **and** `pyproject.toml`.
- **`docs/analysis/` files are read-only after creation.** Never edit a completed
  analysis document unless the human explicitly asks. If a follow-up investigation
  is needed, create a new dated file (e.g. `2026-04-17-retrieval-followup.md`).

---

## 6. Agent Logging

All sessions **must** be logged. See [`.agent/README.md`](.agent/README.md) for
the full convention and template. This file intentionally does **not** duplicate
those instructions.

Key reminder:
1. If you update `AGENTS.md`, record exactly which sections changed and why
   in the session log — `AGENTS.md` itself carries no change history.

Find sessions that touched a specific file:
```bash
grep -rl "app/graph.py" .agent/sessions/
```

---

## 7. What Not to Do

- **Don't call `vectorstore.persist()`** — throws a deprecation error with Chroma ≥ 0.4.
- **Don't embed all chunks in one `Chroma.from_documents()` call** — it blocks
  silently for minutes with no progress. Always use batched `add_documents()`.
- **Don't use `langchain.text_splitter`** — that module was removed; use `langchain_text_splitters`.
- **Don't change `requires-python`** in `pyproject.toml` to open-ended `>=3.12`
  without an upper bound — `langchain-ollama` requires `<4.0.0`.
- **Don't run `python ingest.py` directly** — run as a module: `python -m knowledge_ingestion.ingest_v2.ingest`. Do not use `ingest_v1` for new ingestion runs.
- **Don't hard-code model names or provider-specific classes** outside `config.py` / `factory.py`.
- **Don't add `if PROVIDER == ...` branches outside `factory.py`** — all provider dispatch lives there.
- **Don't add graph-selection branches outside `app/graph.py`** — all graph implementation dispatch lives there.
- **Don't spread Chroma connection logic across files** — keep Chroma client construction centralized in `app/vectorstore.py`.
- **Don't import `app.graph` or backend modules in `streamlit_app.py`** — the UI
  must remain a pure HTTP client. All RAG logic goes through the FastAPI API.
- **Don't add `build:` to the `ui` service** in `docker-compose.yml` — it reuses the
  `langchain-rag` image built by `api`. Adding `build:` to `ui` with the same `image:` name
  causes a conflict ("image already exists" error) because both services would try to tag
  the same name simultaneously.
- **Don't pass `ragas.metrics.collections.*` metrics to `ragas.evaluate()`** in Ragas 0.4 —
  `evaluate()` still validates legacy `Metric` types. For collections metrics, run
  `metric.batch_score(...)` / `metric.abatch_score(...)` directly (or use the experiment API).
- **Don't use a synchronous OpenAI client for Ragas collections metrics** — these metrics call
  async generation internally. With Ollama's OpenAI-compatible endpoint, use `AsyncOpenAI`
  for judge LLM/embeddings clients to avoid errors like
  `Cannot use agenerate() with a synchronous client`.
- **Don't bake `chroma_db/` or `data/` into the Docker image** — vectors live in the
  `chroma_data` named volume and `data/` remains bind-mounted from the host.
  The `.dockerignore` excludes these paths.
- **Don't set `OLLAMA_BASE_URL=http://localhost:11434`** inside containers — use
  `http://ollama:11434` so containers resolve the Ollama service by its Compose service name.
- **Don't set `CHROMA_HOST=localhost` inside containers** — use
  `CHROMA_HOST=chroma` so containers resolve the Chroma service by its Compose service name.
- **Don't edit files under `docs/analysis/`** — analysis documents are records, not living docs.
  They are read-only once written. If you need to add information, create a new dated file.
  Only edit an existing analysis file if the human explicitly instructs you to.

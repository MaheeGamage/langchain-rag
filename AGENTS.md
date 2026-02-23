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

### Ingest PDFs into ChromaDB

```bash
poetry run python -m app.ingest
# or
source .venv/bin/activate && python -m app.ingest
```

- Place PDFs in `./data/` before running.
- Logs go to `ingest.log` (auto-created). Tail with `tail -f ingest.log`.
- Embeddings are written to `./chroma_db/` (auto-persisted, no `.persist()` call needed).

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
# Build and start all services (Ollama + API + UI)
docker compose up --build

# Ingest PDFs inside Docker (place files in ./data/ first)
docker compose run --rm api python -m app.ingest

# Full rebuild from scratch (re-downloads Ollama models)
docker compose down -v && docker compose up --build
```

Image layout:
- The `api` service builds and tags the shared image as `langchain-rag`.
- The `ui` service reuses `langchain-rag` via `image: langchain-rag` (no `build:`).
- `chroma_db/` and `data/` are bind-mounted from the host — never baked into the image.
- Ollama model weights are stored in the `ollama_data` named volume.
- `OLLAMA_BASE_URL` is overridden to `http://ollama:11434` inside containers so they
  resolve the Ollama service by name rather than `localhost`.

---

## 3. Project Structure

```
app/
  config.py       — Central config: model names, paths. Change models here.
  ingest.py       — PDF → chunks → embeddings → ChromaDB
  retriever.py    — Wraps ChromaDB as a LangChain retriever
  graph.py        — LangGraph pipeline: retrieve → generate
  api.py          — FastAPI app exposing POST /query
  __init__.py

ui/
  streamlit_app.py  — Streamlit UI (chat interface)

run.py            — Starts both servers locally (no Docker)
Dockerfile        — Two-stage build; produces the shared `langchain-rag` image
docker-compose.yml — Four services: ollama, ollama-init, api, ui
.dockerignore     — Excludes .venv/, chroma_db/, data/, .env, etc.
pyproject.toml    — Dependencies (Poetry / PEP 621 hybrid)

chroma_db/        — Vector database (bind-mounted at runtime; do not edit manually)
data/             — Input PDFs (bind-mounted at runtime; not committed)
ingest.log        — Last ingest run log (not committed)

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
# app/config.py — two independent switches
LLM_PROVIDER       = os.getenv("LLM_PROVIDER",       "ollama")  # ollama | openai | gemini
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")  # ollama | openai | gemini

# Resolved independently
LLM_MODEL,    LLM_API_KEY,    LLM_BASE_URL       = ...  # from _LLM_DEFAULTS[LLM_PROVIDER]
EMBEDDING_MODEL, EMBEDDING_API_KEY, EMBEDDING_BASE_URL = ...  # from _EMBEDDING_DEFAULTS[EMBEDDING_PROVIDER]
COLLECTION_NAME = ...  # derived from EMBEDDING_MODEL — keeps collections separate
```

To switch providers, set `LLM_PROVIDER` and/or `EMBEDDING_PROVIDER` in `.env`
(copy `.env.example`). They are fully independent.  
To override a specific model within a provider, set e.g. `OLLAMA_LLM_MODEL=llama3` in `.env`.

> **Note:** each embedding model gets its own ChromaDB collection (`COLLECTION_NAME`).
> Switching `EMBEDDING_PROVIDER` requires a re-ingest. Switching only `LLM_PROVIDER`
> does **not** require re-ingest.

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

### LangGraph state

The graph state is defined as a `TypedDict` in `graph.py`:

```python
class RAGState(TypedDict):
    question:  str
    documents: List[Document]
    answer:    str
```

- Add new fields here when extending the pipeline (e.g. `rewritten_question`, `citations`).
- Each node receives and returns a **partial state dict** — only return the keys you changed.

### Adding a new graph node

1. Define a function `def my_node(state: RAGState) -> dict:` in `graph.py`.
2. Register it: `builder.add_node("my_node", my_node)`.
3. Wire edges: `builder.add_edge("previous_node", "my_node")`.
4. Return only the state keys the node modifies.

### Retriever

`get_retriever()` in `retriever.py` returns a LangChain `VectorStoreRetriever`.
`search_kwargs={"k": 4}` controls how many chunks are returned. Increase `k`
for more context; decrease for speed.

### Ingest batching

Embeddings are sent to Ollama in batches of `BATCH_SIZE = 25` chunks
(configured at the top of `ingest.py`). Lower this if Ollama OOMs; raise it
to speed up ingest on machines with more VRAM.

### Streamlit UI

`ui/streamlit_app.py` imports `graph` directly from `app.graph` — it does **not**
go through the FastAPI layer. This keeps the UI standalone.

- Chat history is stored in `st.session_state.messages`.
- Each message dict: `{"role": "user"|"assistant", "content": str, "sources": list}`.
- Source documents are shown in a collapsible expander below each answer.

---

## 5. Code Style & Structure Rules

- **One concern per file.** `graph.py` = graph only; `retriever.py` = retriever only, etc.
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

---

## 6. Agent Logging

All sessions **must** be logged. See [`.agent/README.md`](.agent/README.md) for
the full convention. Summary:

1. **Create a session file** at `.agent/sessions/YYYY-MM-DD_session-NNN.md`
   before (or immediately after) starting work.
2. Use the template sections: Goal, Prompts Summary, Actions Taken, Outcome, Agent.
3. Be specific in **Actions Taken** — list every file created/edited and every
   command run.
4. Log the session in the same commit as the code changes it describes.
5. **Never delete** old session logs.
6. **If you update `AGENTS.md`**, record exactly which sections changed and why
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
- **Don't run `python ingest.py` directly** — run as a module: `python -m app.ingest`.
- **Don't hard-code model names or provider-specific classes** outside `config.py` / `factory.py`.
- **Don't add `if PROVIDER == ...` branches outside `factory.py`** — all provider dispatch lives there.
- **Don't add `build:` to the `ui` service** in `docker-compose.yml` — it reuses the
  `langchain-rag` image built by `api`. Adding `build:` to `ui` with the same `image:` name
  causes a conflict ("image already exists" error) because both services would try to tag
  the same name simultaneously.
- **Don't bake `chroma_db/` or `data/` into the Docker image** — they are bind-mounted
  from the host at runtime. The `.dockerignore` excludes them intentionally.
- **Don't set `OLLAMA_BASE_URL=http://localhost:11434`** inside containers — use
  `http://ollama:11434` so containers resolve the Ollama service by its Compose service name.

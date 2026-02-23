# Langchain RAG

A retrieval-augmented generation (RAG) system using LangChain, ChromaDB, and Ollama.

## Setup

1. **Install dependencies:**
   ```bash
   poetry install
   ```

2. **Ensure Ollama is running** (default: `localhost:11434`):
   ```bash
   ollama serve
   ```

3. **Pull required models:**
   ```bash
   ollama pull nomic-embed-text  # embedding model
   ollama pull tinyllama            # LLM
   ```

## Usage

### Ingest PDFs into ChromaDB

```bash
poetry run python -m app.ingest
```

**Output:**
- Progress bar shows chunk embedding status with ETA
- Detailed logs written to `ingest.log` (tail with `tail -f ingest.log`)
- Embeds ~1.6 chunks/sec (varies by hardware)

**Place PDFs in:** `./data/`

### Run the Streamlit UI + FastAPI Server

```bash
python run.py
```

Starts both servers with a single command:

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI | http://localhost:8000 |

Press **Ctrl+C** to stop both. Output from each process is prefixed with `[api]` or `[ui ]`.

To run either server standalone:

```bash
# Streamlit only
source .venv/bin/activate && streamlit run ui/streamlit_app.py

# FastAPI only
source .venv/bin/activate && uvicorn app.api:app --port 8000 --reload
```

### Run with Docker (includes Ollama)

```bash
# Build and start all services (Ollama + API + UI)
docker compose up --build
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| Ollama | http://localhost:11434 |

On first start, `ollama-init` automatically pulls `tinyllama` and `nomic-embed-text`.

Ingest PDFs inside Docker (place files in `./data/` first):
```bash
docker compose run --rm api python -m app.ingest
```

Rebuild from scratch:
```bash
docker compose down -v          # removes containers + volumes (re-downloads models)
docker compose up --build       # rebuilds image and starts all services
```

> **Note:** `chroma_db/` and `data/` are bind-mounted from the host, so ingested
> data persists across container restarts and rebuilds.

## Project Structure

```
app/
  ├── config.py       # Model & path config
  ├── ingest.py       # PDF → chunks → embeddings
  ├── retriever.py    # Semantic search from ChromaDB
  ├── api.py          # FastAPI endpoints
  └── graph.py        # LangGraph RAG pipeline
ui/
  └── streamlit_app.py  # Streamlit chat UI
run.py                # Starts both servers locally (no Docker)
Dockerfile            # Two-stage build; shared image for api + ui services
docker-compose.yml    # Ollama + api + ui services
.dockerignore         # Excludes .venv/, chroma_db/, data/, .env, etc.
chroma_db/            # Vector database (bind-mounted at runtime)
data/                 # Input PDFs (bind-mounted at runtime)
AGENTS.md             # Guidance for AI agents working in this repo
.agent/               # Session logs (development notes)
```

## Dependencies

- **langchain** — LLM framework
- **langchain-chroma** — Vector store
- **langchain-ollama** — Ollama integration
- **chromadb** — Vector database
- **fastapi** / **uvicorn** — API server
- **tqdm** — Progress tracking
- **pypdf** — PDF parsing
- **streamlit** — Chat UI

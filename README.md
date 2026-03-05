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

### ChromaDB Connection

The app connects to ChromaDB over HTTP only, and this repository configures it
through `docker-compose.yml` (`chroma` service at `chroma:8000` inside the network).

For local development (API + ingest on host, Chroma in Docker), use:
```bash
CHROMA_HOST=localhost
CHROMA_PORT=8001
CHROMA_SSL=false
```

When switching Chroma instances, re-ingest your documents:
```bash
poetry run python -m app.ingest
```

### Ingest Content into ChromaDB

```bash
poetry run python -m app.ingest
```

**Output:**
- Progress bars for each stage (parsing, embedding) with ETA
- Detailed logs written to `ingest_pipeline.log` (tail with `tail -f ingest_pipeline.log`)
- Summary of chunks by content type and corpus at the end

**Place content in:** `./data/`

#### Supported file types

| Extension | Parser | Content type |
|-----------|--------|--------------|
| `.mdx`, `.md` | MDX/Markdown parser | `narrative` |
| `.ipynb` | Jupyter Notebook parser | `narrative` (markdown cells) + `code` (code cells) |
| `.pdf` | PDF parser | `narrative` |
| `.py` | Python AST parser | `narrative` (docstrings) + `code` (source bodies) |

Files with extensions `.rst`, `.json`, `.csv` and git artefacts (`.pack`, `.idx`, `.rev`, `.sample`) are intentionally skipped.

#### Corpus detection

Every chunk is automatically tagged with a `source_corpus` field derived from its file path:

| Keyword in path | Corpus tag |
|-----------------|------------|
| `mlflow` | `mlflow` |
| `qiskit` | `qiskit` |
| `qprov` | `qprov` |
| `sample-quantum-circuit` | `sample` |
| *(none of the above)* | `unknown` |

This allows Chroma `where` filters to scope retrieval to a specific knowledge domain without requiring separate collections.

#### Ingestion pipeline stages

```
DATA_ROOT
  │
  ├─ [1/4] Walk    — recursively collect all files
  ├─ [2/4] Parse   — route each file to its parser by extension
  ├─ [3/4] Chunk   — split documents into retrieval-ready chunks
  │           • narrative: MarkdownHeaderTextSplitter → RecursiveCharacterTextSplitter (two-pass)
  │           • code:      RecursiveCharacterTextSplitter (def/class boundaries)
  └─ [4/4] Embed   — add chunks to ChromaDB in batches of BATCH_SIZE
```

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

### Run with Docker (includes Ollama + ChromaDB)

```bash
# Build and start all services (Ollama + ChromaDB + API + UI)
docker compose up --build
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| Ollama | http://localhost:11434 |
| ChromaDB | http://localhost:8001 |

On first start, `ollama-init` automatically pulls `tinyllama` and `nomic-embed-text`.

Ingest content inside Docker (place files in `./data/` first):
```bash
docker compose run --rm api python -m app.ingest
```

Rebuild from scratch:
```bash
docker compose down -v          # removes containers + volumes (re-downloads models)
docker compose up --build       # rebuilds image and starts all services
```

> **Note:** in Docker, vectors persist in the `chroma_data` named volume. `data/`
> is bind-mounted from the host so PDFs stay local.

## Project Structure

```
app/
  ├── config.py          # Model & path config
  ├── vectorstore.py     # Chroma client + LangChain vectorstore builder
  ├── ingest.py          # Main ingestion entry point (orchestrates the pipeline)
  ├── ingest_pipeline/   # Multi-format ingestion pipeline
  │   ├── router.py      # Routes files to parsers by extension; corpus detection
  │   ├── chunker.py     # Narrative (two-pass) and code chunking strategies
  │   └── parsers/
  │       ├── mdx_parser.py      # .mdx / .md — strips JSX, extracts frontmatter
  │       ├── notebook_parser.py # .ipynb — splits markdown and code cells
  │       ├── pdf_parser.py      # .pdf
  │       └── python_parser.py   # .py — AST-based docstring + source extraction
  ├── retriever.py       # Semantic search from ChromaDB
  ├── api.py             # FastAPI endpoints
  └── graph.py           # LangGraph RAG pipeline
ui/
  └── streamlit_app.py   # Streamlit chat UI
run.py                   # Starts both servers locally (no Docker)
Dockerfile               # Two-stage build; shared image for api + ui services
docker-compose.yml       # Ollama + ChromaDB + api + ui services
.dockerignore            # Excludes .venv/, chroma_db/, data/, .env, etc.
data/                    # Input content (bind-mounted at runtime)
AGENTS.md                # Guidance for AI agents working in this repo
.agent/                  # Session logs (development notes)
```

## Dependencies

- **langchain** — LLM framework
- **langchain-chroma** — Vector store
- **langchain-ollama** — Ollama integration
- **chromadb** — Vector database
- **fastapi** / **uvicorn** — API server
- **tqdm** — Progress tracking
- **pypdf** — PDF parsing
- **pyyaml** — YAML frontmatter parsing (MDX/MD/notebook)
- **streamlit** — Chat UI

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
source .venv/bin/activate && streamlit run streamlit_app.py

# FastAPI only
source .venv/bin/activate && uvicorn app.api:app --port 8000 --reload
```

## Project Structure

```
app/
  ├── config.py       # Model & path config
  ├── ingest.py       # PDF → chunks → embeddings
  ├── retriever.py    # Semantic search from ChromaDB
  ├── api.py          # FastAPI endpoints
  └── graph.py        # LangGraph RAG pipeline
streamlit_app.py      # Streamlit chat UI
chroma_db/            # Vector database
data/                 # Input PDFs
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

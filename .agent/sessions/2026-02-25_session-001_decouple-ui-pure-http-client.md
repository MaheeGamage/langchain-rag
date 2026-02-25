# Session: 2026-02-25 #001

## Summary
Decoupled Streamlit UI from backend — UI is now a pure HTTP client calling the FastAPI API.

## Goal
Fix empty retrieval results in Docker and refactor the UI so it no longer
imports `app.graph` or touches ChromaDB directly. All RAG work should go
through the FastAPI API.

## Prompts Summary
- "When I run with `poetry run python run.py` the retriever works fine, but dockerised it returns empty array. Why?"
- "Fix the issue with ui and api. I want ui to act as ui only without doing any agent tasks."
- "Add a summary slug to session filenames so the purpose is visible in a directory listing."

## Actions Taken
- Diagnosed root cause: `ui` container lacked `./chroma_db` bind-mount, so its
  in-process retriever read an empty database.
- Rewrote `ui/streamlit_app.py` — removed `from app.graph import graph` and
  `from app.config import ...`; UI now calls `GET /config` and `POST /query`
  via `requests`. `API_URL` env var controls the backend address.
- Rewrote `app/api.py` — added Pydantic models (`QueryRequest`, `QueryResponse`,
  `SourceChunk`), `GET /config` endpoint (returns model info), and updated
  `POST /query` to accept a JSON body and return `{answer, sources[]}`.
- Updated `docker-compose.yml` — removed `chroma_db` volume and `OLLAMA_BASE_URL`
  from `ui` service; added `API_URL=http://api:8000`; changed `ui.depends_on`
  from `ollama` to `api`.
- Updated `AGENTS.md` — Streamlit UI section rewritten to reflect pure-HTTP
  architecture; added "Don't import app.graph in streamlit_app.py" rule;
  updated session filename convention to `YYYY-MM-DD_session-NNN_short-summary.md`;
  added Summary field requirement to logging rules.
- Updated `.agent/README.md` — added Summary section to session template;
  updated filename format and directory structure to include kebab-case slug.
- Rebuilt Docker image and verified end-to-end: `GET /config` returns model
  info; `POST /query` from UI container returns answer + 4 source chunks.

## Outcome
UI container no longer needs ChromaDB, Ollama access, or any ML dependencies
at runtime. All RAG logic flows through the FastAPI API. Empty-retrieval bug
is resolved. Session filename convention updated for better scanability.

## Agent
GitHub Copilot (Claude Opus 4.6)

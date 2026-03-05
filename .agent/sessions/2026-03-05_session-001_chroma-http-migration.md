# Session: 2026-03-05 #001

## Goal
Implement and iterate on Chroma storage migration in one conversation:
1) add dual-mode local/http support,
2) remove local embedded support (Docker-hosted HTTP Chroma only),
3) restore development-friendly host defaults/env values for local API/ingest
   with Ollama+Chroma running in Docker.

## Prompts Summary
- User requested implementation of the full migration plan for local+remote Chroma support.
- User then requested removing local Chroma support and related config, and setting Chroma only in Docker.
- User clarified development workflow: Ollama+Chroma in Docker while API+ingest run locally, and asked to keep Chroma env values in `.env.example`.

## Actions Taken
- Reviewed current implementation and docs with:
  - `rg -n "Chroma|chroma|persist_directory|HttpClient|chromadb" app ui run.py pyproject.toml docker-compose.yml`
  - `sed -n ...` on `app/config.py`, `app/ingest.py`, `app/retriever.py`, `app/factory.py`,
    `app/api.py`, `docker-compose.yml`, `.env.example`, `README.md`, `AGENTS.md`
  - `ls -1 .agent/sessions | tail -n 20`
- Implemented dual-mode in `app/config.py`:
  - Added `CHROMA_MODE`, `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_SSL`.
  - Added validation/parsing helpers for int/bool env values.
  - Added derived `CHROMA_TARGET` for observability.
- Added `app/vectorstore.py`:
  - Centralized vectorstore creation using `chromadb.PersistentClient` (local) or
    `chromadb.HttpClient` (http).
- Edited `app/ingest.py`:
  - Replaced direct `Chroma(...persist_directory=...)` construction with `get_vectorstore()`.
  - Updated startup log/print output to include Chroma mode and target.
- Edited `app/retriever.py`:
  - Replaced direct local Chroma construction with `get_vectorstore()`.
- Edited `app/api.py`:
  - Extended `GET /config` to return `chroma_mode` and `chroma_target`.
- Edited `.env.example`:
  - Added documented Chroma mode/server env vars.
- Edited `docker-compose.yml`:
  - Added `chroma` service (`chromadb/chroma`), published on host `8001`.
  - Added `chroma_data` named volume for persistence.
  - Wired `api` env to remote Chroma (`CHROMA_MODE=http`, `CHROMA_HOST=chroma`, `CHROMA_PORT=8000`, `CHROMA_SSL=false`).
  - Added `api` dependency on `chroma`.
  - Removed `./chroma_db:/app/chroma_db` mount from `api`.
- Edited `README.md`:
  - Added “Local vs Remote Chroma” section with exact env examples.
  - Updated Docker section and service table to include ChromaDB.
  - Updated persistence note for Docker mode.
- Edited `AGENTS.md`:
  - Updated ingest and Docker guidance for dual Chroma modes.
  - Updated project structure (`app/vectorstore.py`, compose services count).
  - Updated configuration pattern section to include `CHROMA_MODE` and `CHROMA_TARGET`.
  - Added guardrail to centralize Chroma mode branching in `app/vectorstore.py`.
  - Updated “What Not to Do” for Docker Chroma host and storage behavior.
- Per follow-up request, removed local embedded support:
  - `app/config.py`: removed `CHROMA_MODE` and `CHROMA_PATH`; set HTTP-only Chroma connection.
  - `app/vectorstore.py`: removed `PersistentClient` branch; always uses `chromadb.HttpClient`.
  - `app/ingest.py`: removed `CHROMA_MODE` logging.
  - `app/api.py`: removed `chroma_mode` from `/config`.
  - `docker-compose.yml`: removed `CHROMA_MODE=http` environment variable.
  - `.env.example`: temporarily removed Chroma env values.
  - `README.md`/`AGENTS.md`: updated wording to reflect HTTP-only Chroma.
- Per final clarification for development flow, restored dev env visibility/defaults:
  - `app/config.py`: defaulted to host-accessible values (`CHROMA_HOST=localhost`, `CHROMA_PORT=8001`).
  - `.env.example`: re-added `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_SSL` with dev comments.
  - `README.md`: documented local-dev setup (host-run API/ingest + Docker Chroma).
  - `AGENTS.md`: documented host-run development defaults for Chroma host/port.
- Validation:
  - Attempted `source .venv/bin/activate && python -m compileall app` (failed due stale venv activation path in script).
  - Ran `.venv/bin/python -m compileall app` successfully multiple times after updates.

## Outcome
Conversation-final state:
- Chroma is HTTP-only (no embedded local Chroma code path).
- Docker Compose runs Chroma service and API container points to `chroma:8000`.
- Local development remains supported with env values/defaults targeting `localhost:8001`
  when API/ingest run on host and Chroma runs in Docker.
- Docs and agent guidance were updated accordingly.

## Agent
Codex (GPT-5)

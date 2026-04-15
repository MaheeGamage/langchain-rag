# Session: 2026-04-15 #001

## Goal
Add switchable RAG graph implementations so multiple graph styles (baseline and agentic) can be tested without losing existing behavior, while keeping a single graph export entrypoint.

## Prompts Summary
- User requested adding an agentic RAG implementation based on LangChain docs.
- User requested a single switch mechanism in one place.
- User requested separate files/folder for graph implementations, while exposing one export file.

## Actions Taken
- Added `RAG_GRAPH_IMPLEMENTATION` constant and validation in `app/config.py`.
- Created `app/graphs/` package with:
  - `app/graphs/types.py` for shared graph state type.
  - `app/graphs/common.py` for shared prompt, retrieval retry, context/source formatting, and token utilities.
  - `app/graphs/baseline.py` containing the original retrieve -> generate graph implementation.
  - `app/graphs/agentic.py` containing an iterative agentic graph (`plan -> retrieve -> assess -> ... -> generate`).
- Refactored `app/graph.py` into a selector/export module only:
  - Chooses implementation using `RAG_GRAPH_IMPLEMENTATION`.
  - Compiles and exports the selected `graph`.
  - Exposes `stream_query(...)` for streaming endpoint compatibility.
- Updated `app/api.py` to use `stream_query(...)` and report active graph implementation in `GET /config`.
- Updated `.env.example` with `RAG_GRAPH_IMPLEMENTATION=baseline`.
- Updated `README.md` project structure and graph switch docs.
- Updated `AGENTS.md` sections:
  - Project structure
  - Configuration snippet
  - Graph implementation switch pattern
  - LangGraph state location
  - Adding node guidance
  - Code style rules
  - What-not-to-do graph dispatch rule
- Ran syntax validation via `python -m compileall app`.

## Outcome
Implemented a switchable graph architecture with separate implementation files and a single export surface. Existing imports can continue to use `app.graph` while graph behavior is selected centrally via config.

## Agent
GitHub Copilot (GPT-5.3-Codex)

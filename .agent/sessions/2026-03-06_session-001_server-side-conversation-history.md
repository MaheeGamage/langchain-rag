# Session: 2026-03-06 #001

## Goal
Move conversation history storage from the client (Streamlit) to the server,
using a LangGraph SQLite checkpointer keyed by a `thread_id`. Improve the
`.gitignore` to cover new runtime files.

## Prompts Summary
- User asked whether the graph sends conversation history to the LLM.
- User asked whether history should be sent, or if there is a better LangChain pattern.
- User applied a partial change (`sections.extend(state["messages"])`) that broke with `TypeError` because `str.join` can't mix strings and `BaseMessage` objects.
- User requested the fix — refactored to pass a `list[BaseMessage]` directly to the LLM using `SystemMessage` for the base prompt + RAG context.
- Error: Gemma models don't support `SystemMessage` (`INVALID_ARGUMENT: Developer instruction is not enabled`).
- User added a `SqliteSaver` checkpointer inside a `with` block (leaking closed connection) and asked to wire it up properly.
- User asked to move the DB path to `config.py`.
- User asked to save the DB in a subdirectory rather than the repo root.
- User asked to send only a `thread_id` from the client instead of the full history.
- User asked the API to generate the `thread_id` (reverted — LangGraph requires caller to supply it; client now owns ID generation).
- User asked to remove `uuid` generation from the API and make `thread_id` optional in the response.
- `ModuleNotFoundError: No module named 'langgraph.checkpoint.sqlite'` — package was not installed.
- User asked to use Poetry to install it.
- `FileExistsError` on `os.makedirs` — stale SQLite files (`conversations`, `conversations-shm`, `conversations-wal`) existed at root from a prior failed run.
- User asked to improve `.gitignore` and specifically to ignore local SQLite files.

## Actions Taken
- **`app/graph.py`**:
  - Added `SystemMessage` import.
  - Replaced `build_generate_prompt` (returned a string) with `build_messages` (returns `list[BaseMessage]`).
  - System prompt + RAG context → `SystemMessage`; for Gemma models → `HumanMessage`/`AIMessage("Understood.")` workaround.
  - `generate()` now calls `llm.invoke(messages)` with the message list directly.
  - Fixed `SqliteSaver` lifetime: replaced `with SqliteSaver.from_conn_string(...)` block with a module-level `_db_conn` / `_checkpointer` (connection kept alive for process lifetime).
  - Added `check_same_thread=False` on the SQLite connection for async safety.
  - Added `import os` and `os.makedirs(..., exist_ok=True)` to auto-create the DB directory.
  - Imported `CONVERSATIONS_DB` from `config.py` instead of hardcoding the path.
- **`app/config.py`**:
  - Added `CONVERSATIONS_DB: str = os.getenv("CONVERSATIONS_DB", "./conversations/conversations.db")` under the Paths section.
- **`app/api.py`**:
  - Added `thread_id` handling: uses `req.conversation.id` if provided, else `None` (stateless).
  - Passes `config={"configurable": {"thread_id": thread_id}}` to `graph.invoke` when a thread ID is present.
  - Returns `thread_id` in `QueryResponse`.
  - Removed `import uuid` (no longer generates IDs server-side).
- **`app/schemas.py`**:
  - Added `thread_id: str | None` field to `QueryResponse`.
- **`ui/streamlit_app.py`**:
  - Added `import uuid`.
  - Introduced `st.session_state.thread_id` (starts `None`; populated from first API response).
  - `_query_api` now accepts `thread_id` instead of `history`; sends only `{"id": thread_id}` in the conversation payload.
  - "Clear chat" resets `thread_id` to `None` to start a fresh server-side thread.
  - Stores returned `thread_id` from each response into session state.
- **`pyproject.toml`**:
  - Added `langgraph-checkpoint-sqlite` via `poetry add` (version `^3.0.3`; also pulled in `aiosqlite` and `sqlite-vec`).
- **`.gitignore`**:
  - Rewrote with organised sections covering: environment/secrets, Python bytecode, virtual envs, logs, data/vector stores, SQLite files (`*.db`, `*.db-shm`, `*.db-wal`), build artifacts, test/coverage, IDEs, OS metadata.
- **Filesystem cleanup**:
  - Created `conversations/` directory.
  - Moved `conversations.db`, `conversations.db-shm`, `conversations.db-wal` into `conversations/`.
  - Removed stale `conversations-shm` and `conversations-wal` files from repo root.

## Outcome
Server-side conversation persistence working via LangGraph SQLite checkpointer.
Client sends a `thread_id` (UUID) on every request; server rehydrates history automatically.
System prompt is rebuilt fresh each turn and never stored in checkpointed state.
`.gitignore` covers all new runtime files.

## Agent
GitHub Copilot (Claude Sonnet 4.6)

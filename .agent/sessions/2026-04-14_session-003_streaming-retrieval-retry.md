# Session: 2026-04-14 #003

## Goal
Harden the new streaming query endpoint so transient embedding/retriever transport errors do not crash the request with HTTP 500 and instead return a structured stream error.

## Prompts Summary
- User asked why `httpx.ReadError: [Errno 104] Connection reset by peer` occurred in `query_stream`.
- User approved implementing the suggested fix (retry/backoff and graceful streaming error handling).

## Actions Taken
- Inspected `app/api.py` and traceback location around `query_stream` retrieval call.
- Added shared `_invoke_retriever_with_retry(...)` in `app/graph.py` with short exponential backoff and warning logs.
- Updated `retrieve(...)` in `app/graph.py` to use the shared retry helper.
- Moved retrieval execution inside the stream generator in `query_stream` so failures are emitted as stream events.
- Added graceful NDJSON fallback on retrieval failure:
  - emits `metadata` event with empty sources/thread id
  - emits `error` event with retrieval failure message
  - returns without raising pre-stream exception
- Removed duplicate API-level retry helper/imports and now rely on graph-level retry for both `/query` and `/query/stream`.
- Ran `python -m py_compile app/graph.py app/api.py ui/streamlit_app.py` to validate syntax.
- Added runtime mode toggle in `ui/streamlit_app.py` sidebar (`Stream responses`).
- Updated chat request path in `ui/streamlit_app.py` to switch between `_stream_query(...)` and `_query_api(...)` based on `st.session_state.stream_mode`.
- Re-ran `python -m py_compile ui/streamlit_app.py` after toggle wiring.
- Added a streaming-only waiting indicator in `ui/streamlit_app.py` that displays until the first token arrives, then clears automatically.
- Re-ran `python -m py_compile ui/streamlit_app.py` after the first-token indicator change.
- Replaced the waiting text with an animated loading icon (CSS spinner) in `ui/streamlit_app.py`.
- Fixed an indentation regression introduced during spinner insertion and re-validated with `python -m py_compile ui/streamlit_app.py`.

## Outcome
Both `/query` and `/query/stream` now share retrieval retry behavior via `app/graph.py`. The streaming endpoint also avoids pre-stream 500s by returning structured stream events on retrieval failure.
The UI can now switch live between streaming and non-streaming responses without code edits.
In streaming mode, users now see an explicit waiting indicator during retrieval/initial generation latency before the first token appears.

## Agent
GitHub Copilot (GPT-5.3-Codex)

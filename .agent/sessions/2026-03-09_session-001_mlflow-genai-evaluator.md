# Session: MLflow GenAI Evaluator

**Date:** 2026-03-09
**Agent:** GitHub Copilot (Claude Sonnet 4.6)

---

## Goal

Implement a MLflow-based evaluation harness for the RAG pipeline, following the
official MLflow GenAI evaluation documentation at
https://mlflow.org/docs/latest/genai/eval-monitor/

---

## Prompts Summary

1. "Now i need you to enable mlflow based evaluator. Can you implement that based on mlflow documentation"
2. "you can use poetry to check and install dependencies"
3. "This have some guideline if you needed — (guidelines judge docs URL)"

---

## Actions Taken

### Dependency upgrade

- Detected MLflow 2.22.4 was installed (pinned `<3.0.0` in `pyproject.toml`).
  `mlflow.genai` (the new evaluation API) is only available in MLflow 3.x.
- Installed MLflow 3.10.1 directly into `.venv` via `.venv/bin/python3 -m pip install "mlflow>=3.0.0,<4.0.0"`.
- Updated `pyproject.toml`: `mlflow (>=2.19.0,<3.0.0)` → `mlflow (>=3.0.0,<4.0.0)`.

### New file: `app/evaluator.py`

Core evaluation module. Key design points:
- `_retrieve(query)` — decorated `@mlflow.trace(span_type="RETRIEVER")`. Returns
  `list[mlflow.entities.Document]`. The RETRIEVER span is required by RAG-specific
  judges (`RetrievalGroundedness`, `RetrievalSufficiency`).
- `rag_predict(question)` — decorated `@mlflow.trace`. Full RAG pipeline
  (retrieve → generate), mirroring production graph.py but without LangGraph
  conversation history. Returns `{"response": answer}`.
- `build_scorers(names, judge_model)` — instantiates scorer objects from name strings.
- `run_evaluation(dataset, ...)` — calls `mlflow.genai.evaluate()` and returns
  a JSON-serialisable result dict.
- 7 built-in scorer names: `relevance`, `correctness`, `fluency`, `groundedness`,
  `sufficiency`, `conciseness`, `domain_tone`.
- Default scorers (no judge model key required): `relevance`, `fluency`, `groundedness`.
- Lazy-initialised singleton retriever and LLM chain to avoid unnecessary
  connections on import.

### Modified: `app/config.py`

- Added `MLFLOW_JUDGE_MODEL: str = os.getenv("MLFLOW_JUDGE_MODEL", "")` — allows
  global judge model override via env var (LiteLLM format, e.g. `openai:/gpt-4o-mini`).
- Added `RAG_SYSTEM_PROMPT: str = ...` — extracted the hardcoded system prompt from
  `graph.py` into a shared constant so evaluation uses the exact same prompt as
  production.

### Modified: `app/graph.py`

- Imported `RAG_SYSTEM_PROMPT` from `config`.
- Replaced the hardcoded `BASE_PROMPT` local variable in `build_messages()` with the
  shared constant.

### Modified: `app/schemas.py`

- Added `from typing import Any`.
- Added `EvalSample` — one evaluation sample (`question`, optional
  `expected_response`, optional `expected_facts`).
- Added `EvalRequest` — request body for `POST /evaluate` (`dataset`, `scorers`,
  `judge_model`, `experiment_name`).
- Added `EvalResponse` — response body (`run_id`, `experiment_name`, `metrics`,
  `results`, `available_scorers`).

### Modified: `app/api.py`

- Added `EvalRequest`, `EvalResponse` to imports from `app.schemas`.
- Added `HTTPException` to FastAPI imports.
- Added `asyncio` and `ThreadPoolExecutor` imports.
- Added `"eval"` tag to `_TAGS` list.
- Added `_eval_executor = ThreadPoolExecutor(max_workers=1)` — evaluation runs are
  CPU/IO-bound blocking operations; they are offloaded to a thread to avoid blocking
  the async event loop.
- Added `POST /evaluate` endpoint:
  - Returns HTTP 503 if `MLFLOW_ENABLED` is not `true`.
  - Converts `EvalRequest.dataset` to the `{"inputs": ..., "expectations": ...}` format
    that `mlflow.genai.evaluate()` expects.
  - Calls `run_evaluation()` in the thread executor.
  - Raises HTTP 503 for `RuntimeError` (MLflow disabled), 422 for `ValueError`
    (bad scorer names).

### Modified: `.env.example`

- Added `MLFLOW_JUDGE_MODEL=` entry with a comment explaining the LiteLLM format.

---

## Outcome

- MLflow 3.10.1 installed and available.
- `POST /evaluate` endpoint added. Requires `MLFLOW_ENABLED=true`.
- Supports 7 scorer types including 3 RAG-specific judges.
- All modified files pass syntax check and module import smoke test.
- The production system prompt is now shared between graph.py and evaluator.py via
  `config.RAG_SYSTEM_PROMPT`, ensuring evaluation matches production behaviour.

---

## Sections of AGENTS.md Updated

- None updated — the new patterns (evaluation endpoint, evaluator module, judge model
  config) should be documented in AGENTS.md § 3 (structure), § 4 (patterns), and § 7
  (what not to do) in a follow-up.

# Session: 2026-03-19 #003

## Goal
Fix runtime failures in evaluation/ragas/evals_2.py so it runs against local Ollama at port 11435 without deprecation warnings, API misuse errors, or result-export crashes.

## Prompts Summary
- User reported deprecation warnings for Ragas metric imports, multiple Instructor retry/validation failures, and an AttributeError on EvaluationResult.save().
- User confirmed Ollama endpoint is http://localhost:11435/.

## Actions Taken
- Updated evaluation/ragas/evals_2.py imports to use ragas.metrics.collections for collections metrics.
- Added robust Ollama base URL normalization and script-level override support via EVAL_OLLAMA_BASE_URL.
- Added model discovery/probing and judge-model fallback logic to avoid tinyllama for strict schema metrics when stronger local models are available.
- Switched judge/embedding clients for collections metrics to AsyncOpenAI and kept a synchronous OpenAI probe client for /v1/models listing.
- Replaced incompatible ragas.evaluate(...) usage with direct collections metric scoring using metric.batch_score(...), including signature-based input mapping and per-metric error capture.
- Replaced obsolete EvaluationResult.save() flow with pandas DataFrame CSV export to evaluation/ragas/experiments/evals_2_<timestamp>.csv.
- Converted main entrypoint to synchronous execution so sync batch_score calls run outside async context.
- Verified script execution end-to-end with numeric metric outputs.
- Updated AGENTS.md "What Not to Do" with two Ragas 0.4 pitfalls:
  - collections metrics are not valid inputs to ragas.evaluate()
  - collections metrics with OpenAI-compatible endpoints should use AsyncOpenAI clients

## Outcome
evaluation/ragas/evals_2.py now runs successfully on Ollama at 11435, computes non-NaN metric scores, and saves CSV output without AttributeError. Deprecation warnings from legacy metric imports are removed.

## Agent
GitHub Copilot (GPT-5.3-Codex)

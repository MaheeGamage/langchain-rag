# Session: 2026-04-14 #001

## Goal
Fix a Chroma notebook query error where query embeddings had the wrong dimension for the target collection.

## Prompts Summary
- User reported `InvalidArgumentError: Collection expecting embedding with dimension of 768, got 384` while running keyword query code in a notebook.

## Actions Taken
- Inspected notebook cells in `chromadb/chroma_keyword_search.ipynb` and confirmed `client.get_collection(...)` was called without an embedding function.
- Edited the setup cell to:
  - import embedding provider/model config from `app.config`
  - initialize LangChain embeddings via `app.factory.get_embeddings()`
  - define a Chroma-compatible embedding function class implementing `__call__(self, input)`
  - pass that embedding function to `client.get_collection(...)`
- Re-ran the setup cell and the previously failing keyword query cell to verify the fix.

## Outcome
Notebook query now succeeds with the configured embedding backend (`ollama/nomic-embed-text`), and the dimension mismatch error is resolved.

## Agent
GitHub Copilot (GPT-5.3-Codex)

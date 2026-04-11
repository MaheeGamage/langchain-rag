# Session: 2026-04-09 — Testset Generation

## Goal
Implement a Ragas single-hop synthetic testset generation script for the RAG system,
following the Ragas tutorial at https://docs.ragas.io/en/latest/howtos/applications/singlehop_testset_gen/

## Files changed
- `experimentation/testset-generation/generate_testset.py` — new script
- `experimentation/testset-generation/__init__.py` — new (makes folder a package)

## What was done
- Loads all `.md`, `.mdx`, `.rst`, `.txt` files from `knowledge_ingestion/content/v1` recursively
- Builds a Ragas `KnowledgeGraph` from the documents
- Applies `HeadlinesExtractor`, `HeadlineSplitter`, `KeyphrasesExtractor` transforms
- Uses `SingleHopSpecificQuerySynthesizer` with a 50/50 split between headlines and keyphrases
- Three domain-specific personas: Quantum Software Researcher, MLflow Practitioner, Quantum Computing Student
- Reuses `get_ragas_judge_llm()` / `get_ragas_judge_embeddings()` from `evaluation/ragas/ragas_factory.py`
- Output saved as JSON matching the `evaluation/eval_dataset.json` shape (`inputs.question` / `expectations.expected_response`)
- `TESTSET_SIZE` and `OUTPUT_PATH` are configurable via env vars

## Run command
```bash
source .venv/bin/activate
python -m experimentation.testset-generation.generate_testset

# With overrides
TESTSET_SIZE=20 OUTPUT_PATH=evaluation/eval_dataset.json python -m experimentation.testset-generation.generate_testset
```

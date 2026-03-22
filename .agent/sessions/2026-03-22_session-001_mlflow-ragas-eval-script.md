# Session: Convert MLflow RAGAS Evaluation Notebook to Script

**Date:** 2026-03-22  
**Session ID:** 001  
**Agent:** Kiro  
**Goal:** Convert notebook-style code from reference implementation to a Python script for RAG evaluation using RAGAS and MLflow

## Context

User provided a reference notebook implementation at `evaluation/experimentation/2026-01-08-ragas-in-mlfow-rag-eval-demo-main/` and requested conversion to a Python script in `evaluation/experimentation/mlflow/eval.py` that integrates with the existing RAG system.

## Changes Made

### 1. Created Main Evaluation Script
**File:** `evaluation/experimentation/mlflow/eval.py`

Converted Jupyter notebook cells to a structured Python script with:
- Provider configuration using existing app config (`LLM_PROVIDER`, `EMBEDDING_PROVIDER`)
- Environment validation for different LLM providers (OpenAI, Azure, Ollama, Gemini)
- MLflow model URI generation for litellm compatibility
- Knowledge base loading (placeholder for integration with actual data)
- RAG chain creation using LangChain LCEL
- Traced RAG prediction function with RETRIEVER and LLM spans for RAGAS metrics
- Evaluation dataset loading and preparation
- RAGAS scorer configuration (Faithfulness, FactualCorrectness, ContextPrecision, ContextRecall)
- Evaluation execution with MLflow tracking
- Results analysis and display

Key adaptations:
- Imports from existing `app.config` and `app.factory` modules
- Provider enum mapping from app config strings
- Placeholder for actual data loading (marked with TODO comments)
- Modular functions for reusability
- Main function for script execution

### 2. Created Example Evaluation Dataset
**File:** `evaluation/experimentation/mlflow/eval_dataset.json`

Minimal example dataset with 3 questions for testing:
- Questions about RAG, pipeline workflow, and ChromaDB
- Ground truth answers
- Expected contexts

### 3. Created Documentation
**File:** `evaluation/experimentation/mlflow/README.md`

Comprehensive guide covering:
- Overview of RAGAS metrics
- Setup instructions
- Usage examples
- Customization options (using actual data, changing judge model, comparing configurations)
- Score interpretation guidelines
- Troubleshooting common issues
- Reference to original notebook

## Technical Notes

### RAGAS Integration Requirements

1. **Traced Functions**: RAGAS scorers require MLflow traces with specific span types:
   - `RETRIEVER` spans for Context Precision/Recall
   - `LLM` spans for generation tracking
   - `CHAIN` span for overall pipeline

2. **Function Signature**: `traced_rag_predict(question: str)` parameter name must match `inputs['question']` key in evaluation data

3. **Judge Model**: RAGAS uses LLM-as-judge via litellm:
   - OpenAI recommended for reliable scoring
   - Ollama may have issues with structured output parsing
   - Model URI format: `provider:/model-name` (e.g., `openai:/gpt-4o-mini`)

4. **Environment Variables**: litellm requires provider-specific env vars:
   - Ollama: `OLLAMA_API_BASE`
   - Azure: `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION`

### Integration Points

The script integrates with existing codebase:
- Uses `app.config` for provider/model configuration
- Uses `app.factory.get_llm()` and `get_embeddings()`
- Follows existing project structure conventions
- Marked TODOs for actual data integration:
  - `load_knowledge_base()` - replace with actual document loading
  - `load_evaluation_dataset()` - create proper evaluation dataset

### Next Steps for User

1. **Create Evaluation Dataset**: Build a comprehensive `eval_dataset.json` with questions relevant to their knowledge base
2. **Integrate Data Loading**: Replace placeholder `load_knowledge_base()` with actual data source (ChromaDB or data/ directory)
3. **Run Evaluation**: Execute script and verify metrics
4. **Iterate**: Use results to improve RAG pipeline (chunk size, retrieval k, prompts)

## Files Modified

- Created: `evaluation/experimentation/mlflow/eval.py` (main script)
- Created: `evaluation/experimentation/mlflow/eval_dataset.json` (example data)
- Created: `evaluation/experimentation/mlflow/README.md` (documentation)

## Testing

Script structure is complete but requires:
1. Actual evaluation dataset creation
2. Data loading integration
3. Test run with MLflow UI verification

## References

- Source notebook: `evaluation/experimentation/2026-01-08-ragas-in-mlfow-rag-eval-demo-main/2026_01_08_ragas_in_mlfow_rag_eval_demo.ipynb`
- RAGAS documentation: https://github.com/explodinggradients/ragas
- MLflow GenAI evaluation: https://mlflow.org/docs/latest/genai/eval-monitor/scorers/third-party/ragas/

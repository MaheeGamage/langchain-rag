# Ragas RAG Evaluation

This script evaluates your RAG agent using the Ragas framework with Ollama models.

## Prerequisites

1. **Ollama running** with the `mistral` model:
   ```bash
   ollama list  # Check if mistral is available
   ollama pull mistral  # If not, pull it
   ```

2. **ChromaDB running** (your RAG agent needs it):
   ```bash
   # Check if Chroma is running on port 8001
   curl http://localhost:8001/api/v1/heartbeat
   ```

3. **Environment configured** - Make sure your `.env` file is set up with:
   ```
   LLM_PROVIDER=ollama
   EMBEDDING_PROVIDER=ollama
   CHROMA_HOST=localhost
   CHROMA_PORT=8001
   ```

4. **Data ingested** - Your ChromaDB should have documents:
   ```bash
   # From project root
   poetry run python -m app.ingest
   ```

## How to Run

From the **project root directory** (not from this folder):

```bash
# Activate your virtual environment
source .venv/bin/activate

# Run the evaluation script
python evaluation/experimentation/ragas_eval/evals.py
```

Or with Poetry:

```bash
poetry run python evaluation/experimentation/ragas_eval/evals.py
```

## Architecture

- **Your RAG Agent**: Uses Ollama (configured via your `.env`)
- **Judge LLM**: Uses Ollama's mistral model via OpenAI-compatible API
- **Note**: The OpenAI client is only used to connect to Ollama's API endpoint - no actual OpenAI API calls are made

## What It Does

1. **Loads a test dataset** with 3 sample questions about "ragas 0.3"
2. **Runs your RAG agent** (from `app/graph.py`) on each question
3. **Scores responses** using Ollama's mistral model as a judge
4. **Saves results** to CSV in `evaluation/experimentation/ragas_eval/experiments/`

## Output

- Dataset saved to: `evaluation/experimentation/ragas_eval/evals/test_dataset.csv`
- Results saved to: `evaluation/experimentation/ragas_eval/experiments/<experiment_name>.csv`

## Customizing the Dataset

Edit the `data_samples` list in `load_dataset()` to add your own questions and grading notes:

```python
data_samples = [
    {
        "question": "Your question here",
        "grading_notes": "Expected answer points here",
    },
]
```

## Troubleshooting

- **Import errors**: Make sure you run from the project root, not from this directory
- **Ollama connection errors**: Check Ollama is running on port 11434
- **ChromaDB errors**: Ensure ChromaDB is running and has ingested documents
- **Empty responses**: Verify your RAG agent works by testing it directly first

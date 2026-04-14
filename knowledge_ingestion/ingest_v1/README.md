# Required knowledge areas

1. MLflow documentation - https://github.com/mlflow/mlflow/releases/tag/v2.21.1
2. QProv specification
    1. Original Paper - https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/qtc2.12012
    2. Extracted provenance data model
3. Quantum software engineering
    1. Qiskit basics
4. Quantum experiment tracking context
    1. Sample quantum programs with experiment tracking 

# How to use ingest.py

This script ingests content files, chunks them, and writes embeddings to the
configured Chroma collection.

### Prerequisites

1. Install dependencies:

```bash
poetry install
```

2. Ensure your `.env` is configured (embedding provider/model and Chroma
settings are read from `app/config.py`).

3. Place source content under the configured data root (default comes from
`DATA_ROOT` in config/environment).

### Run ingestion

From the repository root:

```bash
poetry run python -m knowledge_ingestion.ingest
```

### Use a custom data root

```bash
DATA_ROOT=./knowledge_ingestion/content/refined-content poetry run python knowledge_ingestion/ingest.py
```

### What the script does

1. Walks all files under `DATA_ROOT`.
2. Routes each file to the correct parser by extension.
3. Chunks parsed documents for retrieval.
4. Embeds and stores chunks in Chroma in batches.

### Output and logs

- Console output shows stage-by-stage progress (discover, parse, chunk, embed).
- Detailed logs are written to `ingest_pipeline.log`.
- Duplicate chunks are skipped via deterministic content-based IDs.source .venv/bin/activate && python -c "
import chromadb
from app.config import CHROMA_HOST, CHROMA_PORT

client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
cols = client.list_collections()
print(f'Deleting {len(cols)} collection(s):', [c.name for c in cols])
for col in cols:
    client.delete_collection(col.name)
print('Done.')
"

# Remove all data in chromaDB

```
source .venv/bin/activate && python -c "
import chromadb
from app.config import CHROMA_HOST, CHROMA_PORT

client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
cols = client.list_collections()
print(f'Deleting {len(cols)} collection(s):', [c.name for c in cols])
for col in cols:
    client.delete_collection(col.name)
print('Done.')
"
```
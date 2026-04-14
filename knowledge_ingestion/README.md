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
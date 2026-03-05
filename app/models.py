# app/models.py
#
# Shared data models used by both the API layer (api.py) and the graph
# pipeline (graph.py).  Keep this module free of heavy imports so it can be
# imported from anywhere without triggering circular dependencies.

from pydantic import BaseModel


class ContextEntry(BaseModel):
    """A single typed context item carried through the RAG pipeline.

    type: "file" | "selection" | "document" | "snippet" | "url" | "image" | ...
    """

    type: str
    name: str | None = None
    content: str | None = None
    mimeType: str | None = None
    range: dict | None = None   # {"start": int, "end": int} for selection entries
    score: float | None = None  # relevance score (snippet / retriever entries)
    source: str | None = None   # e.g. "retriever"

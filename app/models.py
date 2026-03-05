# app/models.py
#
# Shared data models used by both the API layer (api.py) and the graph
# pipeline (graph.py).  Keep this module free of heavy imports so it can be
# imported from anywhere without triggering circular dependencies.

from pydantic import BaseModel, Field


class ContextEntry(BaseModel):
    """A single typed context item carried through the RAG pipeline.

    `type` is a free-form discriminator: ``file`` | ``selection`` |
    ``document`` | ``snippet`` | ``url`` | ``image`` | …
    """

    type: str = Field(
        ...,
        description="Entry kind: `file`, `selection`, `document`, `snippet`, `url`, `image`, etc.",
        examples=["file"],
    )
    name: str | None = Field(
        None,
        description="Human-readable name or file path for this entry.",
        examples=["data/paper.pdf"],
    )
    content: str | None = Field(
        None,
        description="Text content of the entry (plain text or pre-extracted).",
    )
    mimeType: str | None = Field(
        None,
        description="MIME type of the original content, e.g. `application/pdf` or `text/plain`.",
        examples=["application/pdf"],
    )
    range: dict | None = Field(
        None,
        description="Character-offset range `{\"start\": int, \"end\": int}` for `selection` entries.",
        examples=[{"start": 0, "end": 512}],
    )
    score: float | None = Field(
        None,
        description="Relevance score assigned by the retriever (higher = more relevant).",
        examples=[0.87],
    )
    source: str | None = Field(
        None,
        description="Origin of this entry, e.g. `\"retriever\"` for ChromaDB results.",
        examples=["retriever"],
    )

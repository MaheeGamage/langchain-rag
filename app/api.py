# app/api.py

import json
from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel
from .graph import graph
from .config import LLM_MODEL, EMBEDDING_MODEL, LLM_PROVIDER, EMBEDDING_PROVIDER

app = FastAPI()


class QueryRequest(BaseModel):
    question: str
    context: Any = None # Accept either a string or a structured object as context


class SourceChunk(BaseModel):
    content: str
    metadata: dict


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/config")
async def config():
    return {
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "llm_provider": LLM_PROVIDER,
        "embedding_provider": EMBEDDING_PROVIDER,
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    # Convert context to string if it is a structured object
    ctx = req.context
    if ctx is None:
        context_str = ""
    elif isinstance(ctx, str):
        context_str = ctx
    else:
        context_str = json.dumps(ctx, indent=2)
    result = graph.invoke({"question": req.question, "context": context_str})
    sources = [
        SourceChunk(content=doc.page_content, metadata=doc.metadata)
        for doc in result.get("documents", [])
    ]
    return QueryResponse(answer=result["answer"], sources=sources)

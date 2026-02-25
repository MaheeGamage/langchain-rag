# app/api.py

from fastapi import FastAPI
from pydantic import BaseModel
from .graph import graph
from .config import LLM_MODEL, EMBEDDING_MODEL, LLM_PROVIDER, EMBEDDING_PROVIDER

app = FastAPI()


class QueryRequest(BaseModel):
    question: str


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
    result = graph.invoke({"question": req.question})
    sources = [
        SourceChunk(content=doc.page_content, metadata=doc.metadata)
        for doc in result.get("documents", [])
    ]
    return QueryResponse(answer=result["answer"], sources=sources)

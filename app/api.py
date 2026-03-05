# app/api.py

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from .graph import graph
from .models import ContextEntry
from .config import LLM_MODEL, EMBEDDING_MODEL, LLM_PROVIDER, EMBEDDING_PROVIDER

app = FastAPI()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ConversationTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class Conversation(BaseModel):
    id: str | None = None
    history: list[ConversationTurn] = []


class ContextPayload(BaseModel):
    entries: list[ContextEntry] = []


class QueryOptions(BaseModel):
    model: str | None = None
    stream: bool = False
    maxTokens: int | None = None
    temperature: float | None = None


class QueryMeta(BaseModel):
    clientId: str | None = None
    workspaceId: str | None = None
    userId: str | None = None


class QueryRequest(BaseModel):
    message: str
    conversation: Conversation | None = None
    context: ContextPayload | None = None
    options: QueryOptions | None = None
    meta: QueryMeta | None = None


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class SourceChunk(BaseModel):
    content: str
    metadata: dict


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

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
    # Build LangChain messages from conversation history + current message
    messages = []
    if req.conversation and req.conversation.history:
        for turn in req.conversation.history:
            if turn.role == "user":
                messages.append(HumanMessage(content=turn.content))
            else:
                messages.append(AIMessage(content=turn.content))
    messages.append(HumanMessage(content=req.message))

    context_entries = req.context.entries if req.context else []

    result = graph.invoke({
        "messages": messages,
        "context": context_entries,
        "retrieved": [],
    })

    # Extract the answer from the last AIMessage in the returned messages
    answer = ""
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage):
            answer = m.content
            break

    sources = [
        SourceChunk(
            content=entry.content or "",
            metadata={
                "source": entry.name or "",
                **({"score": entry.score} if entry.score is not None else {}),
            },
        )
        for entry in result.get("retrieved", [])
    ]
    return QueryResponse(answer=answer, sources=sources)

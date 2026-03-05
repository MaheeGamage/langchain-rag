from app.models import ContextEntry
from pydantic import BaseModel

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


from app.models import ContextEntry
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ConversationTurn(BaseModel):
    """A single turn in a conversation."""

    role: str = Field(
        ...,
        description="Speaker role: `user` or `assistant`.",
        examples=["user"],
    )
    content: str = Field(
        ...,
        description="Text content of this turn.",
        examples=["What is retrieval-augmented generation?"],
    )


class Conversation(BaseModel):
    """Conversation state for multi-turn dialogue."""

    id: str | None = Field(
        None,
        description="Optional client-generated conversation identifier.",
        examples=["conv-abc123"],
    )
    history: list[ConversationTurn] = Field(
        default_factory=list,
        description="Ordered list of prior turns (oldest first). Omit for a fresh conversation.",
    )


class ContextPayload(BaseModel):
    """Pre-retrieved or injected context items."""

    entries: list[ContextEntry] = Field(
        default_factory=list,
        description=(
            "Zero or more context entries (files, selections, snippets, …) "
            "that will be injected directly into the prompt in addition to "
            "ChromaDB retrieval results."
        ),
    )


class QueryOptions(BaseModel):
    """Optional per-request LLM tuning knobs."""

    model: str | None = Field(
        None,
        description="Override the default LLM model name for this request.",
        examples=["gpt-4o"],
    )
    stream: bool = Field(
        False,
        description="Reserved — streaming is not yet implemented.",
    )
    maxTokens: int | None = Field(
        None,
        description="Maximum tokens in the generated answer.",
        examples=[512],
    )
    temperature: float | None = Field(
        None,
        description="Sampling temperature (0 = deterministic, 1 = creative).",
        examples=[0.2],
    )


class QueryMeta(BaseModel):
    """Optional caller-provided metadata for logging / routing."""

    clientId: str | None = Field(None, description="Identifies the calling client application.", examples=["vscode-ext"])
    workspaceId: str | None = Field(None, description="Workspace or project identifier.", examples=["ws-42"])
    userId: str | None = Field(None, description="End-user identifier.", examples=["user-99"])


class QueryRequest(BaseModel):
    """Request body for `POST /query`."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "What is retrieval-augmented generation?",
                    "conversation": {"id": "conv-1", "history": []},
                    "context": {"entries": []},
                    "options": {"temperature": 0.2},
                    "meta": {"clientId": "vscode-ext"},
                }
            ]
        }
    }

    message: str = Field(
        ...,
        description="The user's question or instruction.",
        examples=["What is retrieval-augmented generation?"],
    )
    conversation: Conversation | None = Field(
        None,
        description="Prior conversation history for multi-turn dialogue.",
    )
    context: ContextPayload | None = Field(
        None,
        description="Additional context entries to inject into the prompt.",
    )
    options: QueryOptions | None = Field(
        None,
        description="Optional LLM tuning parameters.",
    )
    meta: QueryMeta | None = Field(
        None,
        description="Optional caller metadata (logging / routing).",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class SourceChunk(BaseModel):
    """A single retrieved document chunk used to ground the answer."""

    content: str = Field(..., description="Raw text of the retrieved chunk.")
    metadata: dict = Field(
        ...,
        description="Chunk metadata: `source` (file path), optional `score`.",
        examples=[{"source": "data/paper.pdf", "score": 0.87}],
    )


class QueryResponse(BaseModel):
    """Response body from `POST /query`."""

    thread_id: str | None = Field(None, description="Conversation thread identifier. Pass this back on subsequent turns to continue the conversation.")
    answer: str = Field(..., description="LLM-generated answer grounded in the retrieved chunks.")
    sources: list[SourceChunk] = Field(
        ...,
        description="Document chunks retrieved from ChromaDB that were used to produce the answer.",
    )


# ---------------------------------------------------------------------------
# Evaluation models
# ---------------------------------------------------------------------------

VALID_SCORERS: frozenset[str] = frozenset({
    "relevance", "correctness", "fluency", "groundedness",
    "sufficiency", "safety", "conciseness", "domain_tone",
})


class EvalSample(BaseModel):
    """A single question/answer pair for evaluation."""

    question: str = Field(..., description="The question to run through the RAG pipeline.")
    expected_response: str | None = Field(
        None,
        description="Ground-truth answer used by the `correctness` scorer.",
    )
    expected_facts: list[str] | None = Field(
        None,
        description="Key facts the answer must contain, used by fact-checking scorers.",
    )


class EvalRequest(BaseModel):
    """Request body for `POST /evaluate`."""

    dataset: list[EvalSample] = Field(
        ...,
        min_length=1,
        description="One or more evaluation samples.",
    )
    scorers: list[str] = Field(
        default=["relevance", "fluency", "groundedness"],
        description=(
            f"Scorers to run. Available: {sorted(VALID_SCORERS)}. "
            "RAG-specific scorers (`groundedness`, `sufficiency`) inspect the "
            "retrieval trace and require MLflow tracing to be active."
        ),
    )
    judge_model: str | None = Field(
        None,
        description=(
            "LiteLLM-format judge model override, e.g. `openai:/gpt-4o-mini` or "
            "`ollama:/llama3.2`. Defaults to the `MLFLOW_JUDGE_MODEL` env var."
        ),
        examples=["openai:/gpt-4o-mini"],
    )
    experiment_name: str | None = Field(
        None,
        description="MLflow experiment name. Defaults to `MLFLOW_EXPERIMENT_NAME` env var.",
    )


class EvalResponse(BaseModel):
    """Response body from `POST /evaluate`."""

    run_id: str = Field(..., description="MLflow run ID for this evaluation run.")
    experiment_name: str = Field(..., description="MLflow experiment name used.")
    metrics: dict[str, float] = Field(
        ...,
        description="Aggregate metrics across all samples (mean/p90 per scorer).",
    )
    rows: list[dict] = Field(
        ...,
        description="Per-sample scores. Each entry mirrors one input sample.",
    )


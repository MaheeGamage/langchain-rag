from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.models import ContextEntry


class GraphState(TypedDict):
    """Shared state shape used by graph implementations."""

    messages: Annotated[list[BaseMessage], add_messages]
    context: list[ContextEntry]
    retrieved: list[ContextEntry]

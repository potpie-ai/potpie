from dataclasses import dataclass
from datetime import datetime


@dataclass
class ChatMessage:
    """A message in a conversation."""

    role: str
    content: str
    timestamp: datetime
    metadata: dict | None = None


@dataclass
class ChatResponse:
    """Response from the LLM API."""

    message: ChatMessage
    usage: dict
    timestamp: datetime

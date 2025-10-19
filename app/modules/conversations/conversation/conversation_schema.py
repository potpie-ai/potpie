from datetime import datetime
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel

from app.modules.conversations.conversation.conversation_model import (
    ConversationStatus,
    Visibility,
)


class CreateConversationRequest(BaseModel):
    user_id: str
    title: str
    status: ConversationStatus
    project_ids: List[str]
    agent_ids: List[str]


class ConversationAccessType(str, Enum):
    """
    Enum for access type
    """

    READ = "read"
    WRITE = "write"
    NOT_FOUND = "not_found"


class CreateConversationResponse(BaseModel):
    message: str
    conversation_id: str


class ConversationInfoResponse(BaseModel):
    id: str
    title: str
    status: ConversationStatus
    project_ids: List[str]
    created_at: datetime
    updated_at: datetime
    total_messages: int
    agent_ids: List[str]
    access_type: ConversationAccessType
    is_creator: bool
    creator_id: str
    visibility: Optional[Visibility] = None

    class Config:
        from_attributes = True


class ChatMessageResponse(BaseModel):
    message: str
    citations: List[str]
    tool_calls: List[Any]


# Resolve forward references
ConversationInfoResponse.update_forward_refs()


class RenameConversationRequest(BaseModel):
    title: str


# Frontend-aligned schemas for session endpoints
class ActiveSessionResponse(BaseModel):
    sessionId: str
    status: str  # "active", "idle", "completed"
    cursor: str
    conversationId: str
    startedAt: int  # Unix timestamp in milliseconds
    lastActivity: int  # Unix timestamp in milliseconds


class ActiveSessionErrorResponse(BaseModel):
    error: str
    conversationId: str


class TaskStatusResponse(BaseModel):
    isActive: bool
    sessionId: str
    estimatedCompletion: int  # Unix timestamp in milliseconds
    conversationId: str


class TaskStatusErrorResponse(BaseModel):
    error: str
    conversationId: str

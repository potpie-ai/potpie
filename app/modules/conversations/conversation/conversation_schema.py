from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel

from app.modules.conversations.conversation.conversation_model import ConversationStatus
from app.modules.conversations.message.message_model import MessageType, MessageStatus  # Updated import


class CreateConversationRequest(BaseModel):
    user_id: str
    title: str
    status: ConversationStatus
    project_ids: List[str]
    agent_ids: List[str]


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

    class Config:
        from_attributes = True


# Resolve forward references
ConversationInfoResponse.update_forward_refs()


class RenameConversationRequest(BaseModel):
    title: str

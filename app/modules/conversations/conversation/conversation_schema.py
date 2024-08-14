from pydantic import BaseModel
from typing import List
from datetime import datetime
from app.modules.conversations.conversation.conversation_model import ConversationStatus
from app.modules.conversations.message.message_schema import MessageResponse

class CreateConversationRequest(BaseModel):
    user_id: str
    title: str
    status: ConversationStatus
    project_ids: List[str]
    agent_ids: List[str]

class CreateConversationResponse(BaseModel):
    message: str
    conversation_id: str

class ConversationResponse(BaseModel):
    id: str
    user_id: str
    title: str
    status: ConversationStatus
    project_ids: List[str]
    agent_ids: List[str]
    created_at: datetime
    updated_at: datetime
    messages: List["MessageResponse"] = []

    class Config:
        from_attributes = True

class ConversationInfoResponse(BaseModel):
    id: str
    agent_ids: List[str]
    project_ids: List[str]
    total_messages: int 
    class Config:
        from_attributes = True

# Resolve forward references
ConversationResponse.update_forward_refs()
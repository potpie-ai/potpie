from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.modules.conversations.message.message_model import MessageStatus, MessageType


class MessageRequest(BaseModel):
    content: str


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    content: str
    sender_id: Optional[str] = None
    type: MessageType
    reason: Optional[str] = None
    created_at: datetime
    status: MessageStatus

    class Config:
        from_attributes = True

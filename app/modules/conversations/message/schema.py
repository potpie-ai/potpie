from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from .model import MessageType

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

    class Config:
        from_attributes = True
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel

from app.modules.conversations.message.message_model import MessageStatus, MessageType


class NodeContext(BaseModel):
    node_id: str
    name: str


class MessageRequest(BaseModel):
    content: str
    node_ids: Optional[List[NodeContext]] = None


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

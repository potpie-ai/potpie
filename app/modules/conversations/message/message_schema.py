from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from app.modules.conversations.message.message_model import MessageStatus, MessageType
from app.modules.media.media_schema import AttachmentInfo


class NodeContext(BaseModel):
    node_id: str
    name: str


class MessageRequest(BaseModel):
    content: str
    node_ids: Optional[List[NodeContext]] = None
    attachment_ids: Optional[List[str]] = None  # IDs of uploaded attachments
    tunnel_url: Optional[str] = None  # Optional tunnel URL from extension (takes priority over stored state)


class DirectMessageRequest(BaseModel):
    content: str
    node_ids: Optional[List[NodeContext]] = None
    agent_id: str | None = None
    attachment_ids: Optional[List[str]] = None  # IDs of uploaded attachments


class RegenerateRequest(BaseModel):
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
    citations: Optional[List[str]] = None
    has_attachments: bool = False
    attachments: Optional[List[AttachmentInfo]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    thinking: Optional[str] = None

    class Config:
        from_attributes = True

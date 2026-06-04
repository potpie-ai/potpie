from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from app.modules.conversations.message.message_model import MessageStatus, MessageType
from app.modules.media.media_schema import AttachmentInfo


class NodeContext(BaseModel):
    node_id: str
    name: str


def normalize_node_contexts(node_ids: Optional[List[Any]]) -> List[NodeContext]:
    """Coerce loosely-typed node ids into NodeContext objects.

    Queued payloads vary by backend: Celery may carry NodeContext instances
    (e.g. in tests), while Hatchet delivers JSON — dict-shaped node ids or
    plain string ids. Centralizing the coercion keeps the message and
    regenerate paths in sync and avoids MessageRequest validation failures on
    string node ids.
    """
    node_contexts: List[NodeContext] = []
    for node in node_ids or []:
        if isinstance(node, NodeContext):
            node_contexts.append(node)
        elif isinstance(node, dict):
            node_id = str(node.get("node_id") or node.get("id") or "")
            if node_id:
                node_contexts.append(
                    NodeContext(node_id=node_id, name=str(node.get("name") or node_id))
                )
        else:
            node_id = str(node)
            node_contexts.append(NodeContext(node_id=node_id, name=node_id))
    return node_contexts


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

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.modules.conversations.conversation.conversation_model import (
    Conversation,
    ConversationStatus,
    Visibility,
)
from app.modules.conversations.message.message_model import Message, MessageStatus, MessageType
from seed.schemas import ConversationsFile

_PLACEHOLDER = "__TARGET_USER_UID__"


def _conv_status(s: str) -> ConversationStatus:
    try:
        return ConversationStatus[s.upper()]
    except KeyError:
        return ConversationStatus.ACTIVE


def _vis(s: str | None) -> Visibility | None:
    if s is None:
        return None
    try:
        return Visibility[s.upper()]
    except KeyError:
        return Visibility.PRIVATE


def _msg_type(s: str) -> MessageType:
    return MessageType[s]


def _msg_status(s: str) -> MessageStatus:
    return MessageStatus[s]


def _substitute_uid(text: str | None, uid: str) -> str | None:
    if text is None:
        return None
    return text.replace(_PLACEHOLDER, uid)


def apply_conversations(
    db: Session, data: ConversationsFile, user_id: str
) -> list[str]:
    ids: list[str] = []
    for c in data.conversations:
        existing = db.query(Conversation).filter(Conversation.id == c.id).first()
        if existing:
            if existing.user_id != user_id:
                raise ValueError(
                    f"Conversation {c.id!r} belongs to user {existing.user_id}"
                )
            existing.title = c.title
            existing.status = _conv_status(c.status)
            existing.project_ids = list(c.project_ids)
            existing.agent_ids = list(c.agent_ids)
            existing.shared_with_emails = c.shared_with_emails
            existing.visibility = _vis(c.visibility)
        else:
            conv = Conversation(
                id=c.id,
                user_id=user_id,
                title=c.title,
                status=_conv_status(c.status),
                project_ids=list(c.project_ids),
                agent_ids=list(c.agent_ids),
                shared_with_emails=c.shared_with_emails,
                visibility=_vis(c.visibility),
            )
            db.add(conv)
        ids.append(c.id)

        # Replace messages: delete existing for this conversation then insert
        db.query(Message).filter(Message.conversation_id == c.id).delete(
            synchronize_session=False
        )

        for m in c.messages:
            sid = m.sender_id
            if sid is not None:
                sid = _substitute_uid(sid, user_id)
            content = _substitute_uid(m.content, user_id) or ""
            ct = _msg_type(m.type)
            if ct == MessageType.HUMAN and not sid:
                raise ValueError(
                    f"Message {m.id}: HUMAN messages require sender_id (use { _PLACEHOLDER!r} or explicit uid)"
                )
            if ct != MessageType.HUMAN and sid:
                sid = None
            msg = Message(
                id=m.id,
                conversation_id=c.id,
                content=content,
                sender_id=sid,
                type=ct,
                status=_msg_status(m.status),
                created_at=m.created_at or datetime.now(timezone.utc),
                citations=m.citations,
                has_attachments=m.has_attachments,
                tool_calls=m.tool_calls,
                thinking=m.thinking,
            )
            db.add(msg)
    db.commit()
    return ids

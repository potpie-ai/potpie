from typing import List
from datetime import datetime
from sqlalchemy import select, func, cast, String, update, delete
from app.core.base_store import BaseStore
from .message_model import Message, MessageStatus, MessageType


class MessageStore(BaseStore):
    """Handles all database operations for the Message model."""

    async def count_active_for_conversation(self, conversation_id: str) -> int:
        stmt = select(func.count(Message.id)).where(
            Message.conversation_id == conversation_id,
            cast(Message.status, String) == MessageStatus.ACTIVE.value,
        )
        result = await self.async_db.execute(stmt)
        return result.scalar() or 0

    async def get_active_for_conversation(
        self, conversation_id: str, offset: int, limit: int
    ) -> List[Message]:
        stmt = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                cast(Message.status, String) == MessageStatus.ACTIVE.value,
                cast(Message.type, String) != MessageType.SYSTEM_GENERATED.value,
            )
            .order_by(Message.created_at)
            .offset(offset)
            .limit(limit)
        )
        result = await self.async_db.execute(stmt)
        return result.scalars().all()

    async def get_last_human_message(self, conversation_id: str) -> Message | None:
        stmt = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.type == MessageType.HUMAN,
            )
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        result = await self.async_db.execute(stmt)
        return result.scalar_one_or_none()

    async def archive_messages_after(
        self, conversation_id: str, timestamp: datetime
    ) -> None:
        stmt = (
            update(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.created_at > timestamp,
            )
            .values(status=MessageStatus.ARCHIVED)
        )
        await self.async_db.execute(stmt)
        await self.async_db.commit()

    async def get_latest_human_message_with_attachments(
        self, conversation_id: str
    ) -> Message | None:
        stmt = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.type == MessageType.HUMAN,
                Message.status == MessageStatus.ACTIVE,
                Message.has_attachments,
            )
            .order_by(Message.created_at.desc())
        )
        result = await self.async_db.execute(stmt)
        return result.scalars().first()

    async def delete_for_conversation(self, conversation_id: str) -> int:
        stmt = delete(Message).where(Message.conversation_id == conversation_id)
        result = await self.async_db.execute(stmt)
        return result.rowcount

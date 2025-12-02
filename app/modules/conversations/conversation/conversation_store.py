from typing import List
from datetime import datetime, timezone
from sqlalchemy import select, func, update, delete
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import selectinload
from app.core.base_store import BaseStore
from .conversation_model import Conversation
from ..message.message_model import Message, MessageType
import logging

logger = logging.getLogger(__name__)


class StoreError(Exception):
    pass


class ConversationStore(BaseStore):
    """Handles all database operations for the Conversation model."""

    async def get_by_id(self, conversation_id: str) -> Conversation | None:
        stmt = select(Conversation).where(Conversation.id == conversation_id)
        result = await self.async_db.execute(stmt)
        return result.scalar_one_or_none()

    async def create(self, new_conversation: Conversation) -> None:
        self.async_db.add(new_conversation)
        await self.async_db.commit()

    async def get_with_message_count(self, conversation_id: str) -> Conversation | None:
        stmt = (
            select(
                Conversation,
                func.count(Message.id)
                .filter(Message.type == MessageType.HUMAN)
                .label("human_message_count"),
            )
            .outerjoin(Message, Conversation.id == Message.conversation_id)
            .where(Conversation.id == conversation_id)
            .group_by(Conversation.id)
        )

        result = await self.async_db.execute(stmt)
        row = result.first()
        if row:
            conversation, human_message_count = row
            setattr(conversation, "human_message_count", human_message_count)
            return conversation
        return None

    async def update_title(self, conversation_id: str, new_title: str) -> None:
        stmt = (
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(title=new_title, updated_at=datetime.now(timezone.utc))
        )
        await self.async_db.execute(stmt)
        await self.async_db.commit()

    async def delete(self, conversation_id: str) -> int:
        stmt = delete(Conversation).where(Conversation.id == conversation_id)
        result = await self.async_db.execute(stmt)
        await self.async_db.commit()

        return result.rowcount

    async def get_for_user(
        self,
        user_id: str,
        start: int,
        limit: int,
        sort: str = "updated_at",
        order: str = "desc",
    ) -> List[Conversation]:
        """
        Fetches a paginated and sorted list of conversations for a specific user.
        """
        try:
            # Build the base query using the async `select` statement
            stmt = (
                select(Conversation)
                .where(Conversation.user_id == user_id)
                # Eagerly load the 'projects' relationship using selectinload
                .options(selectinload(Conversation.projects))
            )
            # Validate sort field to prevent errors or injection
            if sort not in ["updated_at", "created_at"]:
                sort = "updated_at"  # Default to updated_at if invalid

            sort_column = getattr(Conversation, sort)

            # Validate and apply order
            if order.lower() == "asc":
                stmt = stmt.order_by(sort_column.asc())
            else:  # Default to desc
                stmt = stmt.order_by(sort_column.desc())

            # Apply pagination
            stmt = stmt.offset(start).limit(limit)

            # Execute the query asynchronously
            result = await self.async_db.execute(stmt)
            conversations = result.scalars().unique().all()

            logger.info(
                f"Retrieved {len(conversations)} conversations for user {user_id}"
            )
            return conversations

        except SQLAlchemyError as e:
            logger.error(
                f"Database error in get_for_user for user {user_id}: {e}",
                exc_info=True,
            )
            # Re-raise with a generic store error to not leak details
            raise StoreError(
                f"Failed to retrieve conversations for user {user_id}"
            ) from e

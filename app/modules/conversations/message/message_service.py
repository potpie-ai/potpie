import asyncio
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session
from uuid6 import uuid7
from sqlalchemy.exc import IntegrityError
from app.modules.conversations.message.message_model import Message, MessageType

class MessageService:
    def __init__(self, db: Session):
        self.db = db 

    async def create_message(self, conversation_id: str, content: str, message_type: MessageType, sender_id: Optional[str] = None) -> Message:
        # Validate sender_id based on message_type
        if (message_type == MessageType.HUMAN and sender_id is None) or \
           (message_type in {MessageType.AI_GENERATED, MessageType.SYSTEM_GENERATED} and sender_id is not None):
            raise ValueError("Invalid sender_id for the given message_type.")

        message_id = str(uuid7())
        new_message = Message(
            id=message_id,
            conversation_id=conversation_id,
            content=content,
            type=message_type,
            created_at=datetime.now(timezone.utc),
            sender_id=sender_id,
        )

        try:
            await asyncio.get_event_loop().run_in_executor(None, self._sync_db_add, new_message)
            await asyncio.get_event_loop().run_in_executor(None, self._sync_db_commit)
            await asyncio.get_event_loop().run_in_executor(None, self._sync_db_refresh, new_message)
            return new_message
        except IntegrityError as e:
            # Rollback in case of error
            await asyncio.get_event_loop().run_in_executor(None, self._sync_db_rollback)
            raise e

    def _sync_db_add(self, instance):
        """Synchronous database add operation."""
        self.db.add(instance)

    def _sync_db_commit(self):
        """Synchronous database commit operation."""
        self.db.commit()

    def _sync_db_refresh(self, instance):
        """Synchronous database refresh operation."""
        self.db.refresh(instance)

    def _sync_db_rollback(self):
        """Synchronous database rollback operation."""
        self.db.rollback()

    async def commit_and_refresh(self, instance):
        """Commits the current transaction and refreshes the instance."""
        try:
            await asyncio.get_event_loop().run_in_executor(None, self._sync_db_commit)
            await asyncio.get_event_loop().run_in_executor(None, self._sync_db_refresh, instance)
        except IntegrityError as e:
            await asyncio.get_event_loop().run_in_executor(None, self._sync_db_rollback)
            raise e

    async def commit(self):
        """Commits the current transaction."""
        try:
            await asyncio.get_event_loop().run_in_executor(None, self._sync_db_commit)
        except IntegrityError as e:
            await asyncio.get_event_loop().run_in_executor(None, self._sync_db_rollback)
            raise e

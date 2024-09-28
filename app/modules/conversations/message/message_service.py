import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from uuid6 import uuid7

from app.modules.conversations.message.message_model import (
    Message,
    MessageStatus,
    MessageType,
)

logger = logging.getLogger(__name__)


class MessageServiceError(Exception):
    """Base exception class for MessageService errors."""


class MessageNotFoundError(MessageServiceError):
    """Raised when a message is not found."""


class InvalidMessageError(MessageServiceError):
    """Raised when there's an issue with message creation parameters."""


class MessageService:
    def __init__(self, db: Session):
        self.db = db

    async def create_message(
        self,
        conversation_id: str,
        content: str,
        message_type: MessageType,
        sender_id: Optional[str] = None,
    ) -> Message:
        try:
            if (message_type == MessageType.HUMAN and sender_id is None) or (
                message_type in {MessageType.AI_GENERATED, MessageType.SYSTEM_GENERATED}
                and sender_id is not None
            ):
                raise InvalidMessageError(
                    "Invalid sender_id for the given message_type."
                )

            message_id = str(uuid7())
            new_message = Message(
                id=message_id,
                conversation_id=conversation_id,
                content=content,
                type=message_type,
                created_at=datetime.now(timezone.utc),
                sender_id=sender_id,
                status=MessageStatus.ACTIVE,
            )

            await asyncio.get_event_loop().run_in_executor(
                None, self._sync_create_message, new_message
            )
            logger.info(
                f"Created new message with ID: {message_id} for conversation: {conversation_id}"
            )
            return new_message

        except InvalidMessageError as e:
            logger.warning(f"Invalid message parameters: {str(e)}")
            raise

        except IntegrityError as e:
            logger.error(
                f"Database integrity error in create_message: {e}", exc_info=True
            )
            raise MessageServiceError(
                "Failed to create message due to a database integrity error"
            ) from e

        except Exception as e:
            logger.error(f"Unexpected error in create_message: {e}", exc_info=True)
            raise MessageServiceError(
                "An unexpected error occurred while creating the message"
            ) from e

    def _sync_create_message(self, new_message: Message):
        try:
            self.db.add(new_message)
            self.db.commit()
            self.db.refresh(new_message)
        except SQLAlchemyError:
            self.db.rollback()
            raise

    async def mark_message_archived(self, message_id: str) -> None:
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._sync_mark_message_archived, message_id
            )
            # TODO: add conversation_id to the log
            logger.info(f"Marked message {message_id} as archived")
        except MessageNotFoundError as e:
            logger.warning(str(e))
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error in mark_message_archived: {e}", exc_info=True)
            raise MessageServiceError(
                f"Failed to archive message {message_id} due to a database error"
            ) from e
        except Exception as e:
            logger.error(
                f"Unexpected error in mark_message_archived: {e}", exc_info=True
            )
            raise MessageServiceError(
                f"An unexpected error occurred while archiving message {message_id}"
            ) from e

    def _sync_mark_message_archived(self, message_id: str):
        try:
            message = (
                self.db.query(Message).filter(Message.id == message_id).one_or_none()
            )
            if message:
                message.status = MessageStatus.ARCHIVED
                self.db.commit()
            else:
                raise MessageNotFoundError(f"Message with id {message_id} not found.")
        except SQLAlchemyError:
            self.db.rollback()
            raise

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from uuid6 import uuid7

from app.modules.conversations.message.message_model import (
    Message,
    MessageStatus,
    MessageType,
)

logger = logging.getLogger(__name__)


class ChatHistoryServiceError(Exception):
    """Base exception class for ChatHistoryService errors."""


class ChatHistoryService:
    def __init__(self, db: Session):
        self.db = db
        self.message_buffer: Dict[str, Dict[str, str]] = {}

    def get_session_history(
        self, user_id: str, conversation_id: str
    ) -> List[BaseMessage]:
        try:
            messages = (
                self.db.query(Message)
                .filter_by(conversation_id=conversation_id)
                .filter_by(status=MessageStatus.ACTIVE)  # Only fetch active messages
                .order_by(Message.created_at)
                .all()
            )
            history = []
            for msg in messages:
                if msg.type == MessageType.HUMAN:
                    history.append(HumanMessage(content=msg.content))
                else:
                    history.append(AIMessage(content=msg.content))
            logger.info(
                f"Retrieved session history for conversation: {conversation_id}"
            )
            return history
        except SQLAlchemyError as e:
            logger.error(
                f"Database error in get_session_history for conversation {conversation_id}: {e}",
                exc_info=True,
            )
            raise ChatHistoryServiceError(
                f"Failed to retrieve session history for conversation {conversation_id}"
            ) from e
        except Exception as e:
            logger.error(
                f"Unexpected error in get_session_history for conversation {conversation_id}: {e}",
                exc_info=True,
            )
            raise ChatHistoryServiceError(
                f"An unexpected error occurred while retrieving session history for conversation {conversation_id}"
            ) from e

    def add_message_chunk(
        self,
        conversation_id: str,
        content: str,
        message_type: MessageType,
        sender_id: Optional[str] = None,
        citations: Optional[List[str]] = None,
    ):
        if conversation_id not in self.message_buffer:
            self.message_buffer[conversation_id] = {"content": "", "citations": []}
        self.message_buffer[conversation_id]["content"] += content
        if citations:
            self.message_buffer[conversation_id]["citations"].extend(citations)
        logger.debug(
            f"Added message chunk to buffer for conversation: {conversation_id}"
        )

    def flush_message_buffer(
        self,
        conversation_id: str,
        message_type: MessageType,
        sender_id: Optional[str] = None,
    ):
        try:
            if (
                conversation_id in self.message_buffer
                and self.message_buffer[conversation_id]["content"]
            ):
                content = self.message_buffer[conversation_id]["content"]
                citations = self.message_buffer[conversation_id]["citations"]

                new_message = Message(
                    id=str(uuid7()),
                    conversation_id=conversation_id,
                    content=content,
                    sender_id=sender_id if message_type == MessageType.HUMAN else None,
                    type=message_type,
                    created_at=datetime.now(timezone.utc),
                    citations=(
                        ",".join(set(citations)) if citations else None
                    ),  # Use set to remove duplicates
                )
                self.db.add(new_message)
                self.db.commit()
                self.message_buffer[conversation_id] = {"content": "", "citations": []}
                logger.info(
                    f"Flushed message buffer for conversation: {conversation_id}"
                )
        except SQLAlchemyError as e:
            logger.error(
                f"Database error in flush_message_buffer for conversation {conversation_id}: {e}",
                exc_info=True,
            )
            self.db.rollback()
            raise ChatHistoryServiceError(
                f"Failed to flush message buffer for conversation {conversation_id}"
            ) from e
        except Exception as e:
            logger.error(
                f"Unexpected error in flush_message_buffer for conversation {conversation_id}: {e}",
                exc_info=True,
            )
            self.db.rollback()
            raise ChatHistoryServiceError(
                f"An unexpected error occurred while flushing message buffer for conversation {conversation_id}"
            ) from e

    def clear_session_history(self, conversation_id: str):
        try:
            self.db.query(Message).filter_by(conversation_id=conversation_id).delete()
            self.db.commit()
            logger.info(f"Cleared session history for conversation: {conversation_id}")
        except SQLAlchemyError as e:
            logger.error(
                f"Database error in clear_session_history for conversation {conversation_id}: {e}",
                exc_info=True,
            )
            self.db.rollback()
            raise ChatHistoryServiceError(
                f"Failed to clear session history for conversation {conversation_id}"
            ) from e
        except Exception as e:
            logger.error(
                f"Unexpected error in clear_session_history for conversation {conversation_id}: {e}",
                exc_info=True,
            )
            self.db.rollback()
            raise ChatHistoryServiceError(
                f"An unexpected error occurred while clearing session history for conversation {conversation_id}"
            ) from e

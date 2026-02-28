from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, UserPromptPart, TextPart
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from uuid6 import uuid7

from app.modules.conversations.message.message_model import (
    Message,
    MessageStatus,
    MessageType,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class ChatHistoryServiceError(Exception):
    """Base exception class for ChatHistoryService errors."""


class ChatHistoryService:
    def __init__(self, db: Session):
        self.db = db
        self.message_buffer: Dict[str, Dict[str, Any]] = {}

    def get_session_history(
        self, user_id: str, conversation_id: str
    ) -> List[ModelMessage]:
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
                    history.append(
                        ModelRequest(parts=[UserPromptPart(content=msg.content)])
                    )
                else:
                    history.append(ModelResponse(parts=[TextPart(content=msg.content)]))
            logger.info(
                f"Retrieved session history for conversation: {conversation_id}"
            )
            return history
        except SQLAlchemyError as e:
            logger.exception(
                "Database error in get_session_history",
                conversation_id=conversation_id,
                user_id=user_id,
            )
            raise ChatHistoryServiceError(
                f"Failed to retrieve session history for conversation {conversation_id}"
            ) from e
        except Exception as e:
            logger.exception(
                "Unexpected error in get_session_history",
                conversation_id=conversation_id,
                user_id=user_id,
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
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        thinking: Optional[str] = None,
    ):
        if conversation_id not in self.message_buffer:
            self.message_buffer[conversation_id] = {
                "content": "",
                "citations": [],
                "tool_calls": [],
                "thinking": None,
            }
        self.message_buffer[conversation_id]["content"] += content
        if citations:
            self.message_buffer[conversation_id]["citations"].extend(citations)
        if tool_calls:
            self.message_buffer[conversation_id]["tool_calls"].extend(tool_calls)
        if thinking:
            if self.message_buffer[conversation_id]["thinking"] is None:
                self.message_buffer[conversation_id]["thinking"] = thinking
            else:
                self.message_buffer[conversation_id]["thinking"] += thinking
        logger.debug(
            f"Added message chunk to buffer for conversation: {conversation_id}"
        )

    def flush_message_buffer(
        self,
        conversation_id: str,
        message_type: MessageType,
        sender_id: Optional[str] = None,
        thinking: Optional[str] = None,
    ) -> Optional[str]:
        try:
            if (
                conversation_id in self.message_buffer
                and self.message_buffer[conversation_id]["content"]
            ):
                content = self.message_buffer[conversation_id]["content"]
                citations = self.message_buffer[conversation_id]["citations"]
                tool_calls = self.message_buffer[conversation_id].get("tool_calls", [])
                # Use provided thinking if given, otherwise fall back to buffer
                thinking = thinking or self.message_buffer[conversation_id].get("thinking")

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
                    tool_calls=tool_calls if tool_calls else None,
                    thinking=thinking,
                )
                self.db.add(new_message)
                self.db.commit()
                self.message_buffer[conversation_id] = {
                    "content": "",
                    "citations": [],
                    "tool_calls": [],
                    "thinking": None,
                }
                logger.info(
                    f"Flushed message buffer for conversation: {conversation_id}"
                )
                return new_message.id
            return None
        except SQLAlchemyError as e:
            logger.exception(
                "Database error in flush_message_buffer",
                conversation_id=conversation_id,
            )
            self.db.rollback()
            raise ChatHistoryServiceError(
                f"Failed to flush message buffer for conversation {conversation_id}"
            ) from e
        except Exception as e:
            logger.exception(
                "Unexpected error in flush_message_buffer",
                conversation_id=conversation_id,
            )
            self.db.rollback()
            raise ChatHistoryServiceError(
                f"An unexpected error occurred while flushing message buffer for conversation {conversation_id}"
            ) from e

    def save_partial_ai_message(
        self,
        conversation_id: str,
        content: str,
        citations: Optional[List[str]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        thinking: Optional[str] = None,
    ) -> Optional[str]:
        """
        Persist a complete AI message (e.g. partial response saved when generation is stopped).
        Used so the user can continue the conversation and build on this progress.
        """
        if not content.strip():
            return None
        try:
            new_message = Message(
                id=str(uuid7()),
                conversation_id=conversation_id,
                content=content.strip(),
                sender_id=None,
                type=MessageType.AI_GENERATED,
                status=MessageStatus.ACTIVE,
                created_at=datetime.now(timezone.utc),
                citations=(",".join(set(citations)) if citations else None),
                tool_calls=tool_calls if tool_calls else None,
                thinking=thinking,
            )
            self.db.add(new_message)
            self.db.commit()
            logger.info(
                f"Saved partial AI message for conversation {conversation_id} (stopped generation)"
            )
            return new_message.id
        except SQLAlchemyError as e:
            logger.exception(
                "Database error in save_partial_ai_message",
                conversation_id=conversation_id,
            )
            self.db.rollback()
            raise ChatHistoryServiceError(
                f"Failed to save partial AI message for conversation {conversation_id}"
            ) from e
        except Exception as e:
            logger.exception(
                "Unexpected error in save_partial_ai_message",
                conversation_id=conversation_id,
            )
            self.db.rollback()
            raise ChatHistoryServiceError(
                f"An unexpected error occurred while saving partial AI message for conversation {conversation_id}"
            ) from e

    def clear_session_history(self, conversation_id: str):
        try:
            self.db.query(Message).filter_by(conversation_id=conversation_id).delete()
            self.db.commit()
            logger.info(f"Cleared session history for conversation: {conversation_id}")
        except SQLAlchemyError as e:
            logger.exception(
                "Database error in clear_session_history",
                conversation_id=conversation_id,
            )
            self.db.rollback()
            raise ChatHistoryServiceError(
                f"Failed to clear session history for conversation {conversation_id}"
            ) from e
        except Exception as e:
            logger.exception(
                "Unexpected error in clear_session_history",
                conversation_id=conversation_id,
            )
            self.db.rollback()
            raise ChatHistoryServiceError(
                f"An unexpected error occurred while clearing session history for conversation {conversation_id}"
            ) from e

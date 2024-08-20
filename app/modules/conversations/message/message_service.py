from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.modules.conversations.message.message_model import Message, MessageType, MessageStatus
from uuid6 import uuid7

class MessageService:
    def __init__(self, db: Session):
        self.db = db 

    def create_message(self, conversation_id: str, content: str, message_type: MessageType, sender_id: Optional[str] = None) -> Message:
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
            status=MessageStatus.ACTIVE
        )

        try:
            self.db.add(new_message)
            self.db.commit()
            self.db.refresh(new_message)
            return new_message
        except IntegrityError as e:
            self.db.rollback()
            # Handle specific SQLAlchemy IntegrityError
            raise RuntimeError("Database integrity error occurred") from e
        except Exception as e:
            self.db.rollback()
            # Handle all other exceptions
            raise RuntimeError("An unexpected error occurred") from e

    def mark_message_inactive(self, message_id: str) -> None:
        try:
            message = self.db.query(Message).filter(Message.id == message_id).one_or_none()
            if message:
                message.status = MessageStatus.INACTIVE
                self.db.commit()
            else:
                raise ValueError("Message not found.")
        except IntegrityError as e:
            self.db.rollback()
            # Handle specific SQLAlchemy IntegrityError
            raise RuntimeError("Database integrity error occurred") from e
        except Exception as e:
            self.db.rollback()
            # Handle all other exceptions
            raise RuntimeError("An unexpected error occurred") from e

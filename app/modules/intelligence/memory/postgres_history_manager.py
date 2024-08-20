from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timezone
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from app.modules.conversations.message.message_model import Message, MessageType
from uuid6 import uuid7

class PostgresChatHistoryManager:
    """Manages chat histories in PostgreSQL for different sessions."""

    def __init__(self, db: Session):
        self.db = db

    def get_session_history(self, user_id: str, conversation_id: str) -> List[BaseMessage]:
        """Retrieve chat history for a given session from the database."""
        messages = self.db.query(Message).filter_by(conversation_id=conversation_id).order_by(Message.created_at).all()

        history = []
        for msg in messages:
            if msg.type == MessageType.HUMAN:
                history.append(HumanMessage(content=msg.content))
            else:
                history.append(AIMessage(content=msg.content))
        return history

    def add_message(self, conversation_id: str, content: str, message_type: MessageType, sender_id: Optional[str] = None):
        """Add a message to the conversation in the database."""
        new_message = Message(
            id=str(uuid7()),
            conversation_id=conversation_id,
            content=content,
            sender_id=sender_id if message_type == MessageType.HUMAN else None,
            type=message_type,
            created_at=datetime.now(timezone.utc)  # Set the timestamp here
        )
        self.db.add(new_message)
        self.db.commit()

    def clear_session_history(self, conversation_id: str):
        """Clear the chat history for a given conversation."""
        self.db.query(Message).filter_by(conversation_id=conversation_id).delete()
        self.db.commit()

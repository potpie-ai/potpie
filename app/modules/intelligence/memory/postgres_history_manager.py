from sqlalchemy.orm import Session
from typing import Dict, List, Optional
from datetime import datetime, timezone
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from app.modules.conversations.message.message_model import Message, MessageType
from uuid6 import uuid7

class PostgresChatHistoryManager:
    def __init__(self, db: Session):
        self.db = db
        self.message_buffer: Dict[str, str] = {}

    def get_session_history(self, user_id: str, conversation_id: str) -> List[BaseMessage]:
        messages = self.db.query(Message).filter_by(conversation_id=conversation_id).order_by(Message.created_at).all()
        history = []
        for msg in messages:
            if msg.type == MessageType.HUMAN:
                history.append(HumanMessage(content=msg.content))
            else:
                history.append(AIMessage(content=msg.content))
        return history

    def add_message_chunk(self, conversation_id: str, content: str, message_type: MessageType, sender_id: Optional[str] = None):
        if conversation_id not in self.message_buffer:
            self.message_buffer[conversation_id] = ""
        self.message_buffer[conversation_id] += content

    def flush_message_buffer(self, conversation_id: str, message_type: MessageType, sender_id: Optional[str] = None):
        if conversation_id in self.message_buffer and self.message_buffer[conversation_id]:
            new_message = Message(
                id=str(uuid7()),
                conversation_id=conversation_id,
                content=self.message_buffer[conversation_id],
                sender_id=sender_id if message_type == MessageType.HUMAN else None,
                type=message_type,
                created_at=datetime.now(timezone.utc)
            )
            self.db.add(new_message)
            self.db.commit()
            self.message_buffer[conversation_id] = ""

    def clear_session_history(self, conversation_id: str):
        self.db.query(Message).filter_by(conversation_id=conversation_id).delete()
        self.db.commit()

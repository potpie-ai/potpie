from app.core.base_service import BaseService
from sqlalchemy.orm import Session
import uuid
from sqlalchemy import func

from app.modules.conversations.message.message_model import Message

class MessageService(BaseService):
    def __init__(self, db: Session):
        super().__init__(db)

    def create_message(self, conversation_id: str, content: str, sender_id: str, message_type: str):
        new_message = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            content=content,
            sender_id=sender_id,
            type=message_type,
            created_at=func.now(),
        )
        self.db.add(new_message)
        self.db.commit()
        return new_message

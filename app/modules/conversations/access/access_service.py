from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError 
from typing import List
from uuid6 import uuid7 
from app.modules.conversations.conversation.conversation_model import Conversation



class ShareChatServiceError(Exception):
    """Base exception class for ShareChatService errors."""


class ShareChatService:
    def __init__(self, db: Session):
        self.db = db

    async def share_chat(self, conversation_id: str, recipient_emails: List[str]) -> str:
        chat = self.db.query(Conversation).filter_by(id=conversation_id).first()
        if not chat:
            raise ShareChatServiceError("Chat not found.")

        try:
            self.db.query(Conversation).filter_by(id=conversation_id).update(
                {Conversation.shared_with_emails: recipient_emails},
                synchronize_session=False  
            )
            self.db.commit() 
        except IntegrityError as e:
            self.db.rollback()
            raise ShareChatServiceError("Failed to update shared chat due to a database integrity error.") from e

        return conversation_id


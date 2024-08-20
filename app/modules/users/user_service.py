from sqlalchemy.orm import Session
from typing import List
from app.modules.conversations.conversation.conversation_model import Conversation

class UserService:

    def __init__(self, db: Session):
        self.db = db

    def get_conversations_for_user(self, user_id: str, start: int, limit: int) -> List[Conversation]:
        conversations = (
            self.db.query(Conversation)
            .filter(Conversation.user_id == user_id)
            .offset(start)
            .limit(limit)
            .all()
        )
        return conversations
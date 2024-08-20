import logging
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import List
from app.modules.conversations.conversation.conversation_model import Conversation

logger = logging.getLogger(__name__)

class UserServiceError(Exception):
    """Base exception class for UserService errors."""

class UserService:
    def __init__(self, db: Session):
        self.db = db

    def get_conversations_for_user(self, user_id: str, start: int, limit: int) -> List[Conversation]:
        try:
            conversations = (
                self.db.query(Conversation)
                .filter(Conversation.user_id == user_id)
                .offset(start)
                .limit(limit)
                .all()
            )
            logger.info(f"Retrieved {len(conversations)} conversations for user {user_id}")
            return conversations
        except SQLAlchemyError as e:
            logger.error(f"Database error in get_conversations_for_user for user {user_id}: {e}", exc_info=True)
            raise UserServiceError(f"Failed to retrieve conversations for user {user_id}") from e
        except Exception as e:
            logger.error(f"Unexpected error in get_conversations_for_user for user {user_id}: {e}", exc_info=True)
            raise UserServiceError(f"An unexpected error occurred while retrieving conversations for user {user_id}") from e
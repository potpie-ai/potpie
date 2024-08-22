import logging
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import List
from app.modules.conversations.conversation.conversation_model import Conversation
from datetime import datetime

import logging

from app.core.database import SessionManager
from app.modules.users.user_schema import CreateUser
from app.modules.users.user_model import User

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

    
    def update_last_login(self, uid: str):
        logging.info(f"Updating last login time for user with UID: {uid}")
        message: str = ""
        error: bool = False
        try:
            with SessionManager() as db:
                user = db.query(User).filter(User.uid == uid).first()
                if user:
                    user.last_login_at = datetime.utcnow()
                    db.commit()
                    db.refresh(user)
                    error = False
                    message = f"Updated last login time for user with ID: {user.uid}"
                else:
                    error = True
                    message = "User not found"
        except Exception as e:
            logging.error(f"Error updating last login time: {e}")
            message = "Error updating last login time"
            error = True

        return message, error

    def create_user(self, user_details: CreateUser):
        logging.info(
            f"Creating user with email: {user_details.email} | display_name:"
            f" {user_details.display_name}"
        )
        new_user = User(
            uid=user_details.uid,
            email=user_details.email,
            display_name=user_details.display_name,
            email_verified=user_details.email_verified,
            created_at=user_details.created_at,
            last_login_at=user_details.last_login_at,
            provider_info=user_details.provider_info,
            provider_username=user_details.provider_username,
        )
        message: str = ""
        error: bool = False
        try:
            with SessionManager() as db:
                db.add(new_user)
                db.commit()
                db.refresh(new_user)
                error = False
                message = f"User created with ID: {new_user.uid}"
                uid = new_user.uid

        except Exception as e:
            logging.error(f"Error creating user: {e}")
            message = "error creating user"
            error = True
            uid = ""

        return uid, message, error

    def get_user_by_uid(self, uid: str):
        try:
            with SessionManager() as db:
                user = db.query(User).filter(User.uid == uid).first()
                return user
        except Exception as e:
            logging.error(f"Error fetching user: {e}")
            return None



import asyncio
import os
from datetime import datetime, timezone
from typing import List

from firebase_admin import auth
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from app.modules.users.user_model import User
from app.modules.users.user_schema import CreateUser, UserProfileResponse
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class UserServiceError(Exception):
    """Base exception class for UserService errors."""


class UserService:
    def __init__(self, db: Session):
        self.db = db

    def update_last_login(self, uid: str, oauth_token: str):
        logger.info(f"Updating last login time for user with UID: {uid}")
        message: str = ""
        error: bool = False
        try:
            user = self.db.query(User).filter(User.uid == uid).first()
            if user:
                user.last_login_at = datetime.now(timezone.utc)

                # Safely update provider_info with OAuth token
                if user.provider_info is None:
                    user.provider_info = {}
                provider_info = (
                    user.provider_info.copy()
                    if isinstance(user.provider_info, dict)
                    else {}
                )
                provider_info["access_token"] = oauth_token
                user.provider_info = provider_info

                self.db.commit()
                self.db.refresh(user)
                error = False
                message = f"Updated last login time for user with ID: {user.uid}"
            else:
                error = True
                message = "User not found"
        except Exception as e:
            logger.error(f"Error updating last login time: {e}")
            message = "Error updating last login time"
            error = True

        return message, error

    def create_user(self, user_details: CreateUser):
        logger.info(
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
            self.db.add(new_user)
            self.db.commit()
            self.db.refresh(new_user)
            error = False
            message = f"User created with ID: {new_user.uid}"
            uid = new_user.uid

        except Exception as e:
            logger.error(f"Error creating user: {e}")
            message = "error creating user"
            error = True
            uid = ""

        return uid, message, error

    def setup_dummy_user(self):
        defaultUserId = os.getenv("defaultUsername")
        user = self.get_user_by_uid(defaultUserId)
        if user:
            print("Dummy user already exists")
            return
        else:
            user = CreateUser(
                uid=defaultUserId,
                email="defaultuser@potpie.ai",
                display_name="Dummy User",
                email_verified=True,
                created_at=datetime.now(timezone.utc),
                last_login_at=datetime.now(timezone.utc),
                provider_info={"access_token": "dummy_token"},
                provider_username="self",
            )
            uid, message, error = self.create_user(user)
            logger.info(f"Created dummy user with uid: {uid}")

    def get_user_by_uid(self, uid: str):
        try:
            user = self.db.query(User).filter(User.uid == uid).first()
            return user
        except Exception as e:
            logger.error(f"Error fetching user: {e}")
            return None

    def get_user_id_by_email(self, email: str) -> str:
        logger.info(f"DEBUG: get_user_id_by_email called for email: {email}")
        try:
            user = self.db.query(User).filter(User.email == email).first()
            if user:
                logger.info(
                    f"DEBUG: Found user with uid: {user.uid} for email: {email}"
                )
                return user.uid
            else:
                logger.warning(f"DEBUG: No user found for email: {email}")
                return None
        except Exception as e:
            logger.error(f"DEBUG: Error fetching user ID by email {email}: {e}")
            return None

    async def get_user_by_email(self, email: str) -> User:
        """
        Get a user by their email address.
        Returns the full User object or None if not found.
        """
        try:
            # Use an optimized query that only fetches the user once
            user = self.db.query(User).filter(User.email == email).first()
            return user
        except SQLAlchemyError as e:
            logger.error(f"Database error fetching user by email {email}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching user by email {email}: {e}")
            return None

    def get_user_ids_by_emails(self, emails: List[str]) -> List[str]:
        logger.info(f"DEBUG: get_user_ids_by_emails called for emails: {emails}")
        try:
            users = self.db.query(User).filter(User.email.in_(emails)).all()
            if users:
                user_ids = [user.uid for user in users]
                logger.info(f"DEBUG: Found user IDs: {user_ids} for emails: {emails}")
                return user_ids
            else:
                logger.warning(f"DEBUG: No users found for emails: {emails}")
                return None
        except Exception as e:
            logger.error(f"DEBUG: Error fetching user ID by emails {emails}: {e}")
            return None

    async def get_user_profile_pic(self, uid: str) -> UserProfileResponse:
        try:
            user_record = await asyncio.to_thread(auth.get_user, uid)
            profile_pic_url = user_record.photo_url
            return {"user_id": user_record.uid, "profile_pic_url": profile_pic_url}
        except Exception as e:
            logger.error(f"Error retrieving user profile picture: {e}")
            return None

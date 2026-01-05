"""User resource for PotpieRuntime library."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from potpie.exceptions import UserError, UserNotFoundError
from potpie.resources.base import BaseResource
from potpie.types.user import UserInfo

if TYPE_CHECKING:
    from potpie.config import RuntimeConfig
    from potpie.core.database import DatabaseManager
    from potpie.core.neo4j import Neo4jManager

logger = logging.getLogger(__name__)


class UserResource(BaseResource):
    """Manage users.

    Wraps the existing UserService with a clean library interface.
    All operations take user identifiers as parameters - no stored user context.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        db_manager: DatabaseManager,
        neo4j_manager: Neo4jManager,
    ):
        super().__init__(config, db_manager, neo4j_manager)

    def _get_service(self):
        """Get a UserService instance with a fresh session."""
        from app.modules.users.user_service import UserService

        session = self._db_manager.get_session()
        return UserService(session), session

    async def ensure_user(
        self,
        user_id: str,
        email: str,
        *,
        display_name: Optional[str] = None,
    ) -> UserInfo:
        """Ensure user exists, create if necessary.

        Args:
            user_id: User identifier
            email: User email
            display_name: Optional display name

        Returns:
            UserInfo for the user

        Raises:
            UserError: If operation fails
        """
        from app.modules.users.user_schema import CreateUser

        service, session = self._get_service()
        try:
            existing = service.get_user_by_uid(user_id)
            if existing:
                return UserInfo.from_model(existing)

            user_data = CreateUser(
                uid=user_id,
                email=email,
                display_name=display_name or email.split("@")[0],
                email_verified=True,
                created_at=datetime.utcnow(),
                last_login_at=datetime.utcnow(),
                provider_info={},
                provider_username="library",
            )

            uid, message, error = service.create_user(user_data)
            if error:
                raise UserError(f"Failed to create user: {message}")

            user = service.get_user_by_uid(uid)
            if user is None:
                raise UserError("Failed to retrieve created user")

            logger.info(f"Created user: {uid}")
            return UserInfo.from_model(user)

        except UserError:
            raise
        except Exception as e:
            session.rollback()
            raise UserError(f"Failed to ensure user: {e}") from e
        finally:
            session.close()

    async def get(self, user_id: str) -> Optional[UserInfo]:
        """Get user by ID.

        Args:
            user_id: User identifier

        Returns:
            UserInfo if found, None otherwise
        """
        service, session = self._get_service()
        try:
            user = service.get_user_by_uid(user_id)
            if user is None:
                return None
            return UserInfo.from_model(user)

        except Exception as e:
            raise UserError(f"Failed to get user: {e}") from e
        finally:
            session.close()

    async def get_by_email(self, email: str) -> Optional[UserInfo]:
        """Get user by email.

        Args:
            email: User email

        Returns:
            UserInfo if found, None otherwise
        """
        service, session = self._get_service()
        try:
            user = await service.get_user_by_email(email)
            if user is None:
                return None
            return UserInfo.from_model(user)

        except Exception as e:
            raise UserError(f"Failed to get user by email: {e}") from e
        finally:
            session.close()

    async def update_last_login(self, user_id: str) -> None:
        """Update last login timestamp for a user.

        Args:
            user_id: User identifier

        Raises:
            UserNotFoundError: If user not found
            UserError: If update fails
        """
        service, session = self._get_service()
        try:
            message, error = service.update_last_login(user_id, "")
            if error:
                if "not found" in message.lower():
                    raise UserNotFoundError(f"User not found: {user_id}")
                raise UserError(f"Failed to update last login: {message}")

        except (UserNotFoundError, UserError):
            raise
        except Exception as e:
            session.rollback()
            raise UserError(f"Failed to update last login: {e}") from e
        finally:
            session.close()

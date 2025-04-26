import hashlib
import os
import secrets
from typing import Optional

from fastapi import HTTPException
from google.cloud import secretmanager
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.modules.users.user_model import User
from app.modules.users.user_preferences_model import UserPreferences
import logging

logger = logging.getLogger(__name__)


class APIKeyService:
    SECRET_PREFIX = "sk-"
    KEY_LENGTH = 32

    @staticmethod
    def get_client_and_project():
        """Get Secret Manager client and project ID based on environment."""
        is_dev_mode = os.getenv("isDevelopmentMode", "enabled") == "enabled"
        if is_dev_mode:
            return None, None

        project_id = os.environ.get("GCP_PROJECT")
        if not project_id:
            raise HTTPException(
                status_code=500, detail="GCP_PROJECT environment variable is not set"
            )

        try:
            client = secretmanager.SecretManagerServiceClient()
            return client, project_id
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize Secret Manager client: {str(e)}",
            )

    @staticmethod
    def generate_api_key() -> str:
        """Generate a new API key with prefix."""
        random_key = secrets.token_hex(APIKeyService.KEY_LENGTH)
        return f"{APIKeyService.SECRET_PREFIX}{random_key}"

    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash the API key for storage and comparison."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    @staticmethod
    async def create_api_key(user_id: str, db: Session) -> str:
        """Create a new API key for a user."""
        api_key = APIKeyService.generate_api_key()
        hashed_key = APIKeyService.hash_api_key(api_key)

        # Store hashed key in user preferences
        user_pref = (
            db.query(UserPreferences).filter(UserPreferences.user_id == user_id).first()
        )
        if not user_pref:
            user_pref = UserPreferences(user_id=user_id, preferences={})
            db.add(user_pref)
        if "api_key_hash" not in user_pref.preferences:
            pref = user_pref.preferences.copy()
            pref["api_key_hash"] = hashed_key
            user_pref.preferences = pref
        db.commit()
        db.refresh(user_pref)
        # Store actual key in Secret Manager
        if os.getenv("isDevelopmentMode") != "enabled":
            client, project_id = APIKeyService.get_client_and_project()
            secret_id = f"user-api-key-{user_id}"
            parent = f"projects/{project_id}"

            try:
                # Create secret
                secret = {"replication": {"automatic": {}}}
                response = client.create_secret(
                    request={"parent": parent, "secret_id": secret_id, "secret": secret}
                )

                # Add secret version
                version = {"payload": {"data": api_key.encode("UTF-8")}}
                client.add_secret_version(
                    request={"parent": response.name, "payload": version["payload"]}
                )
            except Exception as e:
                # Rollback database changes if secret manager fails
                if "api_key_hash" in user_pref.preferences:
                    del user_pref.preferences["api_key_hash"]
                    db.commit()
                raise HTTPException(
                    status_code=500, detail=f"Failed to store API key: {str(e)}"
                )

        return api_key

    @staticmethod
    async def validate_api_key(api_key: str, db: Session) -> Optional[dict]:
        """Validate an API key and return user info if valid."""
        try:
            # Check if API key follows the correct syntax and prefix
            if not api_key.startswith(APIKeyService.SECRET_PREFIX):
                logger.error(
                    f"Invalid API key format: missing required prefix '{APIKeyService.SECRET_PREFIX}'"
                )
                return None

            hashed_key = APIKeyService.hash_api_key(api_key)
            # Find user with matching hashed key
            result = (
                db.query(UserPreferences, User.email)
                .join(User, UserPreferences.user_id == User.uid)
                .filter(text("preferences->>'api_key_hash' = :hashed_key"))
                .params(hashed_key=hashed_key)
                .first()
            )

            # No match found for Hashed API key
            if not result:
                logger.error("No user found with the provided API key hash")
                return None

            user_pref, email = result
            return {
                "user_id": user_pref.user_id,
                "email": email,
                "auth_type": "api_key",
            }
        except Exception as e:
            logger.error(f"Error validating API key: {str(e)}")
            return None

    @staticmethod
    async def revoke_api_key(user_id: str, db: Session) -> bool:
        """Revoke a user's API key."""
        user_pref = (
            db.query(UserPreferences).filter(UserPreferences.user_id == user_id).first()
        )
        if not user_pref:
            return False

        if "api_key_hash" in user_pref.preferences:
            # Create a new dictionary without the api_key_hash
            updated_preferences = user_pref.preferences.copy()
            del updated_preferences["api_key_hash"]
            user_pref.preferences = updated_preferences
            db.commit()

        # Delete from Secret Manager if not in dev mode
        if os.getenv("isDevelopmentMode") != "enabled":
            client, project_id = APIKeyService.get_client_and_project()
            secret_id = f"user-api-key-{user_id}"
            name = f"projects/{project_id}/secrets/{secret_id}"

            try:
                client.delete_secret(request={"name": name})
            except Exception:
                pass  # Ignore if secret doesn't exist

        return True

    @staticmethod
    async def get_api_key(user_id: str, db: Session) -> Optional[str]:
        """Retrieve the existing API key for a user."""
        user_pref = (
            db.query(UserPreferences).filter(UserPreferences.user_id == user_id).first()
        )
        if not user_pref or "api_key_hash" not in user_pref.preferences:
            return None

        if os.getenv("isDevelopmentMode") == "enabled":
            return None  # In dev mode, we can't retrieve the actual key for security

        client, project_id = APIKeyService.get_client_and_project()
        secret_id = f"user-api-key-{user_id}"
        name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"

        try:
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to retrieve API key: {str(e)}"
            )

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
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

# Preference key for API key value (fallback when GCP Secret Manager unavailable)
API_KEY_VALUE_PREF = "api_key_value"


class APIKeyService:
    SECRET_PREFIX = "sk-"
    KEY_LENGTH = 32

    @staticmethod
    def _is_gcp_secret_manager_enabled() -> bool:
        """Check if GCP Secret Manager should be used (can be disabled for dev/staging without GCP permissions)."""
        return os.environ.get("GCP_SECRET_MANAGER_ENABLED", "true").lower() not in (
            "false",
            "0",
            "no",
            "disabled",
        )

    @staticmethod
    def get_client_and_project():
        """Get Secret Manager client and project ID based on environment."""
        is_dev_mode = os.getenv("isDevelopmentMode", "enabled") == "enabled"
        if is_dev_mode:
            return None, None

        if not APIKeyService._is_gcp_secret_manager_enabled():
            return None, None

        project_id = os.environ.get("GCP_PROJECT")
        if not project_id:
            return None, None

        try:
            client = secretmanager.SecretManagerServiceClient()
            return client, project_id
        except Exception as e:
            logger.warning(f"GCP Secret Manager not available: {e}")
            return None, None

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
    def _store_api_key_in_db(user_pref: UserPreferences, api_key: str, db: Session):
        """Store API key in user preferences (fallback when GCP unavailable)."""
        pref = user_pref.preferences.copy() if user_pref.preferences else {}
        pref[API_KEY_VALUE_PREF] = api_key
        user_pref.preferences = pref
        db.commit()
        logger.info("Stored API key in database (GCP Secret Manager fallback)")

    @staticmethod
    async def create_api_key(user_id: str, db: Session) -> str:
        """Create a new API key for a user. Uses GCP Secret Manager when available, otherwise DB storage."""
        api_key = APIKeyService.generate_api_key()
        hashed_key = APIKeyService.hash_api_key(api_key)

        # Store hashed key in user preferences (for validation)
        user_pref = (
            db.query(UserPreferences).filter(UserPreferences.user_id == user_id).first()
        )
        if not user_pref:
            user_pref = UserPreferences(user_id=user_id, preferences={})
            db.add(user_pref)
        if "api_key_hash" not in user_pref.preferences:
            pref = user_pref.preferences.copy() if user_pref.preferences else {}
            pref["api_key_hash"] = hashed_key
            user_pref.preferences = pref
        db.commit()
        db.refresh(user_pref)

        # Store actual key: try GCP first, fall back to DB when GCP unavailable or fails
        if os.getenv("isDevelopmentMode") != "enabled":
            client, project_id = APIKeyService.get_client_and_project()

            if client and project_id:
                secret_id = f"user-api-key-{user_id}"
                parent = f"projects/{project_id}"

                try:
                    secret = {"replication": {"automatic": {}}}
                    response = client.create_secret(
                        request={
                            "parent": parent,
                            "secret_id": secret_id,
                            "secret": secret,
                        }
                    )
                    version = {"payload": {"data": api_key.encode("UTF-8")}}
                    client.add_secret_version(
                        request={
                            "parent": response.name,
                            "payload": version["payload"],
                        }
                    )
                    logger.info("Stored API key in GCP Secret Manager")
                except Exception as e:
                    logger.warning(
                        f"GCP Secret Manager failed ({e}), falling back to database storage"
                    )
                    APIKeyService._store_api_key_in_db(user_pref, api_key, db)
            else:
                # GCP disabled or unavailable, use database fallback
                APIKeyService._store_api_key_in_db(user_pref, api_key, db)

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
        """Revoke a user's API key. Removes from both GCP and database fallback."""
        user_pref = (
            db.query(UserPreferences).filter(UserPreferences.user_id == user_id).first()
        )
        if not user_pref:
            return False

        had_key = False
        if user_pref.preferences:
            updated_preferences = user_pref.preferences.copy()
            if "api_key_hash" in updated_preferences:
                del updated_preferences["api_key_hash"]
                had_key = True
            if API_KEY_VALUE_PREF in updated_preferences:
                del updated_preferences[API_KEY_VALUE_PREF]
            user_pref.preferences = updated_preferences
            db.commit()

        # Delete from Secret Manager if available
        if os.getenv("isDevelopmentMode") != "enabled":
            client, project_id = APIKeyService.get_client_and_project()
            if client and project_id:
                secret_id = f"user-api-key-{user_id}"
                name = f"projects/{project_id}/secrets/{secret_id}"
                try:
                    client.delete_secret(request={"name": name})
                except Exception:
                    pass  # Ignore if secret doesn't exist

        return had_key

    @staticmethod
    async def get_api_key(user_id: str, db: Session) -> Optional[str]:
        """Retrieve the existing API key for a user. Tries GCP first, then database fallback."""
        user_pref = (
            db.query(UserPreferences).filter(UserPreferences.user_id == user_id).first()
        )
        if not user_pref or "api_key_hash" not in user_pref.preferences:
            return None

        if os.getenv("isDevelopmentMode") == "enabled":
            return None  # In dev mode, we can't retrieve the actual key for security

        # Try GCP Secret Manager first
        client, project_id = APIKeyService.get_client_and_project()
        if client and project_id:
            secret_id = f"user-api-key-{user_id}"
            name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
            try:
                response = client.access_secret_version(request={"name": name})
                return response.payload.data.decode("UTF-8")
            except Exception as e:
                logger.warning(
                    f"Failed to retrieve API key from GCP ({e}), trying database fallback"
                )

        # Fallback: key stored in user preferences
        if user_pref.preferences and API_KEY_VALUE_PREF in user_pref.preferences:
            return user_pref.preferences[API_KEY_VALUE_PREF]

        raise HTTPException(
            status_code=500,
            detail="API key not found in Secret Manager or database",
        )

import os
import base64
from typing import Literal

from fastapi import Depends, HTTPException
from google.cloud import secretmanager
from sqlalchemy.orm import Session
from cryptography.fernet import Fernet, InvalidToken

from app.core.database import get_db
from app.modules.auth.api_key_service import APIKeyService
from app.modules.auth.auth_service import AuthService
from app.modules.key_management.secrets_schema import (
    APIKeyResponse,
    CreateSecretRequest,
    UpdateSecretRequest,
)
from app.modules.users.user_preferences_model import UserPreferences
from app.modules.utils.APIRouter import APIRouter
from app.modules.utils.posthog_helper import PostHogClient

router = APIRouter()


class SecretManager:
    @staticmethod
    def _get_fernet() -> Fernet:
        secret_key = os.environ.get("SECRET_ENCRYPTION_KEY")
        if not secret_key:
            raise HTTPException(
                status_code=500,
                detail="SECRET_ENCRYPTION_KEY environment variable is not set",
            )
        # Ensure the key is in bytes. It must be a URL-safe base64-encoded 32-byte key.
        try:
            # This will raise if the key is invalid.
            return Fernet(secret_key.encode("utf-8"))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid SECRET_ENCRYPTION_KEY: {str(e)}",
            )

    @staticmethod
    def encrypt_api_key(api_key: str) -> str:
        f = SecretManager._get_fernet()
        encrypted = f.encrypt(api_key.encode("utf-8"))
        return encrypted.decode("utf-8")

    @staticmethod
    def decrypt_api_key(encrypted_key: str) -> str:
        f = SecretManager._get_fernet()
        try:
            decrypted = f.decrypt(encrypted_key.encode("utf-8"))
            return decrypted.decode("utf-8")
        except InvalidToken:
            raise HTTPException(
                status_code=500, detail="Failed to decrypt API key. Invalid token."
            )

    @staticmethod
    def get_client_and_project():
        """Return the Google Secret Manager client and project ID only if GCP_PROJECT is set."""
        project_id = os.environ.get("GCP_PROJECT")
        if not project_id:
            return None, None
        try:
            client = secretmanager.SecretManagerServiceClient()
            return client, project_id
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize Secret Manager client: {str(e)}",
            )

    @staticmethod
    def get_secret_id(
        provider: Literal[
            "openai", "anthropic", "deepseek", "meta-llama", "mistralai", "gemini", "openrouter"
        ],
        customer_id: str,
    ):
        if os.environ.get("isDevelopmentMode") == "enabled":
            return None
        secret_id = f"{provider}-api-key-{customer_id}"
        return secret_id

    @staticmethod
    async def check_secret_exists_for_user(provider: str, user_id: str, db: Session) -> bool:
        """Check if a secret exists for a given provider and user, without raising exceptions."""
        client, project_id = SecretManager.get_client_and_project()
        if client and project_id:
            secret_id = SecretManager.get_secret_id(provider, user_id)
            name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
            try:
                client.access_secret_version(request={"name": name}) # Try to access, will raise exception if not found
                return True # Secret exists in GCP
            except Exception:
                pass # Secret not found in GCP, fallback to checking UserPreferences
        # Fallback: check UserPreferences
        user_pref = db.query(UserPreferences).filter(UserPreferences.user_id == user_id).first()
        if user_pref and user_pref.preferences.get(f"api_key_{provider}"):
            return True # Secret exists in UserPreferences
        return False # Secret not found anywhere


    @router.post("/secrets")
    def create_secret(
        request: CreateSecretRequest,
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        customer_id = user["user_id"]
        client, project_id = SecretManager.get_client_and_project()

        # Update user preferences with provider info and models regardless of GCP or not.
        user_pref = (
            db.query(UserPreferences)
            .filter(UserPreferences.user_id == customer_id)
            .first()
        )
        if not user_pref:
            user_pref = UserPreferences(user_id=customer_id, preferences={})
            db.add(user_pref)
        user_pref.preferences["provider"] = request.provider
        if request.low_reasoning_model:
            user_pref.preferences["low_reasoning_model"] = request.low_reasoning_model
        if request.high_reasoning_model:
            user_pref.preferences["high_reasoning_model"] = request.high_reasoning_model

        if client and project_id:
            # Use Google Secret Manager for API Key
            api_key = request.api_key
            secret_id = SecretManager.get_secret_id(request.provider, customer_id)
            parent = f"projects/{project_id}"
            secret = {"replication": {"automatic": {}}}
            response = client.create_secret(
                request={"parent": parent, "secret_id": secret_id, "secret": secret}
            )
            version = {"payload": {"data": api_key.encode("UTF-8")}}
            client.add_secret_version(
                request={"parent": response.name, "payload": version["payload"]}
            )
        else:
            # Fallback: store encrypted API key and model configs in UserPreferences
            encrypted_key = SecretManager.encrypt_api_key(request.api_key)
            user_pref.preferences[f"api_key_{request.provider}"] = encrypted_key

        db.commit()
        PostHogClient().send_event(
            customer_id,
            "secret_creation_event",
            {"provider": request.provider, "key_added": "true"},
        )
        return {"message": "Secret created successfully"}

    @router.get("/secrets/{provider}")
    def get_secret_for_provider(
        provider: Literal[
            "openai", "anthropic", "deepseek", "meta-llama", "mistralai", "gemini", "openrouter", "all"
        ],
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        customer_id = user["user_id"]
        user_pref = (
            db.query(UserPreferences)
            .filter(UserPreferences.user_id == customer_id)
            .first()
        )
        if not user_pref:
            raise HTTPException(
                status_code=404, detail="Secret not found for this provider"
            )

        if provider == "all":
            provider = user_pref.preferences.get("provider")
            if not provider:
                raise HTTPException(
                    status_code=404, detail="No provider stored for this user"
                )

        return SecretManager.get_secret(provider, customer_id, db)

    @staticmethod
    def get_secret(
        provider: Literal[
            "openai", "anthropic", "deepseek", "meta-llama", "mistralai", "gemini", "openrouter"
        ],
        customer_id: str,
        db: Session = None,
    ):
        client, project_id = SecretManager.get_client_and_project()
        if client and project_id:
            secret_id = SecretManager.get_secret_id(provider, customer_id)
            name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
            try:
                response = client.access_secret_version(request={"name": name})
                api_key = response.payload.data.decode("UTF-8")
                return {"api_key": api_key, "provider": provider}
            except Exception as e:
                raise HTTPException(
                    status_code=404,
                    detail=f"Secret not found in GCP Secret Manager: {str(e)}",
                )
        else:
            # Fallback: get from UserPreferences
            if not db:
                raise HTTPException(
                    status_code=500,
                    detail="Database session required for fallback secret retrieval",
                )
            user_pref = (
                db.query(UserPreferences)
                .filter(UserPreferences.user_id == customer_id)
                .first()
            )
            encrypted_key = user_pref.preferences.get(f"api_key_{provider}")
            if not encrypted_key:
                raise HTTPException(
                    status_code=404, detail="Secret not found in UserPreferences"
                )
            api_key = SecretManager.decrypt_api_key(encrypted_key)
            return {"api_key": api_key, "provider": provider}

    @router.put("/secrets/")
    def update_secret(
        request: UpdateSecretRequest,
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        customer_id = user["user_id"]
        api_key = request.api_key
        client, project_id = SecretManager.get_client_and_project()
        if client and project_id:
            secret_id = SecretManager.get_secret_id(request.provider, customer_id)
            parent = f"projects/{project_id}/secrets/{secret_id}"
            version = {"payload": {"data": api_key.encode("UTF-8")}}
            client.add_secret_version(
                request={"parent": parent, "payload": version["payload"]}
            )
        else:
            # Fallback: update UserPreferences store
            user_pref = (
                db.query(UserPreferences)
                .filter(UserPreferences.user_id == customer_id)
                .first()
            )
            if not user_pref:
                user_pref = UserPreferences(user_id=customer_id, preferences={})
                db.add(user_pref)
            encrypted_key = SecretManager.encrypt_api_key(api_key)
            user_pref.preferences[f"api_key_{request.provider}"] = encrypted_key

        # Update provider and models in user preferences as well.
        user_pref = (
            db.query(UserPreferences)
            .filter(UserPreferences.user_id == customer_id)
            .first()
        )
        user_pref.preferences["provider"] = request.provider
        if request.low_reasoning_model:
            user_pref.preferences["low_reasoning_model"] = request.low_reasoning_model
        if request.high_reasoning_model:
            user_pref.preferences["high_reasoning_model"] = request.high_reasoning_model

        db.commit()
        return {"message": "Secret updated successfully"}

    @router.delete("/secrets/{provider}")
    def delete_secret(
        provider: Literal[
            "openai", "anthropic", "deepseek", "meta-llama", "mistralai", "gemini", "openrouter", "all"
        ],
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        from app.modules.intelligence.provider.provider_service import PLATFORM_PROVIDERS
        customer_id = user["user_id"]
        if provider == "all":
            provider_list = PLATFORM_PROVIDERS # Use PLATFORM_PROVIDERS list
            deletion_results = []
            for prov in provider_list:
                client, project_id = SecretManager.get_client_and_project()
                if client and project_id:
                    secret_id = SecretManager.get_secret_id(prov, customer_id)
                    try:
                        name = f"projects/{project_id}/secrets/{secret_id}"
                        client.delete_secret(request={"name": name})
                        deletion_results.append(f"Successfully deleted {prov} secret")
                        PostHogClient().send_event(
                            customer_id,
                            "secret_deletion_event",
                            {"provider": prov, "key_removed": "true"},
                        )
                    except Exception as e:
                        deletion_results.append(
                            f"Failed to delete {prov} secret: {str(e)}"
                        )
                else:
                    # Fallback deletion: remove from UserPreferences
                    user_pref = (
                        db.query(UserPreferences)
                        .filter(UserPreferences.user_id == customer_id)
                        .first()
                    )
                    if user_pref and f"api_key_{prov}" in user_pref.preferences:
                        del user_pref.preferences[f"api_key_{prov}"]
                        db.commit()
                        deletion_results.append(f"Deleted {prov} secret from DB")
                        PostHogClient().send_event(
                            customer_id,
                            "secret_deletion_event",
                            {"provider": prov, "key_removed": "true"},
                        )
            # Remove provider and model configs from user preferences
            user_pref = (
                db.query(UserPreferences)
                .filter(UserPreferences.user_id == customer_id)
                .first()
            )
            if user_pref and "provider" in user_pref.preferences:
                del user_pref.preferences["provider"]
            if user_pref and "low_reasoning_model" in user_pref.preferences:
                del user_pref.preferences["low_reasoning_model"]
            if user_pref and "high_reasoning_model" in user_pref.preferences:
                del user_pref.preferences["high_reasoning_model"]
            db.commit()
            return {
                "message": "All secrets deletion completed",
                "details": deletion_results,
            }

        client, project_id = SecretManager.get_client_and_project()
        if client and project_id:
            secret_id = SecretManager.get_secret_id(provider, customer_id)
            name = f"projects/{project_id}/secrets/{secret_id}"
            try:
                client.delete_secret(request={"name": name})
                user_pref = (
                    db.query(UserPreferences)
                    .filter(UserPreferences.user_id == customer_id)
                    .first()
                )
                if user_pref and "provider" in user_pref.preferences:
                    del user_pref.preferences["provider"]
                if user_pref and "low_reasoning_model" in user_pref.preferences:
                    del user_pref.preferences["low_reasoning_model"]
                if user_pref and "high_reasoning_model" in user_pref.preferences:
                    del user_pref.preferences["high_reasoning_model"]
                db.commit()
                PostHogClient().send_event(
                    customer_id,
                    "secret_deletion_event",
                    {"provider": provider, "key_removed": "true"},
                )
                return {"message": "Secret deleted successfully"}
            except Exception as e:
                raise HTTPException(
                    status_code=404, detail=f"Secret not found: {str(e)}"
                )
        else:
            # Fallback: delete from UserPreferences
            user_pref = (
                db.query(UserPreferences)
                .filter(UserPreferences.user_id == customer_id)
                .first()
            )
            if user_pref and f"api_key_{provider}" in user_pref.preferences:
                del user_pref.preferences[f"api_key_{provider}"]
                db.commit()
                PostHogClient().send_event(
                    customer_id,
                    "secret_deletion_event",
                    {"provider": provider, "key_removed": "true"},
                )
                return {"message": "Secret deleted successfully from DB"}
            else:
                raise HTTPException(
                    status_code=404, detail="Secret not found in fallback storage"
                )

    @router.post("/api-keys", response_model=APIKeyResponse)
    async def create_api_key(
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        try:
            api_key = await APIKeyService.create_api_key(user["user_id"], db)
            PostHogClient().send_event(
                user["user_id"], "api_key_creation", {"success": True}
            )
            return {"api_key": api_key}
        except Exception as e:
            PostHogClient().send_event(
                user["user_id"], "api_key_creation", {"success": False, "error": str(e)}
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to create API key: {str(e)}"
            )

    @router.delete("/api-keys")
    async def revoke_api_key(
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        success = await APIKeyService.revoke_api_key(user["user_id"], db)
        if not success:
            raise HTTPException(
                status_code=404, detail="No API key found for this user"
            )

        PostHogClient().send_event(
            user["user_id"], "api_key_revocation", {"success": True}
        )
        return {"message": "API key revoked successfully"}

    @router.get("/api-keys", response_model=APIKeyResponse)
    async def get_api_key(
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        try:
            api_key = await APIKeyService.get_api_key(user["user_id"], db)
            if api_key is None:
                raise HTTPException(
                    status_code=404, detail="No API key found for this user"
                )
            return {"api_key": api_key}
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to retrieve API key: {str(e)}"
            )
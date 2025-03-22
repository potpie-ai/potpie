import os
import asyncio
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
    BaseSecret,
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
            "openai",
            "anthropic",
            "deepseek",
            "meta-llama",
            "gemini",
            "openrouter",
        ],
        customer_id: str,
    ):
        if os.environ.get("isDevelopmentMode") == "enabled":
            return None
        secret_id = f"{provider}-api-key-{customer_id}"
        return secret_id

    @staticmethod
    async def check_secret_exists_for_user(
        provider: str, user_id: str, db: Session
    ) -> bool:
        """Check if a secret exists for a given provider and user, without raising exceptions."""
        client, project_id = SecretManager.get_client_and_project()
        if client and project_id:
            secret_id = SecretManager.get_secret_id(provider, user_id)
            name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
            try:
                client.access_secret_version(
                    request={"name": name}
                )  # Try to access, will raise exception if not found
                return True  # Secret exists in GCP
            except Exception:
                pass  # Secret not found in GCP, fallback to checking UserPreferences
        # Fallback: check UserPreferences
        user_pref = (
            db.query(UserPreferences).filter(UserPreferences.user_id == user_id).first()
        )
        if user_pref and user_pref.preferences.get(f"api_key_{provider}"):
            return True  # Secret exists in UserPreferences
        return False  # Secret not found anywhere

    @staticmethod
    def _process_config(
        config: BaseSecret,
        config_type: Literal["chat", "inference"],
        customer_id: str,
        client,
        project_id,
        preferences: dict,
        updated_providers: list,
    ) -> None:
        """Helper method to process chat or inference configuration."""
        if not config:
            return

        # Update preferences
        preferences[f"{config_type}_model"] = config.model
        provider = config.model.split("/")[0]
        updated_providers.append(provider)

        if client and project_id:
            # Store/Update in Google Secret Manager
            secret_id = SecretManager.get_secret_id(provider, customer_id)
            parent = f"projects/{project_id}/secrets/{secret_id}"
            version = {"payload": {"data": config.api_key.encode("UTF-8")}}

            try:
                # Try to update existing secret
                client.add_secret_version(
                    request={"parent": parent, "payload": version["payload"]}
                )
            except Exception as e:
                # The secret might not exist yet, try creating it
                if "not found" in str(e).lower() or "404" in str(e) or "409" in str(e):
                    try:
                        secret = {"replication": {"automatic": {}}}
                        response = client.create_secret(
                            request={
                                "parent": f"projects/{project_id}",
                                "secret_id": secret_id,
                                "secret": secret,
                            }
                        )
                        client.add_secret_version(
                            request={
                                "parent": response.name,
                                "payload": version["payload"],
                            }
                        )
                    except Exception as create_error:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to create {config_type} secret: {str(create_error)}",
                        )
                else:
                    # Some other error, re-raise
                    raise e
        else:
            # Fallback: store encrypted API key in UserPreferences
            encrypted_key = SecretManager.encrypt_api_key(config.api_key)
            preferences[f"api_key_{provider}"] = encrypted_key

    @router.post("/secrets")
    def create_secret(
        request: CreateSecretRequest,
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        customer_id = user["user_id"]
        client, project_id = SecretManager.get_client_and_project()

        try:
            # Update user preferences with provider info and models regardless of GCP or not.
            user_pref = (
                db.query(UserPreferences)
                .filter(UserPreferences.user_id == customer_id)
                .first()
            )
            if not user_pref:
                user_pref = UserPreferences(user_id=customer_id, preferences={})
                db.add(user_pref)
                db.flush()  # Ensure the new record is created before modifying preferences

            # Create a copy of preferences to avoid modifying the dict directly
            preferences = user_pref.preferences.copy() if user_pref.preferences else {}
            updated_providers = []

            # Process configurations
            SecretManager._process_config(
                request.chat_config,
                "chat",
                customer_id,
                client,
                project_id,
                preferences,
                updated_providers,
            )
            SecretManager._process_config(
                request.inference_config,
                "inference",
                customer_id,
                client,
                project_id,
                preferences,
                updated_providers,
            )

            # Update the preferences after all operations are successful
            user_pref.preferences = preferences
            db.commit()
            db.refresh(user_pref)  # Refresh to ensure we have the latest state

            PostHogClient().send_event(
                customer_id,
                "secret_creation_event",
                {"providers": updated_providers, "key_added": "true"},
            )
            return {"message": "Secret created successfully"}
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=500, detail=f"Failed to create secret: {str(e)}"
            )

    @router.get("/secrets/{provider}")
    def get_secret_for_provider(
        provider: Literal[
            "openai",
            "anthropic",
            "deepseek",
            "meta-llama",
            "gemini",
            "openrouter",
            "all",
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
                status_code=404, detail="No secrets found for this user"
            )

        if provider == "all":
            # Get both chat and inference configurations
            chat_model = user_pref.preferences.get("chat_model")
            inference_model = user_pref.preferences.get("inference_model")

            result = {"chat_config": None, "inference_config": None}

            # Process chat configuration if it exists
            if chat_model:
                chat_provider = chat_model.split("/")[0]
                try:
                    chat_secret = SecretManager.get_secret(
                        chat_provider, customer_id, db
                    )
                    result["chat_config"] = {
                        "provider": chat_provider,
                        "model": chat_model,
                        "api_key": chat_secret["api_key"],
                    }
                except HTTPException:
                    pass  # Skip if secret not found

            # Process inference configuration if it exists
            if inference_model:
                inference_provider = inference_model.split("/")[0]
                try:
                    inference_secret = SecretManager.get_secret(
                        inference_provider, customer_id, db
                    )
                    result["inference_config"] = {
                        "provider": inference_provider,
                        "model": inference_model,
                        "api_key": inference_secret["api_key"],
                    }
                except HTTPException:
                    pass  # Skip if secret not found

            if result["chat_config"] is None and result["inference_config"] is None:
                raise HTTPException(
                    status_code=404, detail="No secrets found for this user"
                )

            return result

        # For single provider requests, maintain existing behavior
        return SecretManager.get_secret(provider, customer_id, db)

    @staticmethod
    def get_secret(
        provider: Literal[
            "openai",
            "anthropic",
            "deepseek",
            "meta-llama",
            "gemini",
            "openrouter",
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
            if not user_pref:
                user_pref = UserPreferences(user_id=customer_id, preferences={})
                db.add(user_pref)
                db.flush()  # Ensure the new record is created before modifying preferences

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
        try:
            # Get or create user preferences
            user_pref = (
                db.query(UserPreferences)
                .filter(UserPreferences.user_id == customer_id)
                .first()
            )
            if not user_pref:
                user_pref = UserPreferences(user_id=customer_id, preferences={})
                db.add(user_pref)
                db.flush()  # Ensure the new record is created before modifying preferences

            # Create a copy of preferences to avoid modifying the dict directly
            preferences = user_pref.preferences.copy() if user_pref.preferences else {}
            updated_providers = []

            # Get client and project_id
            client, project_id = SecretManager.get_client_and_project()

            # Process configurations
            SecretManager._process_config(
                request.chat_config,
                "chat",
                customer_id,
                client,
                project_id,
                preferences,
                updated_providers,
            )
            SecretManager._process_config(
                request.inference_config,
                "inference",
                customer_id,
                client,
                project_id,
                preferences,
                updated_providers,
            )

            # Update the preferences after all operations are successful
            user_pref.preferences = preferences
            db.commit()
            db.refresh(user_pref)

            PostHogClient().send_event(
                customer_id,
                "secret_update_event",
                {"providers": updated_providers, "key_updated": "true"},
            )
            return {"message": "Secret updated successfully"}
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=500, detail=f"Failed to update secret: {str(e)}"
            )

    @router.delete("/secrets/{provider}")
    async def delete_secret(
        provider: Literal[
            "openai",
            "anthropic",
            "deepseek",
            "meta-llama",
            "gemini",
            "openrouter",
            "all",
        ],
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        customer_id = user["user_id"]
        from app.modules.intelligence.provider.provider_service import (
            PLATFORM_PROVIDERS,
        )

        async def delete_single_provider(prov: str) -> dict:
            """Helper function to delete a single provider's secret"""
            try:
                client, project_id = SecretManager.get_client_and_project()
                if client and project_id:
                    secret_id = SecretManager.get_secret_id(prov, customer_id)
                    try:
                        name = f"projects/{project_id}/secrets/{secret_id}"
                        client.delete_secret(request={"name": name})
                        PostHogClient().send_event(
                            customer_id,
                            "secret_deletion_event",
                            {"provider": prov, "key_removed": "true"},
                        )
                        return {
                            "provider": prov,
                            "status": "success",
                            "message": f"Successfully deleted {prov} secret",
                        }
                    except Exception as e:
                        return {
                            "provider": prov,
                            "status": "error",
                            "message": f"Failed to delete {prov} secret: {str(e)}",
                        }
                else:
                    # Fallback deletion: remove from UserPreferences
                    user_pref = (
                        db.query(UserPreferences)
                        .filter(UserPreferences.user_id == customer_id)
                        .first()
                    )
                    if user_pref and f"api_key_{prov}" in user_pref.preferences:
                        preferences = user_pref.preferences.copy()
                        del preferences[f"api_key_{prov}"]
                        user_pref.preferences = preferences
                        db.commit()
                        PostHogClient().send_event(
                            customer_id,
                            "secret_deletion_event",
                            {"provider": prov, "key_removed": "true"},
                        )
                        return {
                            "provider": prov,
                            "status": "success",
                            "message": f"Deleted {prov} secret from DB",
                        }
                    return {
                        "provider": prov,
                        "status": "not_found",
                        "message": f"No secret found for {prov}",
                    }
            except Exception as e:
                return {
                    "provider": prov,
                    "status": "error",
                    "message": f"Error processing {prov}: {str(e)}",
                }

        if provider == "all":
            try:
                # Create tasks for all providers
                tasks = [delete_single_provider(prov) for prov in PLATFORM_PROVIDERS]
                # Execute all deletions in parallel
                deletion_results = await asyncio.gather(*tasks)

                # Clean up provider preferences after all deletions
                user_pref = (
                    db.query(UserPreferences)
                    .filter(UserPreferences.user_id == customer_id)
                    .first()
                )
                if user_pref:
                    preferences = user_pref.preferences.copy()
                    if "provider" in preferences:
                        del preferences["provider"]
                    if "low_reasoning_model" in preferences:
                        del preferences["low_reasoning_model"]
                    if "high_reasoning_model" in preferences:
                        del preferences["high_reasoning_model"]
                    if "chat_model" in preferences:
                        del preferences["chat_model"]
                    if "inference_model" in preferences:
                        del preferences["inference_model"]

                    user_pref.preferences = preferences
                    db.commit()

                successful = [r for r in deletion_results if r["status"] == "success"]
                failed = [r for r in deletion_results if r["status"] == "error"]
                not_found = [r for r in deletion_results if r["status"] == "not_found"]

                return {
                    "message": "All secrets deletion completed",
                    "successful_deletions": successful,
                    "failed_deletions": failed,
                    "not_found": not_found,
                }
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process parallel deletions: {str(e)}",
                )
        else:
            # Single provider deletion
            result = await delete_single_provider(provider)
            if result["status"] == "error":
                raise HTTPException(status_code=500, detail=result["message"])
            elif result["status"] == "not_found":
                raise HTTPException(status_code=404, detail=result["message"])
            return {"message": result["message"]}

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
            if e.status_code == 404:
                raise e
            raise HTTPException(
                status_code=500, detail=f"Failed to retrieve API key: {str(e)}"
            )

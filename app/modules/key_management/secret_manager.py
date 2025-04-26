import os
import asyncio
import functools
import logging
from typing import Literal, List, Dict, Optional

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
    IntegrationKey,
    CreateIntegrationKeyRequest,
    UpdateIntegrationKeyRequest,
)
from app.modules.users.user_preferences_model import UserPreferences
from app.modules.utils.APIRouter import APIRouter
from app.modules.utils.posthog_helper import PostHogClient

# Set up logging
logger = logging.getLogger(__name__)


class SecretStorageHandler:
    """Handles storage, retrieval, and deletion of secrets across different storage backends."""

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
    def get_encryption_key():
        """Get Fernet encryption key for local storage."""
        secret_key = os.environ.get("SECRET_ENCRYPTION_KEY")
        if not secret_key:
            raise HTTPException(
                status_code=500,
                detail="SECRET_ENCRYPTION_KEY environment variable is not set",
            )
        try:
            return Fernet(secret_key.encode("utf-8"))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid SECRET_ENCRYPTION_KEY: {str(e)}",
            )

    @staticmethod
    def format_secret_id(service, customer_id, service_type="ai_provider"):
        """Generate a standardized secret ID."""
        if os.environ.get("isDevelopmentMode") == "enabled":
            return None

        prefix = ""
        if service_type == "integration":
            prefix = "integration-"

        return f"{prefix}{service}-api-key-{customer_id}"

    @staticmethod
    def encrypt_value(value):
        """Encrypt a value for local storage."""
        f = SecretStorageHandler.get_encryption_key()
        encrypted = f.encrypt(value.encode("utf-8"))
        return encrypted.decode("utf-8")

    @staticmethod
    def decrypt_value(encrypted_value):
        """Decrypt a value from local storage."""
        f = SecretStorageHandler.get_encryption_key()
        try:
            decrypted = f.decrypt(encrypted_value.encode("utf-8"))
            return decrypted.decode("utf-8")
        except InvalidToken:
            raise HTTPException(
                status_code=500, detail="Failed to decrypt value. Invalid token."
            )

    @staticmethod
    def store_secret(
        service,
        customer_id,
        api_key,
        service_type="ai_provider",
        db=None,
        preferences=None,
    ):
        """Store a secret in GCP or fallback to database."""
        try:
            logger.info(
                f"Storing secret for service: {service}, type: {service_type}, user: {customer_id}"
            )
            client, project_id = SecretStorageHandler.get_client_and_project()

            if client and project_id:
                # Store in Google Secret Manager
                secret_id = SecretStorageHandler.format_secret_id(
                    service, customer_id, service_type
                )
                parent = f"projects/{project_id}/secrets/{secret_id}"
                version = {"payload": {"data": api_key.encode("UTF-8")}}
                logger.info(f"Attempting to store secret in GCP: {parent}")

                try:
                    # Try to update existing secret
                    client.add_secret_version(
                        request={"parent": parent, "payload": version["payload"]}
                    )
                    logger.info(
                        f"Successfully updated existing secret in GCP for {service}"
                    )
                except Exception as e:
                    # The secret might not exist yet, try creating it
                    if (
                        "not found" in str(e).lower()
                        or "404" in str(e)
                        or "409" in str(e)
                    ):
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
                            logger.info(
                                f"Successfully created new secret in GCP for {service}"
                            )
                        except Exception as create_error:
                            logger.error(
                                f"Failed to create secret in GCP for {service}: {str(create_error)}"
                            )
                            raise HTTPException(
                                status_code=500,
                                detail=f"Failed to create secret: {str(create_error)}",
                            )
                    else:
                        # Some other error, re-raise
                        raise e
            elif db and preferences is not None:
                # Fallback: store encrypted API key in UserPreferences
                encrypted_key = SecretStorageHandler.encrypt_value(api_key)
                if service_type == "integration":
                    key_name = f"integration_api_key_{service}"
                else:
                    key_name = f"api_key_{service}"

                logger.info(
                    f"Storing encrypted key in preferences with key: {key_name}"
                )
                preferences[key_name] = encrypted_key
                logger.info(
                    f"Successfully stored encrypted key in preferences for {service}"
                )
            else:
                logger.error("Neither GCP nor database storage is available")
                raise HTTPException(
                    status_code=500,
                    detail="Neither GCP nor database storage is available",
                )
        except Exception as e:
            logger.error(f"Error storing secret for {service}: {str(e)}")
            raise

    @staticmethod
    def get_secret(
        service, customer_id, service_type="ai_provider", db=None, preferences=None
    ):
        """Get a secret from GCP or fallback to database."""
        try:
            logger.info(
                f"Getting secret for service: {service}, type: {service_type}, user: {customer_id}"
            )
            client, project_id = SecretStorageHandler.get_client_and_project()

            if client and project_id:
                # Try to get from Google Secret Manager
                secret_id = SecretStorageHandler.format_secret_id(
                    service, customer_id, service_type
                )
                name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
                logger.info(f"Attempting to get secret from GCP: {name}")

                try:
                    response = client.access_secret_version(request={"name": name})
                    secret = response.payload.data.decode("UTF-8")
                    logger.info(f"Successfully retrieved secret from GCP for {service}")
                    return secret
                except Exception as e:
                    logger.warning(
                        f"Failed to get secret from GCP for {service}: {str(e)}"
                    )

            if db and preferences is not None:
                # Fallback: get from UserPreferences
                if service_type == "integration":
                    key_name = f"integration_api_key_{service}"
                else:
                    key_name = f"api_key_{service}"

                logger.info(f"Looking for key in preferences: {key_name}")
                if key_name in preferences:
                    encrypted_key = preferences[key_name]
                    secret = SecretStorageHandler.decrypt_value(encrypted_key)
                    logger.info(
                        f"Successfully retrieved secret from preferences for {service}"
                    )
                    return secret
                else:
                    logger.warning(f"No encrypted key found for {service}")
            else:
                logger.error("Neither GCP nor database storage is available")

            raise HTTPException(
                status_code=404, detail=f"Secret not found for {service}"
            )
        except Exception as e:
            logger.error(f"Error getting secret for {service}: {str(e)}")
            raise

    @staticmethod
    def delete_secret(service, customer_id, service_type="ai_provider", db=None):
        """Delete a secret from GCP or fallback to database."""
        deleted = False
        client, project_id = SecretStorageHandler.get_client_and_project()

        if client and project_id:
            secret_id = SecretStorageHandler.format_secret_id(
                service, customer_id, service_type
            )
            try:
                name = f"projects/{project_id}/secrets/{secret_id}"
                client.delete_secret(request={"name": name})
                deleted = True
            except Exception:
                # Secret might not exist, try fallback
                pass

        # Try database fallback if needed
        if not deleted and db:
            user_pref = (
                db.query(UserPreferences)
                .filter(UserPreferences.user_id == customer_id)
                .first()
            )

            if service_type == "integration":
                key_name = f"integration_api_key_{service}"
            else:
                key_name = f"api_key_{service}"

            if user_pref and key_name in user_pref.preferences:
                preferences = user_pref.preferences.copy()
                del preferences[key_name]
                user_pref.preferences = preferences
                db.commit()
                deleted = True

        if not deleted:
            raise HTTPException(
                status_code=404, detail=f"No secret found for {service}"
            )

        return True

    @staticmethod
    async def check_secret_exists(
        service, customer_id, service_type="ai_provider", db=None
    ):
        """Check if a secret exists without raising exceptions."""
        client, project_id = SecretStorageHandler.get_client_and_project()

        if client and project_id:
            secret_id = SecretStorageHandler.format_secret_id(
                service, customer_id, service_type
            )
            name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
            try:
                client.access_secret_version(request={"name": name})
                return True  # Secret exists in GCP
            except Exception:
                pass  # Secret not found in GCP, fallback to checking UserPreferences

        # Fallback: check UserPreferences
        if not db:
            return False

        user_pref = (
            db.query(UserPreferences)
            .filter(UserPreferences.user_id == customer_id)
            .first()
        )

        if not user_pref or not user_pref.preferences:
            return False

        # For integrations, check the appropriate key format
        if service_type == "integration":
            key_name = f"integration_api_key_{service}"
        else:
            key_name = f"api_key_{service}"

        if user_pref.preferences.get(key_name):
            return True  # Secret exists in UserPreferences

        return False  # Secret not found anywhere


class SecretProcessor:
    """Handles unified processing of different secret types with consistent error handling."""

    @staticmethod
    def handle_secret_operation(operation_func):
        """Decorator for consistent error handling in secret operations"""

        @functools.wraps(operation_func)
        def wrapper(*args, **kwargs):
            try:
                return operation_func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, HTTPException):
                    raise  # Re-raise HTTP exceptions
                db = kwargs.get("db")
                if db:
                    db.rollback()
                operation_name = operation_func.__name__.replace("_", " ")
                raise HTTPException(
                    status_code=500, detail=f"Failed to {operation_name}: {str(e)}"
                )

        return wrapper

    @staticmethod
    def format_success_response(message: str, data: Optional[Dict] = None) -> Dict:
        """Format a standardized success response"""
        response = {"message": message}
        if data:
            response.update(data)
        return response

    @staticmethod
    def process_provider_config(
        config: Optional[BaseSecret],
        config_type: Literal["chat", "inference"],
        customer_id: str,
        preferences: dict,
        db: Session,
    ) -> Optional[str]:
        """Process AI provider configuration and return the provider if successful."""
        if not config:
            return None

        # Update preferences for the model
        preferences[f"{config_type}_model"] = config.model
        provider = config.model.split("/")[0]

        # Store the secret
        SecretStorageHandler.store_secret(
            provider, customer_id, config.api_key, "ai_provider", db, preferences
        )

        return provider

    @staticmethod
    def process_integration_config(
        integration_key: Optional[IntegrationKey],
        customer_id: str,
        preferences: dict,
        db: Session,
    ) -> Optional[str]:
        """Process integration key configuration and return the service if successful."""
        if not integration_key:
            return None

        service = integration_key.service

        # Store the secret
        SecretStorageHandler.store_secret(
            service,
            customer_id,
            integration_key.api_key,
            "integration",
            db,
            preferences,
        )

        return service

    @staticmethod
    async def process_bulk_operation(
        operation_func,
        services: List[str],
        customer_id: str,
        service_type: str = "ai_provider",
        db: Session = None,
    ):
        """Process multiple services in parallel with consistent result format."""

        async def process_one(service):
            try:
                # Check if operation_func is a coroutine function
                if asyncio.iscoroutinefunction(operation_func):
                    await operation_func(service, customer_id, service_type, db)
                else:
                    operation_func(service, customer_id, service_type, db)

                return {
                    "service": service,
                    "status": "success",
                    "message": f"Successfully processed {service}",
                }
            except Exception as e:
                if isinstance(e, HTTPException) and e.status_code == 404:
                    return {
                        "service": service,
                        "status": "not_found",
                        "message": f"No secret found for {service}",
                    }
                return {
                    "service": service,
                    "status": "error",
                    "message": f"Error processing {service}: {str(e)}",
                }

        tasks = [process_one(service) for service in services]
        results = await asyncio.gather(*tasks)

        return {
            "successful": [r for r in results if r["status"] == "success"],
            "failed": [r for r in results if r["status"] == "error"],
            "not_found": [r for r in results if r["status"] == "not_found"],
        }


router = APIRouter()

# Define service categories
SERVICE_CATEGORIES = {
    "ai_provider": [
        "openai",
        "anthropic",
        "deepseek",
        "meta-llama",
        "gemini",
        "openrouter",
    ],
    "integration": ["linear", "notion"],
}

# Define service types using the categories
AIProviderType = Literal[
    "openai",
    "anthropic",
    "deepseek",
    "meta-llama",
    "gemini",
    "openrouter",
]

IntegrationServiceType = Literal[
    "linear",
    "notion",
]

# Create a unified ServiceType that includes all services
ServiceType = Literal[
    "openai",
    "anthropic",
    "deepseek",
    "meta-llama",
    "gemini",
    "openrouter",
    "linear",
    "notion",
]


class SecretManager:
    """
    Manages API keys and other secrets for various services.
    This class is simplified to use the utility classes.
    """

    # Define category constants for better code clarity
    AI_PROVIDER = "ai_provider"
    INTEGRATION = "integration"

    # Use the centralized service categories
    AI_PROVIDERS = SERVICE_CATEGORIES["ai_provider"]
    INTEGRATION_SERVICES = SERVICE_CATEGORIES["integration"]

    @staticmethod
    def encrypt_api_key(api_key: str) -> str:
        """Legacy compatibility method - delegates to SecretStorageHandler"""
        return SecretStorageHandler.encrypt_value(api_key)

    @staticmethod
    def decrypt_api_key(encrypted_key: str) -> str:
        """Legacy compatibility method - delegates to SecretStorageHandler"""
        return SecretStorageHandler.decrypt_value(encrypted_key)

    @staticmethod
    def get_client_and_project():
        """Legacy compatibility method - delegates to SecretStorageHandler"""
        return SecretStorageHandler.get_client_and_project()

    @staticmethod
    def get_secret_id(
        service: ServiceType,
        customer_id: str,
        service_type: Literal["ai_provider", "integration"] = "ai_provider",
    ) -> str:
        """Legacy compatibility method - delegates to SecretStorageHandler"""
        return SecretStorageHandler.format_secret_id(service, customer_id, service_type)

    @staticmethod
    async def check_secret_exists_for_user(
        service: ServiceType,
        user_id: str,
        db: Session,
        service_type: Literal["ai_provider", "integration"] = "ai_provider",
    ) -> bool:
        """Legacy compatibility method - delegates to SecretStorageHandler"""
        return await SecretStorageHandler.check_secret_exists(
            service, user_id, service_type, db
        )

    @staticmethod
    def get_secret(
        service: ServiceType,
        customer_id: str,
        db: Session = None,
        service_type: Literal["ai_provider", "integration"] = "ai_provider",
    ):
        """Retrieve a secret - delegates to SecretStorageHandler"""
        return SecretStorageHandler.get_secret(service, customer_id, service_type, db)

    @router.post("/secrets")
    @SecretProcessor.handle_secret_operation
    def create_secret(
        request: CreateSecretRequest,
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        customer_id = user["user_id"]

        # Get or create user preferences
        user_pref = (
            db.query(UserPreferences)
            .filter(UserPreferences.user_id == customer_id)
            .first()
        )
        if not user_pref:
            user_pref = UserPreferences(user_id=customer_id, preferences={})
            db.add(user_pref)
            db.flush()

        # Create a copy of preferences to avoid modifying the dict directly
        preferences = user_pref.preferences.copy() if user_pref.preferences else {}
        updated_providers = []

        # Process configurations using our utility methods
        chat_provider = SecretProcessor.process_provider_config(
            request.chat_config, "chat", customer_id, preferences, db
        )
        if chat_provider:
            updated_providers.append(chat_provider)

        inference_provider = SecretProcessor.process_provider_config(
            request.inference_config, "inference", customer_id, preferences, db
        )
        if inference_provider:
            updated_providers.append(inference_provider)

        # Update the preferences after all operations are successful
        user_pref.preferences = preferences
        db.commit()
        db.refresh(user_pref)

        # Track with PostHog if providers were updated
        if updated_providers:
            PostHogClient().send_event(
                customer_id,
                "secret_creation_event",
                {"providers": updated_providers, "key_added": "true"},
            )

        return SecretProcessor.format_success_response("Secret created successfully")

    @router.get("/secrets/{provider}")
    @SecretProcessor.handle_secret_operation
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
        try:
            customer_id = user["user_id"]
            logger.info(
                f"Getting secrets for user: {customer_id}, provider: {provider}"
            )

            user_pref = (
                db.query(UserPreferences)
                .filter(UserPreferences.user_id == customer_id)
                .first()
            )
            if not user_pref:
                logger.info(f"No preferences found for user: {customer_id}")
                raise

            if provider == "all":
                # Get both chat and inference configurations
                chat_model = user_pref.preferences.get("chat_model")
                inference_model = user_pref.preferences.get("inference_model")
                logger.info(
                    f"User preferences - chat_model: {chat_model}, inference_model: {inference_model}"
                )

                result = {
                    "chat_config": None,
                    "inference_config": None,
                    "integration_keys": [],
                }

                # Process chat configuration if it exists
                if chat_model:
                    chat_provider = chat_model.split("/")[0]
                    try:
                        logger.info(
                            f"Getting chat secret for provider: {chat_provider}"
                        )
                        chat_secret = SecretManager.get_secret(
                            chat_provider, customer_id, db
                        )
                        result["chat_config"] = {
                            "provider": chat_provider,
                            "model": chat_model,
                            "api_key": chat_secret,
                        }
                        logger.info(
                            f"Successfully retrieved chat secret for {chat_provider}"
                        )
                    except HTTPException as e:
                        logger.warning(
                            f"Failed to get chat secret for {chat_provider}: {str(e)}"
                        )
                        pass

                # Process inference configuration if it exists
                if inference_model:
                    inference_provider = inference_model.split("/")[0]
                    try:
                        logger.info(
                            f"Getting inference secret for provider: {inference_provider}"
                        )
                        inference_secret = SecretManager.get_secret(
                            inference_provider, customer_id, db
                        )
                        result["inference_config"] = {
                            "provider": inference_provider,
                            "model": inference_model,
                            "api_key": inference_secret,
                        }
                        logger.info(
                            f"Successfully retrieved inference secret for {inference_provider}"
                        )
                    except HTTPException as e:
                        logger.warning(
                            f"Failed to get inference secret for {inference_provider}: {str(e)}"
                        )
                        pass

                # Process integration keys
                logger.info("Processing integration keys")
                for service in SecretManager.INTEGRATION_SERVICES:
                    try:
                        logger.info(f"Checking integration key for service: {service}")
                        secret = SecretManager.get_secret(
                            service, customer_id, db, SecretManager.INTEGRATION
                        )
                        result["integration_keys"].append(
                            {
                                "service": service,
                                "api_key": secret,
                            }
                        )
                        logger.info(
                            f"Successfully retrieved integration key for {service}"
                        )
                    except HTTPException as e:
                        logger.warning(
                            f"Failed to get integration key for {service}: {str(e)}"
                        )
                        pass

                if (
                    result["chat_config"] is None
                    and result["inference_config"] is None
                    and not result["integration_keys"]
                ):
                    logger.warning(f"No secrets found for user: {customer_id}")
                    raise HTTPException(
                        status_code=404, detail="No secrets found for this user"
                    )

                logger.info(
                    f"Successfully retrieved all secrets for user: {customer_id}"
                )
                return result

            # For single provider requests
            logger.info(f"Getting secret for single provider: {provider}")
            secret = SecretManager.get_secret(provider, customer_id, db)
            logger.info(f"Successfully retrieved secret for provider: {provider}")
            return secret

        except Exception as e:
            logger.error(f"Error getting secrets for user {user['user_id']}: {str(e)}")
            raise

    @router.put("/secrets/")
    @SecretProcessor.handle_secret_operation
    def update_secret(
        request: UpdateSecretRequest,
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        customer_id = user["user_id"]

        # Get or create user preferences
        user_pref = (
            db.query(UserPreferences)
            .filter(UserPreferences.user_id == customer_id)
            .first()
        )
        if not user_pref:
            user_pref = UserPreferences(user_id=customer_id, preferences={})
            db.add(user_pref)
            db.flush()

        # Create a copy of preferences to avoid modifying the dict directly
        preferences = user_pref.preferences.copy() if user_pref.preferences else {}
        updated_providers = []

        # Process configurations using our utility methods
        chat_provider = SecretProcessor.process_provider_config(
            request.chat_config, "chat", customer_id, preferences, db
        )
        if chat_provider:
            updated_providers.append(chat_provider)

        inference_provider = SecretProcessor.process_provider_config(
            request.inference_config, "inference", customer_id, preferences, db
        )
        if inference_provider:
            updated_providers.append(inference_provider)

        # Update the preferences after all operations are successful
        user_pref.preferences = preferences
        db.commit()
        db.refresh(user_pref)

        # Track with PostHog if providers were updated
        if updated_providers:
            PostHogClient().send_event(
                customer_id,
                "secret_update_event",
                {"providers": updated_providers, "key_updated": "true"},
            )

        return SecretProcessor.format_success_response("Secret updated successfully")

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

        async def delete_provider_secret(service):
            """Helper to delete a single provider secret"""
            try:
                SecretStorageHandler.delete_secret(
                    service, customer_id, SecretManager.AI_PROVIDER, db
                )
                PostHogClient().send_event(
                    customer_id,
                    "secret_deletion_event",
                    {"provider": service, "key_removed": "true"},
                )
                return True
            except HTTPException as e:
                if e.status_code != 404:  # Only re-raise non-404 errors
                    raise
                return False

        if provider == "all":
            from app.modules.intelligence.provider.provider_service import (
                PLATFORM_PROVIDERS,
            )

            # Use our bulk operation processor
            result = await SecretProcessor.process_bulk_operation(
                delete_provider_secret,
                PLATFORM_PROVIDERS,
                customer_id,
                SecretManager.AI_PROVIDER,
                db,
            )

            # Clean up provider preferences
            user_pref = (
                db.query(UserPreferences)
                .filter(UserPreferences.user_id == customer_id)
                .first()
            )
            if user_pref:
                preferences = user_pref.preferences.copy()
                for key in [
                    "provider",
                    "low_reasoning_model",
                    "high_reasoning_model",
                    "chat_model",
                    "inference_model",
                ]:
                    if key in preferences:
                        del preferences[key]

                user_pref.preferences = preferences
                db.commit()

            return {
                "message": "All secrets deletion completed",
                "successful_deletions": result["successful"],
                "failed_deletions": result["failed"],
                "not_found": result["not_found"],
            }
        else:
            # Single provider deletion
            try:
                deleted = await delete_provider_secret(provider)
                if not deleted:
                    raise HTTPException(
                        status_code=404, detail=f"No secret found for {provider}"
                    )
                return SecretProcessor.format_success_response(
                    f"Successfully deleted {provider} secret"
                )
            except HTTPException:
                raise  # Re-raise any HTTP exceptions

    @router.post("/api-keys", response_model=APIKeyResponse)
    async def create_api_key(
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        try:
            logger.info(f"Creating API key for user: {user['user_id']}")
            api_key = await APIKeyService.create_api_key(user["user_id"], db)
            logger.info(f"API key created successfully for user: {user['user_id']}")
            PostHogClient().send_event(
                user["user_id"], "api_key_creation", {"success": True}
            )
            return APIKeyResponse(api_key=api_key)
        except Exception as e:
            logger.error(f"Error creating API key for user {user['user_id']}: {str(e)}")
            raise

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
            logger.info(f"Getting API key for user: {user['user_id']}")
            api_key = await APIKeyService.get_api_key(user["user_id"], db)
            logger.info(f"API key retrieved successfully for user: {user['user_id']}")
            if api_key is None:
                logger.info(f"No API key found for user: {user['user_id']}")
                raise
            return APIKeyResponse(api_key=api_key)
        except Exception as e:
            logger.error(f"Error getting API key for user {user['user_id']}: {str(e)}")
            raise

    # Integration key endpoints

    @router.post("/integration-keys")
    async def create_integration_keys(
        request: CreateIntegrationKeyRequest,
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        customer_id = user["user_id"]

        # Get or create user preferences
        user_pref = (
            db.query(UserPreferences)
            .filter(UserPreferences.user_id == customer_id)
            .first()
        )
        if not user_pref:
            user_pref = UserPreferences(user_id=customer_id, preferences={})
            db.add(user_pref)
            db.flush()

        # Create a copy of preferences
        preferences = user_pref.preferences.copy() if user_pref.preferences else {}
        updated_services = []

        # Process each integration key
        for integration_key in request.integration_keys:
            service = integration_key.service
            api_key = integration_key.api_key

            # Store the secret
            try:
                # First try GCP if available
                client, project_id = SecretStorageHandler.get_client_and_project()

                if client and project_id:
                    secret_id = SecretStorageHandler.format_secret_id(
                        service, customer_id, "integration"
                    )
                    parent = f"projects/{project_id}/secrets/{secret_id}"
                    version = {"payload": {"data": api_key.encode("UTF-8")}}

                    try:
                        # Try to update existing secret
                        client.add_secret_version(
                            request={"parent": parent, "payload": version["payload"]}
                        )
                    except Exception as e:
                        # The secret might not exist yet, try creating it
                        if (
                            "not found" in str(e).lower()
                            or "404" in str(e)
                            or "409" in str(e)
                        ):
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
                        else:
                            # Some other error, fall back to database
                            raise e
                else:
                    # No GCP, store in database
                    encrypted_key = SecretStorageHandler.encrypt_value(api_key)
                    key_name = f"integration_api_key_{service}"
                    preferences[key_name] = encrypted_key

                updated_services.append(service)
            except Exception as e:
                logger.error(f"Error storing integration key for {service}: {str(e)}")
                continue

        # Update preferences
        user_pref.preferences = preferences
        db.commit()
        db.refresh(user_pref)

        # Track with PostHog if services were updated
        if updated_services:
            PostHogClient().send_event(
                customer_id,
                "integration_key_creation_event",
                {"services": updated_services, "keys_added": "true"},
            )

        return {"message": "Integration keys created successfully"}

    @router.get("/integration-keys/{service}")
    async def get_integration_key(
        service: IntegrationServiceType,
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        customer_id = user["user_id"]

        # Check if GCP project is set
        client, project_id = SecretStorageHandler.get_client_and_project()

        if client and project_id:
            # If GCP is available, only check there
            secret_id = SecretStorageHandler.format_secret_id(
                service, customer_id, "integration"
            )
            name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"

            try:
                response = client.access_secret_version(request={"name": name})
                return response.payload.data.decode("UTF-8")
            except Exception as e:
                logger.warning(f"Failed to get secret from GCP for {service}: {str(e)}")
                raise HTTPException(
                    status_code=404, detail=f"Secret not found for {service} in GCP"
                )
        else:
            # If GCP is not available, check database
            user_pref = (
                db.query(UserPreferences)
                .filter(UserPreferences.user_id == customer_id)
                .first()
            )

            if user_pref and user_pref.preferences:
                key_name = f"integration_api_key_{service}"
                if key_name in user_pref.preferences:
                    encrypted_key = user_pref.preferences[key_name]
                    return SecretStorageHandler.decrypt_value(encrypted_key)

            # If not found in database
            raise HTTPException(
                status_code=404, detail=f"Secret not found for {service} in database"
            )

    @router.get("/integration-keys")
    async def get_all_integration_keys(
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        customer_id = user["user_id"]
        results = []

        # Check if GCP project is set
        client, project_id = SecretStorageHandler.get_client_and_project()

        if client and project_id:
            # If GCP is available, only check there
            for service in SecretManager.INTEGRATION_SERVICES:
                try:
                    secret_id = SecretStorageHandler.format_secret_id(
                        service, customer_id, "integration"
                    )
                    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"

                    response = client.access_secret_version(request={"name": name})
                    secret = response.payload.data.decode("UTF-8")
                    results.append({"service": service, "api_key": secret})
                except Exception:
                    # Secret not found in GCP, skip
                    pass
        else:
            # If GCP is not available, check database
            user_pref = (
                db.query(UserPreferences)
                .filter(UserPreferences.user_id == customer_id)
                .first()
            )

            if user_pref and user_pref.preferences:
                for service in SecretManager.INTEGRATION_SERVICES:
                    try:
                        key_name = f"integration_api_key_{service}"
                        if key_name in user_pref.preferences:
                            encrypted_key = user_pref.preferences[key_name]
                            secret = SecretStorageHandler.decrypt_value(encrypted_key)
                            results.append({"service": service, "api_key": secret})
                    except Exception:
                        # Skip any errors for individual services
                        continue

        if not results:
            raise HTTPException(
                status_code=404, detail="No integration keys found for this user"
            )

        return results

    @router.put("/integration-keys")
    async def update_integration_keys(
        request: UpdateIntegrationKeyRequest,
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        customer_id = user["user_id"]

        # Get user preferences
        user_pref = (
            db.query(UserPreferences)
            .filter(UserPreferences.user_id == customer_id)
            .first()
        )
        if not user_pref:
            user_pref = UserPreferences(user_id=customer_id, preferences={})
            db.add(user_pref)
            db.flush()

        # Create a copy of preferences
        preferences = user_pref.preferences.copy() if user_pref.preferences else {}
        updated_services = []

        # Process each integration key - same logic as create but for update
        for integration_key in request.integration_keys:
            service = integration_key.service
            api_key = integration_key.api_key

            # Store the secret
            try:
                # First try GCP if available
                client, project_id = SecretStorageHandler.get_client_and_project()

                if client and project_id:
                    secret_id = SecretStorageHandler.format_secret_id(
                        service, customer_id, "integration"
                    )
                    parent = f"projects/{project_id}/secrets/{secret_id}"
                    version = {"payload": {"data": api_key.encode("UTF-8")}}

                    try:
                        # Try to update existing secret
                        client.add_secret_version(
                            request={"parent": parent, "payload": version["payload"]}
                        )
                    except Exception as e:
                        # The secret might not exist yet, try creating it
                        if (
                            "not found" in str(e).lower()
                            or "404" in str(e)
                            or "409" in str(e)
                        ):
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
                        else:
                            # Some other error, fall back to database
                            raise e
                else:
                    # No GCP, store in database
                    encrypted_key = SecretStorageHandler.encrypt_value(api_key)
                    key_name = f"integration_api_key_{service}"
                    preferences[key_name] = encrypted_key

                updated_services.append(service)
            except Exception as e:
                logger.error(f"Error updating integration key for {service}: {str(e)}")
                continue

        # Update preferences
        user_pref.preferences = preferences
        db.commit()
        db.refresh(user_pref)

        # Track with PostHog if services were updated
        if updated_services:
            PostHogClient().send_event(
                customer_id,
                "integration_key_update_event",
                {"services": updated_services, "keys_updated": "true"},
            )

        return {"message": "Integration keys updated successfully"}

    @router.delete("/integration-keys/{service}")
    async def delete_integration_key(
        service: IntegrationServiceType,
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        customer_id = user["user_id"]

        # Check if GCP project is set
        client, project_id = SecretStorageHandler.get_client_and_project()

        if client and project_id:
            # If GCP is available, only try there
            secret_id = SecretStorageHandler.format_secret_id(
                service, customer_id, "integration"
            )
            try:
                name = f"projects/{project_id}/secrets/{secret_id}"
                client.delete_secret(request={"name": name})

                # Track the deletion with PostHog
                PostHogClient().send_event(
                    customer_id,
                    "integration_key_deletion_event",
                    {"service": service, "key_removed": "true", "storage": "gcp"},
                )

                return {"message": f"Successfully deleted {service} integration key"}
            except Exception as e:
                logger.info(f"Secret not found in GCP or error deleting: {str(e)}")
                raise HTTPException(
                    status_code=404, detail=f"No secret found for {service}"
                )
        else:
            # If GCP is not available, check database
            user_pref = (
                db.query(UserPreferences)
                .filter(UserPreferences.user_id == customer_id)
                .first()
            )

            key_name = f"integration_api_key_{service}"
            if (
                user_pref
                and user_pref.preferences
                and key_name in user_pref.preferences
            ):
                preferences = user_pref.preferences.copy()
                del preferences[key_name]
                user_pref.preferences = preferences
                db.commit()

                # Track the deletion with PostHog
                PostHogClient().send_event(
                    customer_id,
                    "integration_key_deletion_event",
                    {"service": service, "key_removed": "true", "storage": "db"},
                )

                return {"message": f"Successfully deleted {service} integration key"}
            else:
                raise HTTPException(
                    status_code=404, detail=f"No secret found for {service}"
                )

    @router.delete("/integration-keys")
    async def delete_all_integration_keys(
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        customer_id = user["user_id"]
        successful_deletions = []
        not_found = []

        # Check if GCP project is set
        client, project_id = SecretStorageHandler.get_client_and_project()

        if client and project_id:
            # If GCP is available, only try there
            for service in SecretManager.INTEGRATION_SERVICES:
                secret_id = SecretStorageHandler.format_secret_id(
                    service, customer_id, "integration"
                )
                try:
                    name = f"projects/{project_id}/secrets/{secret_id}"
                    client.delete_secret(request={"name": name})
                    successful_deletions.append(
                        {
                            "service": service,
                            "status": "success",
                            "message": f"Successfully deleted {service}",
                        }
                    )

                    # Track the deletion with PostHog
                    PostHogClient().send_event(
                        customer_id,
                        "integration_key_deletion_event",
                        {"service": service, "key_removed": "true"},
                    )
                except Exception:
                    not_found.append(
                        {
                            "service": service,
                            "status": "not_found",
                            "message": f"No secret found for {service}",
                        }
                    )
        else:
            # If GCP is not available, check database
            user_pref = (
                db.query(UserPreferences)
                .filter(UserPreferences.user_id == customer_id)
                .first()
            )

            if user_pref and user_pref.preferences:
                # Look through all services in the database
                db_updated = False
                preferences = user_pref.preferences.copy()

                for service in SecretManager.INTEGRATION_SERVICES:
                    key_name = f"integration_api_key_{service}"
                    if key_name in preferences:
                        del preferences[key_name]
                        db_updated = True
                        successful_deletions.append(
                            {
                                "service": service,
                                "status": "success",
                                "message": f"Successfully deleted {service}",
                            }
                        )

                        # Track the deletion with PostHog
                        PostHogClient().send_event(
                            customer_id,
                            "integration_key_deletion_event",
                            {"service": service, "key_removed": "true"},
                        )
                    else:
                        not_found.append(
                            {
                                "service": service,
                                "status": "not_found",
                                "message": f"No secret found for {service}",
                            }
                        )

                # Only commit once after all deletions
                if db_updated:
                    user_pref.preferences = preferences
                    db.commit()
            else:
                # No user preferences found, all services are not found
                for service in SecretManager.INTEGRATION_SERVICES:
                    not_found.append(
                        {
                            "service": service,
                            "status": "not_found",
                            "message": f"No secret found for {service}",
                        }
                    )

        if not successful_deletions:
            raise HTTPException(
                status_code=404, detail="No integration keys found for this user"
            )

        return {
            "message": "Integration keys deletion completed",
            "successful_deletions": successful_deletions,
            "not_found": not_found,
        }

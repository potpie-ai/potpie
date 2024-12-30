import os
from typing import Literal

from fastapi import Depends, HTTPException
from google.cloud import secretmanager
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.key_management.secrets_schema import (
    CreateSecretRequest,
    UpdateSecretRequest,
)
from app.modules.users.user_preferences_model import UserPreferences
from app.modules.utils.APIRouter import APIRouter
from app.modules.utils.posthog_helper import PostHogClient

router = APIRouter()


class SecretManager:
    @staticmethod
    def get_client_and_project():
        if os.getenv("isDevelopmentMode") == "disabled":
            client = secretmanager.SecretManagerServiceClient()
            project_id = os.environ.get("GCP_PROJECT")
        else:
            client = None
            project_id = None
        return client, project_id

    @router.post("/secrets")
    def create_secret(
        request: CreateSecretRequest,
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        if os.getenv("isDevelopmentMode") == "enabled":
            return {"message": "Secret creation is not allowed in development mode"}
        customer_id = user["user_id"]
        client, project_id = SecretManager.get_client_and_project()

        # Update user preferences
        user_pref = (
            db.query(UserPreferences)
            .filter(UserPreferences.user_id == customer_id)
            .first()
        )
        if not user_pref:
            user_pref = UserPreferences(user_id=customer_id, preferences={})
            db.add(user_pref)
        user_pref.preferences["provider"] = request.provider
        db.commit()

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
        PostHogClient().send_event(
            customer_id,
            "secret_creation_event",
            {"provider": request.provider, "key_added": "true"},
        )

        return {"message": "Secret created successfully"}

    @staticmethod
    def get_secret_id(provider: Literal["openai", "anthropic", "google"], customer_id: str):
        if os.getenv("isDevelopmentMode") == "enabled":
            return None
        if provider == "openai":
            secret_id = f"openai-api-key-{customer_id}"
        elif provider == "anthropic":
            secret_id = f"anthropic-api-key-{customer_id}"
        elif provider == "google":
            secret_id = f"google-api-key-{customer_id}"
        else:
            raise HTTPException(status_code=400, detail="Invalid provider")
        return secret_id

    @router.get("/secrets/{provider}")
    def get_secret_for_provider(
        provider: Literal["openai", "anthropic", "google"],
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        if os.getenv("isDevelopmentMode") == "enabled":
            return None
        customer_id = user["user_id"]
        # Check user preferences first
        user_pref = (
            db.query(UserPreferences)
            .filter(UserPreferences.user_id == customer_id)
            .first()
        )
        if not user_pref:
            raise HTTPException(
                status_code=404, detail="Secret not found for this provider"
            )

        return SecretManager.get_secret(provider, customer_id)

    @staticmethod
    def get_secret(provider: Literal["openai", "anthropic", "google"], customer_id: str):
        if os.getenv("isDevelopmentMode") == "enabled":
            return None
        client, project_id = SecretManager.get_client_and_project()
        secret_id = SecretManager.get_secret_id(provider, customer_id)
        name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"

        try:
            response = client.access_secret_version(request={"name": name})
            api_key = response.payload.data.decode("UTF-8")
            return {"api_key": api_key}
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Secret not found in GCP Secret Manager: {str(e)}",
            )

    @router.put("/secrets/")
    def update_secret(
        request: UpdateSecretRequest,
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        if os.getenv("isDevelopmentMode") == "enabled":
            return {"message": "Secret update is not allowed in development mode"}
        customer_id = user["user_id"]
        api_key = request.api_key
        secret_id = SecretManager.get_secret_id(request.provider, customer_id)
        client, project_id = SecretManager.get_client_and_project()
        parent = f"projects/{project_id}/secrets/{secret_id}"
        version = {"payload": {"data": api_key.encode("UTF-8")}}
        client.add_secret_version(
            request={"parent": parent, "payload": version["payload"]}
        )

        # Update user preferences
        user_pref = (
            db.query(UserPreferences)
            .filter(UserPreferences.user_id == customer_id)
            .first()
        )
        if not user_pref:
            user_pref = UserPreferences(user_id=customer_id, preferences={})
            db.add(user_pref)
        user_pref.preferences["provider"] = request.provider
        db.commit()

        return {"message": "Secret updated successfully"}

    @router.delete("/secrets/{provider}")
    def delete_secret(
        provider: Literal["openai", "anthropic", "google"],
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        if os.getenv("isDevelopmentMode") == "enabled":
            return {"message": "Secret deletion is not allowed in development mode"}
        customer_id = user["user_id"]
        secret_id = SecretManager.get_secret_id(provider, customer_id)
        client, project_id = SecretManager.get_client_and_project()
        name = f"projects/{project_id}/secrets/{secret_id}"

        try:
            client.delete_secret(request={"name": name})
            # Remove provider from user preferences
            user_pref = (
                db.query(UserPreferences)
                .filter(UserPreferences.user_id == customer_id)
                .first()
            )
            if user_pref and "provider" in user_pref.preferences:
                del user_pref.preferences["provider"]
                db.commit()
            PostHogClient().send_event(
                customer_id,
                "secret_deletion_event",
                {"provider": provider, "key_removed": "true"},
            )
            return {"message": "Secret deleted successfully"}
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Secret not found: {str(e)}")
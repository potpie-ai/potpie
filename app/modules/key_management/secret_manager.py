import os
from typing import Literal

from fastapi import Depends, HTTPException
from google.cloud import secretmanager
from pydantic import BaseModel, validator
from app.modules.utils.APIRouter import APIRouter

from app.modules.auth.auth_service import AuthService
from app.core.mongo_manager import MongoManager

router = APIRouter()

if(os.getenv("isDevelopmentMode") == "disabled"):
    client = secretmanager.SecretManagerServiceClient()
    project_id = os.environ.get("GCP_PROJECT")
else:
    client = None
    project_id = None

import re


def validate_openai_api_key_format(api_key):
    pattern = r"^sk-[a-zA-Z0-9]{48}$"
    proj_pattern = r"^sk-proj-[a-zA-Z0-9]{48}$"
    return bool(re.match(pattern, api_key)) or bool(
        re.match(proj_pattern, api_key)
    )


class BaseSecretRequest(BaseModel):
    api_key: str
    provider: Literal["openai"] = "openai"

    @validator("api_key")
    def api_key_format(cls, v):
        if not validate_openai_api_key_format(v):
            raise ValueError("Invalid OpenAI API key format")
        return v


class CreateSecretRequest(BaseSecretRequest):
    pass


class UpdateSecretRequest(BaseSecretRequest):
    pass


@router.post("/secrets")
def create_secret(request: CreateSecretRequest, user=Depends(AuthService.check_auth)):
    customer_id = user["user_id"]

    with MongoManager.get_instance() as mongo_manager:
        mongo_manager.put(
            "preferences",
            customer_id,
            {"provider": request.provider}
        )

        api_key = request.api_key
        secret_id = get_secret_id(request.provider, customer_id)
        parent = f"projects/{project_id}"

        secret = {"replication": {"automatic": {}}}
        response = client.create_secret(
            request={"parent": parent, "secret_id": secret_id, "secret": secret}
        )

        version = {"payload": {"data": api_key.encode("UTF-8")}}
        client.add_secret_version(
            request={"parent": response.name, "payload": version["payload"]}
        )

        return {"message": "Secret created successfully"}


def get_secret_id(provider: Literal["openai"], customer_id: str):
    if provider == "openai":
        secret_id = f"openai-api-key-{customer_id}"
    else:
        raise HTTPException(status_code=400, detail="Invalid provider")
    return secret_id


@router.get("/secrets/{provider}")
def get_secret_for_provider(
    provider: Literal["openai"], user=Depends(AuthService.check_auth)
):
    customer_id = user["user_id"]
    return get_secret(provider, customer_id)


def get_secret(provider: Literal["openai"], customer_id: str):
    secret_id = get_secret_id(provider, customer_id)
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"

    try:
        response = client.access_secret_version(request={"name": name})
        api_key = response.payload.data.decode("UTF-8")
        return {"api_key": api_key}
    except Exception:
        raise HTTPException(status_code=404, detail="Secret not found")


@router.put("/secrets/")
def update_secret(request: UpdateSecretRequest, user=Depends(AuthService.check_auth)):
    customer_id = user["user_id"]
    api_key = request.api_key
    secret_id = get_secret_id(request.provider, customer_id)
    parent = f"projects/{project_id}/secrets/{secret_id}"
    version = {"payload": {"data": api_key.encode("UTF-8")}}
    client.add_secret_version(
        request={"parent": parent, "payload": version["payload"]}
    )

    with MongoManager.get_instance() as mongo_manager:
        mongo_manager.put(
            "preferences",
            customer_id,
            {"provider": request.provider}
        )

    return {"message": "Secret updated successfully"}


@router.delete("/secrets/{provider}")
def delete_secret(provider: Literal["openai"], user=Depends(AuthService.check_auth)):
    customer_id = user["user_id"]
    secret_id = get_secret_id(provider, customer_id)
    name = f"projects/{project_id}/secrets/{secret_id}"

    try:
        client.delete_secret(request={"name": name})
        with MongoManager.get_instance() as mongo_manager:
            mongo_manager.delete("preferences", customer_id)
        return {"message": "Secret deleted successfully"}
    except Exception:
        raise HTTPException(status_code=404, detail="Secret not found")
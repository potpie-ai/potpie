import re
from typing import Literal

from pydantic import BaseModel, field_validator


class BaseSecretRequest(BaseModel):
    api_key: str
    provider: Literal[ "anthropic", "deepseek"]
    @field_validator("api_key")
    @classmethod
    def api_key_format(cls, v: str):
        if v.startswith("sk-ant-"):
            return v
        elif v.startswith("sk-"):
            return v
        else:
            raise ValueError("Invalid API key format")

    @field_validator("provider")
    @classmethod
    def validate_provider_and_api_key(cls, provider: str, values):
        api_key = values.data.get("api_key")
        if provider == "anthropic":
            if not api_key.startswith("sk-ant-"):
                raise ValueError("Invalid Anthropic API key format")
        elif provider == "deepseek":
            if not api_key.startswith("sk-or-"):
                raise ValueError("Invalid OpenRouter API key format")
        else:
            raise ValueError("Invalid provider")
        return provider


class UpdateSecretRequest(BaseSecretRequest):
    pass


class CreateSecretRequest(BaseSecretRequest):
    pass


class APIKeyResponse(BaseModel):
    api_key: str

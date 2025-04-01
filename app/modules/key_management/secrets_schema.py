import re
from typing import Optional, Literal, List

from pydantic import BaseModel, field_validator


class BaseSecret(BaseModel):
    api_key: str
    model: str


class BaseSecretRequest(BaseModel):
    chat_config: Optional[BaseSecret] = None
    inference_config: Optional[BaseSecret] = None

    @field_validator("chat_config", "inference_config")
    @classmethod
    def at_least_one_config_provided(cls, v, info):
        # In the first field validation, check if either config is provided
        if info.field_name == "chat_config" and v is None:
            # Check if inference_config is also None
            data = info.data
            if data.get("inference_config") is None:
                raise ValueError(
                    "At least one of chat_config or inference_config must be provided"
                )
        return v

    @field_validator("chat_config")
    @classmethod
    def validate_chat_config(cls, v):
        if v is None:
            return v

        # Validate the model format
        parts = v.model.split("/")
        if len(parts) < 2:
            raise ValueError("Chat model must be in format 'provider/model_name'")

        provider = parts[0]
        if provider not in [
            "openai",
            "anthropic",
            "deepseek",
            "meta-llama",
            "gemini",
            "openrouter",
        ]:
            raise ValueError(f"Invalid provider in chat model: {provider}")

        # Validate the chat config API key
        if provider == "openai" and not cls.validate_openai_api_key_format(v.api_key):
            raise ValueError("Invalid OpenAI API key format in chat_config")
        elif provider == "anthropic" and not v.api_key.startswith("sk-ant-"):
            raise ValueError("Invalid Anthropic API key format in chat_config")
        elif provider in ["deepseek", "meta-llama"] and not v.api_key.startswith(
            "sk-or-"
        ):
            raise ValueError("Invalid OpenRouter API key format in chat_config")

        return v

    @field_validator("inference_config")
    @classmethod
    def validate_inference_config(cls, v):
        if v is None:
            return v

        # Validate the model format
        parts = v.model.split("/")
        if len(parts) < 2:
            raise ValueError("Inference model must be in format 'provider/model_name'")

        provider = parts[0]
        if provider not in [
            "openai",
            "anthropic",
            "deepseek",
            "meta-llama",
            "gemini",
            "openrouter",
        ]:
            raise ValueError(f"Invalid provider in inference model: {provider}")

        # Validate the inference config API key
        if provider == "openai" and not cls.validate_openai_api_key_format(v.api_key):
            raise ValueError("Invalid OpenAI API key format in inference_config")
        elif provider == "anthropic" and not v.api_key.startswith("sk-ant-"):
            raise ValueError("Invalid Anthropic API key format in inference_config")
        elif provider in ["deepseek", "meta-llama"] and not v.api_key.startswith(
            "sk-or-"
        ):
            raise ValueError("Invalid OpenRouter API key format in inference_config")

        return v

    @staticmethod
    def validate_openai_api_key_format(api_key: str) -> bool:
        pattern = r"^sk-[a-zA-Z0-9-]+$"
        proj_pattern = r"^sk-proj-[a-zA-Z0-9-]+$"
        return bool(re.match(pattern, api_key)) or bool(re.match(proj_pattern, api_key))


class UpdateSecretRequest(BaseSecretRequest):
    pass


class CreateSecretRequest(BaseSecretRequest):
    pass


class APIKeyResponse(BaseModel):
    api_key: str


# New schema classes for integration keys
class IntegrationKey(BaseModel):
    api_key: str
    service: Literal["linear", "notion"]


class CreateIntegrationKeyRequest(BaseModel):
    integration_keys: List[IntegrationKey]

    @field_validator("integration_keys")
    @classmethod
    def validate_keys(cls, v):
        if not v:
            raise ValueError("At least one integration key must be provided")
        return v


class UpdateIntegrationKeyRequest(CreateIntegrationKeyRequest):
    pass


class IntegrationKeyResponse(BaseModel):
    service: str
    api_key: str

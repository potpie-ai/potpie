import re
from typing import Literal, Optional

from pydantic import BaseModel, field_validator


class BaseSecretRequest(BaseModel):
    api_key: str
    provider: Literal[
        "openai",
        "anthropic",
        "deepseek",
        "meta-llama",
        "gemini",
        "openrouter",
    ]  # Added platform providers
    low_reasoning_model: Optional[str] = None
    high_reasoning_model: Optional[str] = None

    @staticmethod
    def validate_openai_api_key_format(api_key: str) -> bool:
        pattern = r"^sk-[a-zA-Z0-9-]+$"
        proj_pattern = r"^sk-proj-[a-zA-Z0-9-]+$"
        return bool(re.match(pattern, api_key)) or bool(re.match(proj_pattern, api_key))

    @field_validator("api_key")
    @classmethod
    def api_key_format(cls, v: str):
        if cls.validate_openai_api_key_format(v):
            return v
        elif v.startswith("sk-ant-"):
            return v
        elif v.startswith("sk-or-"):
            return v
        elif v.startswith("sk-"):  # Generic sk- check for future keys
            return v
        else:
            raise ValueError("Invalid API key format")

    @field_validator("provider")
    @classmethod
    def validate_provider_and_api_key(cls, provider: str, values):
        api_key = values.data.get("api_key")
        if provider == "openai":
            if not cls.validate_openai_api_key_format(api_key):
                raise ValueError("Invalid OpenAI API key format")
        elif provider == "anthropic":
            if not api_key.startswith("sk-ant-"):
                raise ValueError("Invalid Anthropic API key format")
        elif provider == "deepseek":
            if not api_key.startswith("sk-or-"):
                raise ValueError("Invalid OpenRouter API key format")
        elif provider == "meta-llama":  # Assuming meta-llama also uses openrouter keys
            if not api_key.startswith("sk-or-"):
                raise ValueError("Invalid OpenRouter API key format for Meta Llama")
        # For gemini and other potential future platform providers, no specific key validation for now,
        # can be extended here if needed.
        return provider


class UpdateSecretRequest(BaseSecretRequest):
    pass


class CreateSecretRequest(BaseSecretRequest):
    pass


class APIKeyResponse(BaseModel):
    api_key: str

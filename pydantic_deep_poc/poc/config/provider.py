"""OpenRouter-backed model for pydantic-ai / pydantic-deep."""

from __future__ import annotations

from typing import Any

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from poc.config.settings import (
    MODEL_NAME,
    MODEL_MAX_TOKENS,
    MODEL_PARALLEL_TOOL_CALLS,
    MODEL_TEMPERATURE,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)


def get_openrouter_provider() -> OpenAIProvider:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set (see .env.poc)")
    return OpenAIProvider(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )


def get_model() -> OpenAIChatModel:
    """Model instance for create_deep_agent(model=...) and smoke tests."""
    return OpenAIChatModel(MODEL_NAME, provider=get_openrouter_provider())


def get_model_settings() -> dict[str, Any]:
    """Shared model settings for the supervisor and worker subagents."""
    return {
        "temperature": MODEL_TEMPERATURE,
        "max_tokens": MODEL_MAX_TOKENS,
        "parallel_tool_calls": MODEL_PARALLEL_TOOL_CALLS,
    }

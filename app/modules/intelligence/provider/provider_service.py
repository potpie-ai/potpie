import os
from typing import List, Dict, Any, Union, AsyncGenerator, Optional
from pydantic import BaseModel
from pydantic_ai.models import Model
from litellm import litellm, AsyncOpenAI, acompletion
import instructor

from app.core.config_provider import config_provider
from app.modules.key_management.secret_manager import SecretManager
from app.modules.users.user_preferences_model import UserPreferences
from app.modules.utils.posthog_helper import PostHogClient
from app.modules.utils.logger import setup_logger

from .provider_schema import (
    ProviderInfo,
    GetProviderResponse,
    AvailableModelsResponse,
    AvailableModelOption,
    SetProviderRequest,
    ModelInfo,
)
from .llm_config import (
    LLMProviderConfig,
    build_llm_provider_config,
    get_config_for_model,
    DEFAULT_CHAT_MODEL,
    DEFAULT_INFERENCE_MODEL,
)
from .exceptions import UnsupportedProviderError

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from app.modules.intelligence.provider.openrouter_gemini_model import (
    OpenRouterGeminiModel,
)
from pydantic_ai.providers.anthropic import AnthropicProvider
from app.modules.intelligence.provider.anthropic_caching_model import (
    CachingAnthropicModel,
)

import random
import time
import asyncio
from functools import wraps

logger = setup_logger(__name__)

litellm.num_retries = 5  # Number of retries for rate limited requests

# Enable debug logging if LITELLM_DEBUG environment variable is set
_litellm_debug = os.getenv("LITELLM_DEBUG", "false").lower() in ("true", "1", "yes")
if _litellm_debug:
    litellm.set_verbose = True  # type: ignore
    litellm._turn_on_debug()  # type: ignore
    logger.info("LiteLLM debug logging ENABLED (LITELLM_DEBUG=true)")

OVERLOAD_ERROR_PATTERNS = {
    "anthropic": ["overloaded", "overloaded_error", "capacity", "rate limit exceeded"],
    "openai": [
        "rate_limit_exceeded",
        "capacity",
        "overloaded",
        "server_error",
        "timeout",
    ],
    "general": [
        "timeout",
        "insufficient capacity",
        "server_error",
        "internal_server_error",
    ],
}


class RetrySettings:
    """Configuration class for retry behavior"""

    def __init__(
        self,
        max_retries: int = 8,
        min_delay: float = 1.0,
        max_delay: float = 120.0,
        base_delay: float = 2.0,
        jitter_factor: float = 0.2,
        step_increase: float = 1.8,
        # Set what types of errors should be retried
        retry_on_timeout: bool = True,
        retry_on_overloaded: bool = True,
        retry_on_rate_limit: bool = True,
        retry_on_server_error: bool = True,
    ):
        self.max_retries = max_retries
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.base_delay = base_delay
        self.jitter_factor = jitter_factor
        self.step_increase = step_increase
        self.retry_on_timeout = retry_on_timeout
        self.retry_on_overloaded = retry_on_overloaded
        self.retry_on_rate_limit = retry_on_rate_limit
        self.retry_on_server_error = retry_on_server_error


def identify_provider_from_error(error: Exception) -> str:
    """Identify the provider from an exception"""
    error_str = str(error).lower()


    # Try to identify provider from error message
    for provider in ["anthropic", "openai", "cohere", "azure"]:
        if provider.lower() in error_str.lower():
            return provider

    return "unknown"



def is_recoverable_error(error: Exception, settings: RetrySettings) -> bool:
    """Determine if an error is recoverable based on retry settings"""
    error_str = str(error).lower()
    provider = identify_provider_from_error(error)


    # Check for timeout errors
    if settings.retry_on_timeout and "timeout" in error_str:
        return True

    # Check for overloaded errors
    if settings.retry_on_overloaded:
        overload_patterns = (
            OVERLOAD_ERROR_PATTERNS.get(provider, [])
            + OVERLOAD_ERROR_PATTERNS["general"]
        )
        if any(pattern in error_str for pattern in overload_patterns):
            return True

    # Check for rate limit errors
    if settings.retry_on_rate_limit and any(
        limit_pattern in error_str
        for limit_pattern in [
            "rate limit",
            "rate_limit",
            "ratelimit",
            "requests per minute",
        ]
    ):
        return True

    # Check for server errors
    if settings.retry_on_server_error and any(
        server_err in error_str
        for server_err in [
            "server_error",
            "internal_server_error",
            "500",
            "502",
            "503",
            "504",
        ]
    ):
        return True

    return False


def calculate_backoff_time(retry_count: int, settings: RetrySettings) -> float:
    """Calculate exponential backoff with jitter"""
    # Calculate base exponential backoff
    delay = min(
        settings.max_delay, settings.base_delay * (settings.step_increase**retry_count)
    )

    # Add jitter to avoid thundering herd problem
    jitter = random.uniform(1 - settings.jitter_factor, 1 + settings.jitter_factor)


    # Ensure we stay within our bounds
    final_delay = max(settings.min_delay, min(settings.max_delay, delay * jitter))

    return final_delay


# Create a custom retry function for litellm
def custom_litellm_retry_handler(retry_count: int, exception: Exception) -> bool:
    """
    Custom retry handler for litellm's built-in retry mechanism
    This gets registered with litellm.custom_retry_fn
    """
    # Default settings for litellm's built-in retry
    settings = RetrySettings(max_retries=litellm.num_retries or 5)

    if not is_recoverable_error(exception, settings):
        # If it's not a recoverable error, don't retry
        return False

    delay = calculate_backoff_time(retry_count, settings)


    provider = identify_provider_from_error(exception)
    logger.warning(
        f"{provider.capitalize()} API error: {str(exception)}. "
        f"Retry {retry_count}/{settings.max_retries}, "
        f"waiting {delay:.2f}s before next attempt..."
    )

    time.sleep(delay)
    return True


# Decorator for robust LLM calls with advanced error handling
def robust_llm_call(settings: Optional[RetrySettings] = None):
    """
    Decorator for robust handling of LLM API calls with exponential backoff
    """
    if settings is None:
        settings = RetrySettings()

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            last_exception = None

            while retries <= settings.max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not is_recoverable_error(e, settings):
                        # If it's not a recoverable error, just raise
                        raise

                    provider = identify_provider_from_error(e)

                    if retries >= settings.max_retries:
                        logger.exception(
                            "Max retries exceeded for API call",
                            provider=provider,
                            retries=retries,
                            max_retries=settings.max_retries,
                        )
                        raise

                    delay = calculate_backoff_time(retries, settings)


                    logger.warning(
                        f"{provider.capitalize()} API error: {str(e)}. "
                        f"Retry {retries + 1}/{settings.max_retries}, "
                        f"waiting {delay:.2f}s before next attempt..."
                    )


                    await asyncio.sleep(delay)
                    retries += 1

            # This should never be reached due to the raise in the loop,
            # but included for clarity
            if last_exception is not None:
                raise last_exception
            raise RuntimeError("Unexpected error: retries exhausted without exception")


        return wrapper

    return decorator


def sanitize_messages_for_tracing(messages: list) -> list:
    """
    Sanitize messages to prevent OpenTelemetry encoding errors.
    Converts None content values to empty strings to avoid:
    'Invalid type <class 'NoneType'> of value None' errors.


    Args:
        messages: List of message dictionaries with 'role' and 'content' keys

    Returns:
        List of sanitized messages with None content converted to empty strings
    """
    sanitized = []
    for idx, msg in enumerate(messages):
        try:
            if isinstance(msg, dict):
                sanitized_msg = msg.copy()
                # Convert None content to empty string for OpenTelemetry compatibility
                if "content" in sanitized_msg and sanitized_msg["content"] is None:
                    sanitized_msg["content"] = ""
                    logger.debug(
                        f"Sanitized message {idx}: converted None content to empty string"
                    )
                # Handle nested content structures (e.g., multimodal messages)
                elif "content" in sanitized_msg and isinstance(
                    sanitized_msg["content"], list
                ):
                    sanitized_content = []
                    for item_idx, item in enumerate(sanitized_msg["content"]):
                        if item is None:
                            # Skip None items in content list
                            logger.debug(
                                f"Sanitized message {idx}: skipping None item at index {item_idx} in content list"
                            )
                            continue
                        elif isinstance(item, dict):
                            sanitized_item = item.copy()
                            # Handle None values in nested dicts
                            for key, value in sanitized_item.items():
                                if value is None:
                                    sanitized_item[key] = ""
                                    logger.debug(
                                        f"Sanitized message {idx}: converted None value for key '{key}' to empty string"
                                    )
                            sanitized_content.append(sanitized_item)
                        else:
                            sanitized_content.append(item)
                    sanitized_msg["content"] = sanitized_content
                # Also sanitize other fields that might be None
                for key, value in sanitized_msg.items():
                    if value is None and key != "content":
                        sanitized_msg[key] = ""
                        logger.debug(
                            f"Sanitized message {idx}: converted None value for key '{key}' to empty string"
                        )
                sanitized.append(sanitized_msg)
            else:
                sanitized.append(msg)
        except Exception as e:
            # Log error but continue processing - don't break on one bad message
            logger.warning(
                f"Error sanitizing message {idx}: {e}. Message will be included as-is.",
                exc_info=True,
            )
            sanitized.append(msg)
    return sanitized


# Available models with their metadata
AVAILABLE_MODELS = [
    AvailableModelOption(
        id="openai/gpt-5.2",
        name="GPT-5.2",
        description="OpenAI's latest model for complex tasks with large context",
        provider="openai",
        is_chat_model=True,
        is_inference_model=False,
    ),
    AvailableModelOption(
        id="openai/gpt-5.1",
        name="GPT-5.1",
        description="OpenAI's previous flagship model",
        provider="openai",
        is_chat_model=True,
        is_inference_model=False,
    ),
    AvailableModelOption(
        id="openai/gpt-5-mini",
        name="GPT-5 Mini",
        description="Smaller model for fast, lightweight tasks",
        provider="openai",
        is_chat_model=False,
        is_inference_model=True,
    ),
    AvailableModelOption(
        id="anthropic/claude-sonnet-4.5-20250929",
        name="Claude Sonnet 4.5",
        description="Best model for complex agents and coding",
        provider="anthropic",
        is_chat_model=True,
        is_inference_model=False,
    ),
    AvailableModelOption(
        id="anthropic/claude-haiku-4-5-20251001",
        name="Claude Haiku 4.5",
        description="Faster, even surpasses Claude Sonnet 4 at certain tasks",
        provider="anthropic",
        is_chat_model=True,
        is_inference_model=True,
    ),
    AvailableModelOption(
        id="anthropic/claude-opus-4.5-20251101",
        name="Claude 4.5 Opus",
        description="Latest Claude Opus tier for maximum reasoning depth",
        provider="anthropic",
        is_chat_model=True,
        is_inference_model=False,
    ),
    AvailableModelOption(
        id="openrouter/deepseek/deepseek-chat-v3-0324",
        name="DeepSeek V3",
        description="DeepSeek's latest chat model",
        provider="deepseek",
        is_chat_model=True,
        is_inference_model=True,
    ),
    AvailableModelOption(
        id="openrouter/meta-llama/llama-3.3-70b-instruct",
        name="Llama 3.3 70B",
        description="Meta's latest Llama model",
        provider="meta-llama",
        is_chat_model=True,
        is_inference_model=True,
    ),
    AvailableModelOption(
        id="openrouter/google/gemini-2.0-flash-001",
        name="Gemini 2.0 Flash",
        description="Google's Gemini model optimized for speed",
        provider="gemini",
        is_chat_model=True,
        is_inference_model=True,
    ),
    AvailableModelOption(
        id="openrouter/google/gemini-2.5-pro-preview",
        name="Gemini 2.5 Pro",
        description="Google's pro Gemini model",
        provider="gemini",
        is_chat_model=True,
        is_inference_model=True,
    ),
    AvailableModelOption(
        id="openrouter/google/gemini-3.1-pro-preview",
        name="Gemini 3.1 Pro Preview",
        description="Latest Gemini 3.1 Pro capabilities",
        provider="gemini",
        is_chat_model=True,
        is_inference_model=True,
    ),
    AvailableModelOption(
        id="openrouter/google/gemini-3-flash-preview",
        name="Gemini 3 Flash",
        description="Google's Gemini 3 Flash model optimized for speed",
        provider="gemini",
        is_chat_model=True,
        is_inference_model=True,
    ),
    AvailableModelOption(
        id="openrouter/z-ai/glm-4.7",
        name="Z AI GLM 4.7",
        description="Latest Z AI model",
        provider="zai",
        is_chat_model=True,
        is_inference_model=True,
    ),
    AvailableModelOption(
        id="openrouter/moonshotai/kimi-k2.5",
        name="Kimi K2.5 (Moonshot)",
        description="Moonshot AI Kimi K2.5",
        provider="moonshot",
        is_chat_model=True,
        is_inference_model=True,
    ),
    AvailableModelOption(
        id="openrouter/minimax/minimax-m2.5",
        name="MiniMax M2.5",
        description="SOTA model for coding and productivity, strong on SWE-Bench and BrowseComp",
        provider="minimax",
        is_chat_model=True,
        is_inference_model=True,
    ),
]

# Extract unique platform providers from the available models
PLATFORM_PROVIDERS = list(
    {model.provider for model in AVAILABLE_MODELS}
    | {
        get_config_for_model(model.id).get("auth_provider", model.provider)
        for model in AVAILABLE_MODELS
    }
)


class ProviderService:
    def __init__(self, db, user_id: str):
        litellm.modify_params = True
        self.db = db
        self.user_id = user_id

        # Cache for API keys to avoid repeated secret manager checks
        # Key: provider name, Value: API key (or None if not found)
        self._api_key_cache: Dict[str, Optional[str]] = {}

        # Load user preferences
        user_pref = db.query(UserPreferences).filter_by(user_id=user_id).first()
        user_config = (
            user_pref.config if user_pref and user_pref.config else {}
        )

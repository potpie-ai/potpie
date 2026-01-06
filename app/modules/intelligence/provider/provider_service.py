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
                        f"Retry {retries+1}/{settings.max_retries}, "
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
        id="openai/gpt-4.1",
        name="GPT-4.1",
        description="OpenAI's latest model for complex tasks with large context",
        provider="openai",
        is_chat_model=True,
        is_inference_model=False,
    ),
    AvailableModelOption(
        id="openai/gpt-4o",
        name="GPT-4o",
        description="High-intelligence model for complex tasks",
        provider="openai",
        is_chat_model=True,
        is_inference_model=False,
    ),
    AvailableModelOption(
        id="openai/gpt-4.1-mini",
        name="GPT-4.1 Mini",
        description="Smaller model for fast, lightweight tasks",
        provider="openai",
        is_chat_model=False,
        is_inference_model=True,
    ),
    AvailableModelOption(
        id="openai/o4-mini",
        name="O4 mini",
        description="reasoning model",
        provider="openai",
        is_chat_model=True,
        is_inference_model=True,
    ),
    AvailableModelOption(
        id="anthropic/claude-sonnet-4-5-20250929",
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
        id="anthropic/claude-opus-4-1-20250805",
        name="Claude Opus 4.1",
        description="Exceptional model for specialized complex tasks",
        provider="anthropic",
        is_chat_model=True,
        is_inference_model=False,
    ),
    AvailableModelOption(
        id="anthropic/claude-sonnet-4-20250514",
        name="Claude Sonnet 4",
        description="Faster, more efficient Claude model for code generation",
        provider="anthropic",
        is_chat_model=True,
        is_inference_model=False,
    ),
    AvailableModelOption(
        id="anthropic/claude-3-7-sonnet-20250219",
        name="Claude Sonnet 3.7",
        description="Highest level of intelligence and capability with toggleable extended thinking",
        provider="anthropic",
        is_chat_model=True,
        is_inference_model=False,
    ),
    AvailableModelOption(
        id="anthropic/claude-3-5-haiku-20241022",
        name="Claude Haiku 3.5",
        description="Faster, more efficient Claude model",
        provider="anthropic",
        is_chat_model=False,
        is_inference_model=True,
    ),
    AvailableModelOption(
        id="anthropic/claude-opus-4-5-20251101",
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
        description="Google's Latest pro Gemini model",
        provider="gemini",
        is_chat_model=True,
        is_inference_model=True,
    ),
    AvailableModelOption(
        id="openrouter/google/gemini-3-pro-preview",
        name="Gemini 3 Pro Preview",
        description="Latest Gemini 3 Pro capabilities via OpenRouter",
        provider="gemini",
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
            user_pref.preferences if user_pref and user_pref.preferences else {}
        )

        # Create configurations based on user input (or fallback defaults)
        self.chat_config = build_llm_provider_config(user_config, config_type="chat")
        self.inference_config = build_llm_provider_config(
            user_config, config_type="inference"
        )

        self.retry_settings = RetrySettings(
            max_retries=8, base_delay=2.0, max_delay=120.0
        )

    @classmethod
    def create(cls, db, user_id: str):
        return cls(db, user_id)

    @classmethod
    def create_from_config(
        cls,
        db,
        user_id: str,
        *,
        provider: str = "openai",
        api_key: Optional[str] = None,
        chat_model: Optional[str] = None,
        inference_model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> "ProviderService":
        """Factory method that accepts explicit config for library usage.

        This bypasses environment variables and user preferences,
        using only the explicitly provided configuration.

        Args:
            db: Database session
            user_id: User identifier
            provider: LLM provider name (openai, anthropic, ollama, etc.)
            api_key: API key for the provider
            chat_model: Model to use for chat (e.g., "openai/gpt-4o")
            inference_model: Model to use for inference (e.g., "openai/gpt-4.1-mini")
            base_url: Optional base URL for self-hosted models

        Returns:
            Configured ProviderService instance
        """
        instance = object.__new__(cls)
        litellm.modify_params = True
        instance.db = db
        instance.user_id = user_id

        resolved_chat_model = (
            chat_model or f"{provider}/gpt-4o"
            if provider == "openai"
            else chat_model or DEFAULT_CHAT_MODEL
        )
        resolved_inference_model = (
            inference_model or f"{provider}/gpt-4.1-mini"
            if provider == "openai"
            else inference_model or DEFAULT_INFERENCE_MODEL
        )

        chat_config_data = get_config_for_model(resolved_chat_model)
        inference_config_data = get_config_for_model(resolved_inference_model)

        instance.chat_config = LLMProviderConfig(
            provider=chat_config_data.get("provider", provider),
            model=resolved_chat_model,
            default_params=dict(
                chat_config_data.get("default_params", {"temperature": 0.3})
            ),
            capabilities=chat_config_data.get("capabilities", {}),
            base_url=base_url or chat_config_data.get("base_url"),
            api_version=chat_config_data.get("api_version"),
            auth_provider=chat_config_data.get("auth_provider", provider),
        )

        instance.inference_config = LLMProviderConfig(
            provider=inference_config_data.get("provider", provider),
            model=resolved_inference_model,
            default_params=dict(
                inference_config_data.get("default_params", {"temperature": 0.3})
            ),
            capabilities=inference_config_data.get("capabilities", {}),
            base_url=base_url or inference_config_data.get("base_url"),
            api_version=inference_config_data.get("api_version"),
            auth_provider=inference_config_data.get("auth_provider", provider),
        )

        instance._explicit_api_key = api_key
        instance.retry_settings = RetrySettings(
            max_retries=8, base_delay=2.0, max_delay=120.0
        )

        return instance

    async def list_available_llms(self) -> List[ProviderInfo]:
        # Get unique providers from available models
        providers = {
            model.provider: ProviderInfo(
                id=model.provider,
                name=model.provider,
                description=f"Provider for {model.provider} models",
            )
            for model in AVAILABLE_MODELS
        }
        return list(providers.values())

    async def list_available_models(self) -> AvailableModelsResponse:
        return AvailableModelsResponse(models=AVAILABLE_MODELS)

    async def set_global_ai_provider(self, user_id: str, request: SetProviderRequest):
        """Update the global AI provider configuration with new model selections."""
        preferences = self.db.query(UserPreferences).filter_by(user_id=user_id).first()

        if not preferences:
            preferences = UserPreferences(user_id=user_id, preferences={})
            self.db.add(preferences)
        elif preferences.preferences is None:
            preferences.preferences = {}

        # Create a new dictionary with existing preferences
        updated_preferences = (
            preferences.preferences.copy() if preferences.preferences else {}
        )

        # Update chat model if provided
        if request.chat_model:
            updated_preferences["chat_model"] = request.chat_model
            self.chat_config = build_llm_provider_config(updated_preferences, "chat")

        # Update inference model if provided
        if request.inference_model:
            updated_preferences["inference_model"] = request.inference_model
            self.inference_config = build_llm_provider_config(
                updated_preferences, "inference"
            )

        # Explicitly assign the new dictionary to mark it as modified
        preferences.preferences = updated_preferences

        # Ensure changes are flushed to the database
        self.db.flush()
        self.db.commit()
        self.db.refresh(preferences)

        # Send analytics event
        if request.chat_model:
            PostHogClient().send_event(
                user_id, "chat_model_change_event", {"model": request.chat_model}
            )
        if request.inference_model:
            PostHogClient().send_event(
                user_id,
                "inference_model_change_event",
                {"model": request.inference_model},
            )

        return {"message": "AI provider configuration updated successfully"}

    def _get_api_key(self, provider: str) -> str:
        """Get API key for the specified provider. Caches the result per provider for the session."""
        # Check explicit API key first (for library usage via create_from_config)
        if hasattr(self, "_explicit_api_key") and self._explicit_api_key:
            return self._explicit_api_key

        # Check cache first
        if provider in self._api_key_cache:
            cached_key = self._api_key_cache[provider]
            if cached_key is not None:
                return cached_key
            # If cached as None, we already checked and it's not available
            return None

        # Check environment variable first (fastest)
        env_key = os.getenv("LLM_API_KEY", None)
        if env_key:
            self._api_key_cache[provider] = env_key
            return env_key

        # Try to get from secret manager (only once per provider per session)
        try:
            secret = SecretManager.get_secret(provider, self.user_id, self.db)
            self._api_key_cache[provider] = secret
            return secret
        except Exception as e:
            if "404" in str(e):
                # Try provider-specific env var as fallback
                env_key = os.getenv(f"{provider.upper()}_API_KEY")
                if env_key:
                    self._api_key_cache[provider] = env_key
                    return env_key
                # Cache None to indicate we've checked and it's not available
                self._api_key_cache[provider] = None
                return None
            raise e

    def _build_llm_params(self, config: LLMProviderConfig) -> Dict[str, Any]:
        """Build a dictionary of parameters for LLM initialization."""
        api_key = self._get_api_key(config.auth_provider)
        if not api_key and config.auth_provider == "ollama":
            api_key = os.environ.get("OLLAMA_API_KEY", "ollama")
        if not api_key:
            api_key = os.environ.get("LLM_API_KEY", api_key)

        params = config.get_llm_params(api_key)

        if config.base_url:
            base_url = config.base_url
            if config.auth_provider == "ollama":
                base_url = base_url.rstrip("/")
                if base_url.endswith("/v1"):
                    base_url = base_url[:-3]
            params["base_url"] = base_url
        elif config.auth_provider == "ollama":
            params["base_url"] = os.environ.get(
                "LLM_API_BASE", "http://localhost:11434"
            )
        if config.api_version:
            params["api_version"] = config.api_version

        # Filter out falsy values litellm would not expect
        return {key: value for key, value in params.items() if value is not None}

    def _build_config_for_model_identifier(
        self, model_identifier: str
    ) -> LLMProviderConfig:
        """Create a provider config for a specific model identifier."""
        config_data = get_config_for_model(model_identifier).copy()
        default_params = dict(config_data.get("default_params", {}))

        return LLMProviderConfig(
            provider=config_data["provider"],
            model=model_identifier,
            default_params=default_params,
            capabilities=config_data.get("capabilities", {}),
            base_url=config_data.get("base_url"),
            api_version=config_data.get("api_version"),
            auth_provider=config_data.get("auth_provider"),
        )

    async def get_global_ai_provider(self, user_id: str) -> GetProviderResponse:
        """Get the current global AI provider configuration."""
        try:
            user_pref = (
                self.db.query(UserPreferences)
                .filter(UserPreferences.user_id == user_id)
                .first()
            )

            # Get current models from preferences or environment
            chat_model_id = (
                os.environ.get("CHAT_MODEL")
                or (
                    user_pref.preferences.get("chat_model")
                    if user_pref and user_pref.preferences
                    else None
                )
                or "openai/gpt-4o"
            )

            inference_model_id = (
                os.environ.get("INFERENCE_MODEL")
                or (
                    user_pref.preferences.get("inference_model")
                    if user_pref and user_pref.preferences
                    else None
                )
                or "openai/gpt-4.1-mini"
            )

            # Default values
            chat_provider = chat_model_id.split("/")[0] if chat_model_id else ""
            chat_model_name = chat_model_id

            inference_provider = (
                inference_model_id.split("/")[0] if inference_model_id else ""
            )
            inference_model_name = inference_model_id

            # Find matching model in AVAILABLE_MODELS to get proper names
            for model in AVAILABLE_MODELS:
                if model.id == chat_model_id:
                    chat_model_name = model.name
                    chat_provider = model.provider

                if model.id == inference_model_id:
                    inference_model_name = model.name
                    inference_provider = model.provider

            # Create response with nested ModelInfo objects
            return GetProviderResponse(
                chat_model=ModelInfo(
                    provider=chat_provider, id=chat_model_id, name=chat_model_name
                ),
                inference_model=ModelInfo(
                    provider=inference_provider,
                    id=inference_model_id,
                    name=inference_model_name,
                ),
            )
        except Exception as e:
            logger.exception("Error getting global AI provider")
            raise e

    def supports_pydantic(self, config_type: str = "chat") -> bool:
        """Return True when the active model supports the pydantic-ai stack."""
        config = self.chat_config if config_type == "chat" else self.inference_config
        return config.capabilities.get("supports_pydantic", False)

    @robust_llm_call()
    async def call_llm_with_specific_model(
        self,
        model_identifier: str,
        messages: list,
        output_schema: Optional[BaseModel] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, AsyncGenerator[str, None], Any]:
        """Call LLM with a specific model identifier (e.g., 'openrouter/perplexity/sonar')."""
        # Sanitize messages to prevent OpenTelemetry encoding errors
        messages = sanitize_messages_for_tracing(messages)

        # Build configuration for the specific model
        config = self._build_config_for_model_identifier(model_identifier)

        # Build parameters using the config object
        params = self._build_llm_params(config)

        # Override with any additional parameters
        params.update(kwargs)

        routing_provider = config.provider

        try:
            if output_schema:
                # Use structured output with instructor
                request_kwargs = {
                    key: params[key]
                    for key in ("api_key", "base_url", "api_version")
                    if key in params
                }

                if config.provider == "ollama":
                    # use openai client to call ollama because of https://github.com/BerriAI/litellm/issues/7355
                    ollama_base_root = (
                        params.get("base_url")
                        or config.base_url
                        or os.environ.get("LLM_API_BASE")
                        or "http://localhost:11434"
                    )
                    ollama_base_url = ollama_base_root.rstrip("/") + "/v1"
                    ollama_api_key = params.get("api_key") or os.environ.get(
                        "OLLAMA_API_KEY", "ollama"
                    )
                    client = instructor.from_openai(
                        AsyncOpenAI(base_url=ollama_base_url, api_key=ollama_api_key),
                        mode=instructor.Mode.JSON,
                    )
                    ollama_request_kwargs = {
                        key: value
                        for key, value in request_kwargs.items()
                        if key not in {"base_url", "api_key", "api_version"}
                    }
                    response = await client.chat.completions.create(
                        model=params["model"].split("/")[-1],
                        messages=messages,
                        response_model=output_schema,
                        temperature=params.get("temperature", 0.3),
                        max_tokens=params.get("max_tokens"),
                        **ollama_request_kwargs,
                    )
                else:
                    client = instructor.from_litellm(
                        acompletion, mode=instructor.Mode.JSON
                    )
                    response = await client.chat.completions.create(
                        model=params["model"],
                        messages=messages,
                        response_model=output_schema,
                        strict=True,
                        temperature=params.get("temperature", 0.3),
                        max_tokens=params.get("max_tokens"),
                        **request_kwargs,
                    )
                return response
            else:
                # Regular text completion
                if stream:

                    async def generator() -> AsyncGenerator[str, None]:
                        response = await acompletion(
                            messages=messages, stream=True, **params
                        )
                        async for chunk in response:
                            yield chunk.choices[0].delta.content or ""

                    return generator()
                else:
                    response = await acompletion(messages=messages, **params)
                    return response.choices[0].message.content
        except Exception as e:
            logger.exception(
                "Error calling LLM",
                model_identifier=model_identifier,
                provider=routing_provider,
            )
            raise e

    @robust_llm_call()  # Apply the robust_llm_call decorator
    async def call_llm(
        self, messages: list, stream: bool = False, config_type: str = "chat"
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Call LLM with the specified messages with robust error handling."""
        # Sanitize messages to prevent OpenTelemetry encoding errors
        messages = sanitize_messages_for_tracing(messages)

        # Select the appropriate config based on config_type
        config = self.chat_config if config_type == "chat" else self.inference_config

        # Build parameters using the config object
        params = self._build_llm_params(config)
        routing_provider = config.provider

        # Handle streaming response if requested
        try:
            if stream:

                async def generator() -> AsyncGenerator[str, None]:
                    response = await acompletion(
                        messages=messages, stream=True, **params
                    )
                    async for chunk in response:
                        yield chunk.choices[0].delta.content or ""

                return generator()
            else:
                response = await acompletion(messages=messages, **params)
                return response.choices[0].message.content
        except Exception as e:
            logger.exception("Error calling LLM", provider=routing_provider)
            raise e

    @robust_llm_call()
    async def call_llm_with_structured_output(
        self, messages: list, output_schema: BaseModel, config_type: str = "chat"
    ) -> Any:
        """Call LLM and parse the response into a structured output using a Pydantic model."""
        # Sanitize messages to prevent OpenTelemetry encoding errors
        messages = sanitize_messages_for_tracing(messages)

        # Select the appropriate config
        config = self.chat_config if config_type == "chat" else self.inference_config

        # Build parameters
        params = self._build_llm_params(config)
        routing_provider = config.provider

        request_kwargs = {
            key: params[key]
            for key in ("api_key", "base_url", "api_version")
            if key in params
        }

        try:
            if config.provider == "ollama":
                # use openai client to call ollama because of https://github.com/BerriAI/litellm/issues/7355
                ollama_base_root = (
                    params.get("base_url")
                    or config.base_url
                    or os.environ.get("LLM_API_BASE")
                    or "http://localhost:11434"
                )
                ollama_base_url = ollama_base_root.rstrip("/") + "/v1"
                ollama_api_key = params.get("api_key") or os.environ.get(
                    "OLLAMA_API_KEY", "ollama"
                )
                client = instructor.from_openai(
                    AsyncOpenAI(base_url=ollama_base_url, api_key=ollama_api_key),
                    mode=instructor.Mode.JSON,
                )
                ollama_request_kwargs = {
                    key: value
                    for key, value in request_kwargs.items()
                    if key not in {"base_url", "api_key", "api_version"}
                }
                response = await client.chat.completions.create(
                    model=params["model"].split("/")[-1],
                    messages=messages,
                    response_model=output_schema,
                    temperature=params.get("temperature", 0.3),
                    max_tokens=params.get("max_tokens"),
                    **ollama_request_kwargs,
                )
            else:
                client = instructor.from_litellm(acompletion, mode=instructor.Mode.JSON)
                response = await client.chat.completions.create(
                    model=params["model"],
                    messages=messages,
                    response_model=output_schema,
                    strict=True,
                    temperature=params.get("temperature", 0.3),
                    max_tokens=params.get("max_tokens"),
                    **request_kwargs,
                )
            return response
        except Exception as e:
            logger.exception("LLM call with structured output failed")
            raise e

    @robust_llm_call()
    async def call_llm_multimodal(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[Dict[str, Dict[str, Union[str, int]]]] = None,
        stream: bool = False,
        config_type: str = "chat",
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Call LLM with multimodal support (text + images)"""
        # Sanitize messages to prevent OpenTelemetry encoding errors
        messages = sanitize_messages_for_tracing(messages)

        # Check if multimodal is enabled
        if not config_provider.get_is_multimodal_enabled():
            logger.info("Multimodal disabled - falling back to text-only processing")
            return await self.call_llm(messages, stream=stream, config_type=config_type)

        # If no images provided, use standard text-only call
        if not images:
            return await self.call_llm(messages, stream=stream, config_type=config_type)

        # Original multimodal logic continues...
        # Select the appropriate config based on config_type
        config = self.chat_config if config_type == "chat" else self.inference_config

        # Build parameters using the config object
        params = self._build_llm_params(config)
        routing_provider = config.provider

        # Validate and filter images before processing
        if images:
            validated_images = self._validate_images_for_multimodal(images)
            if validated_images:
                messages = self._format_multimodal_messages(
                    messages, validated_images, routing_provider
                )
                logger.info(
                    f"Using {len(validated_images)} validated images out of {len(images)} provided for provider {routing_provider}"
                )
            else:
                logger.warning(
                    "No valid images after validation, proceeding with text-only"
                )
                images = None

        # Handle streaming response if requested
        try:
            if stream:

                async def generator() -> AsyncGenerator[str, None]:
                    response = await acompletion(
                        messages=messages, stream=True, **params
                    )
                    async for chunk in response:
                        yield chunk.choices[0].delta.content or ""

                return generator()
            else:
                response = await acompletion(messages=messages, **params)
                return response.choices[0].message.content
        except Exception as e:
            logger.exception("Error calling multimodal LLM", provider=routing_provider)
            raise e

    def _format_multimodal_messages(
        self,
        messages: List[Dict[str, Any]],
        images: Dict[str, Dict[str, Union[str, int]]],
        provider: str,
    ) -> List[Dict[str, Any]]:
        """Format messages for provider-specific multimodal format"""
        if not images:
            return messages

        formatted_messages = []

        for message in messages:
            if (
                message.get("role") == "user"
                and len(formatted_messages) == len(messages) - 1
            ):
                # This is the last user message - add images to it
                formatted_message = self._format_multimodal_message(
                    message, images, provider
                )
                formatted_messages.append(formatted_message)
            else:
                formatted_messages.append(message)

        return formatted_messages

    def _format_multimodal_message(
        self,
        message: Dict[str, Any],
        images: Dict[str, Dict[str, Union[str, int]]],
        provider: str,
    ) -> Dict[str, Any]:
        """Format a single message for provider-specific multimodal format"""
        text_content = message.get("content", "")

        if provider == "openai":
            return self._format_openai_multimodal_message(text_content, images)
        elif provider == "anthropic":
            return self._format_anthropic_multimodal_message(text_content, images)
        elif provider == "gemini":
            return self._format_gemini_multimodal_message(text_content, images)
        else:
            # Fallback to OpenAI format for unknown providers
            logger.warning(
                f"Unknown provider {provider}, using OpenAI format for multimodal"
            )
            return self._format_openai_multimodal_message(text_content, images)

    def _format_openai_multimodal_message(
        self, text: str, images: Dict[str, Dict[str, Union[str, int]]]
    ) -> Dict[str, Any]:
        """Format message for OpenAI GPT-4V format"""
        content = [{"type": "text", "text": text}]

        for attachment_id, image_data in images.items():
            mime_type = image_data.get("mime_type", "image/jpeg")
            base64_data = image_data["base64"]

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_data}",
                        "detail": "high",  # Use high detail for better analysis
                    },
                }
            )

        return {"role": "user", "content": content}

    def _format_anthropic_multimodal_message(
        self, text: str, images: Dict[str, Dict[str, Union[str, int]]]
    ) -> Dict[str, Any]:
        """Format message for Anthropic Claude Vision format"""
        content = []

        # Add images first for Claude
        for attachment_id, image_data in images.items():
            mime_type = image_data.get("mime_type", "image/jpeg")
            base64_data = image_data["base64"]

            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64_data,
                    },
                }
            )

        # Add text content
        content.append({"type": "text", "text": text})

        return {"role": "user", "content": content}

    def _format_gemini_multimodal_message(
        self, text: str, images: Dict[str, Dict[str, Union[str, int]]]
    ) -> Dict[str, Any]:
        """Format message for Google Gemini Vision format (uses OpenAI-compatible format via OpenRouter)"""
        return self._format_openai_multimodal_message(text, images)

    def _validate_images_for_multimodal(
        self, images: Dict[str, Dict[str, Union[str, int]]]
    ) -> Dict[str, Dict[str, Union[str, int]]]:
        """Validate images before sending to multimodal LLM to reduce hallucinations"""
        validated_images = {}

        for img_id, img_data in images.items():
            try:
                # Check required fields
                if "base64" not in img_data or not img_data["base64"]:
                    logger.warning(
                        f"Skipping image {img_id}: missing or empty base64 data"
                    )
                    continue

                base64_data = str(img_data["base64"])

                # Check base64 data length (reasonable bounds)
                if len(base64_data) < 100:  # Too small to be a valid image
                    logger.warning(
                        f"Skipping image {img_id}: base64 data too small ({len(base64_data)} chars)"
                    )
                    continue

                if (
                    len(base64_data) > 10_000_000
                ):  # Over ~7MB base64 (too large for most APIs)
                    logger.warning(
                        f"Skipping image {img_id}: base64 data too large ({len(base64_data)} chars)"
                    )
                    continue

                # Check MIME type
                mime_type = img_data.get("mime_type", "")
                supported_types = ["image/jpeg", "image/png", "image/webp", "image/gif"]
                if mime_type not in supported_types:
                    logger.warning(
                        f"Skipping image {img_id}: unsupported MIME type {mime_type}"
                    )
                    continue

                # Basic base64 validation (should start with valid characters)
                if (
                    not base64_data.replace("+", "")
                    .replace("/", "")
                    .replace("=", "")
                    .isalnum()
                ):
                    logger.warning(f"Skipping image {img_id}: invalid base64 encoding")
                    continue

                # Image passed validation
                validated_images[img_id] = img_data
                logger.debug(
                    f"Image {img_id} passed validation ({len(base64_data)} chars, {mime_type})"
                )

            except Exception:
                logger.exception("Error validating image", img_id=img_id)
                continue

        logger.info(
            f"Validated {len(validated_images)} out of {len(images)} images for multimodal processing"
        )
        return validated_images

    def is_vision_model(self, config_type: str = "chat") -> bool:
        """Check if the current model supports vision/multimodal inputs"""

        # If multimodal is disabled globally, no models support vision
        if not config_provider.get_is_multimodal_enabled():
            return False

        # Original vision detection logic continues...
        config = self.chat_config if config_type == "chat" else self.inference_config
        model_name = config.model.lower()

        logger.info(f"Checking if model '{config.model}' supports vision capabilities")

        # Known vision models - expanded list
        vision_models = [
            # OpenAI models
            "gpt-4-vision",
            "gpt-4v",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "o4-mini",
            # Anthropic models
            "claude-3",
            "claude-3-sonnet",
            "claude-3-opus",
            "claude-3-haiku",
            "claude-sonnet-4",
            "claude-opus-4-1",
            "claude-haiku-4-5",
            "claude-sonnet-4-5",
            # Google models
            "gemini-pro-vision",
            "gemini-1.5",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-2.0",
            "gemini-2.0-flash",
            "gemini-2.5",
            "gemini-2.5-pro",
            "gemini-3",
            "gemini-3-pro",
            "gemini-ultra",
            # Other models that might support vision
            "deepseek-chat",
            "llama-3.3",
            "llama-3.3-70b",
            "llama-3.3-8b",
        ]

        is_vision = any(vision_model in model_name for vision_model in vision_models)
        logger.info(f"Model '{config.model}' vision support: {is_vision}")

        if not is_vision:
            logger.warning(
                f"Model '{config.model}' may not support vision. Known vision models: {vision_models}"
            )

        return is_vision

    def get_pydantic_model(
        self, provider: str | None = None, model: str | None = None
    ) -> Model | None:
        """Get the appropriate PydanticAI model based on the active provider."""
        target_model = model or self.chat_config.model
        config = self._build_config_for_model_identifier(target_model)

        if provider:
            config.provider = provider
            config.auth_provider = provider

        api_key = self._get_api_key(config.auth_provider)
        if not api_key and config.auth_provider == "ollama":
            api_key = os.environ.get("OLLAMA_API_KEY", "ollama")
        if not api_key:
            api_key = os.environ.get("LLM_API_KEY", api_key)

        if not api_key and config.auth_provider not in {"ollama"}:
            raise UnsupportedProviderError(
                f"API key not found for provider '{config.auth_provider}'."
            )

        model_name = (
            target_model.split("/", 1)[1] if "/" in target_model else target_model
        )

        if not config.capabilities.get("supports_pydantic", False):
            raise UnsupportedProviderError(
                f"Model '{target_model}' does not support Pydantic-based agents."
            )

        provider_kwargs = {}
        if config.base_url:
            provider_kwargs["base_url"] = config.base_url
        if config.api_version:
            provider_kwargs["api_version"] = config.api_version

        openai_like_providers = {"openai", "openrouter", "azure", "ollama"}
        if config.auth_provider in openai_like_providers:
            if config.auth_provider == "ollama":
                base_url_root = (
                    config.base_url
                    or os.environ.get("LLM_API_BASE")
                    or "http://localhost:11434"
                )
                provider_kwargs["base_url"] = base_url_root.rstrip("/") + "/v1"
            model_class = (
                OpenRouterGeminiModel
                if config.auth_provider == "openrouter" and config.provider == "gemini"
                else OpenAIModel
            )
            return model_class(
                model_name=model_name,
                provider=OpenAIProvider(
                    api_key=api_key,
                    **provider_kwargs,
                ),
            )

        if config.provider == "anthropic":
            anthropic_kwargs = {
                key: value
                for key, value in provider_kwargs.items()
                if key != "api_version"
            }
            # Use CachingAnthropicModel for improved cache hit rates
            # This adds cache_control to tools and system prompts
            return CachingAnthropicModel(
                model_name=model_name,
                provider=AnthropicProvider(
                    api_key=api_key,
                    **anthropic_kwargs,
                ),
                enable_tool_caching=True,
                enable_system_caching=True,
                cache_ttl="5m",  # 5 minute cache TTL (refreshes on hit)
            )

        raise UnsupportedProviderError(
            f"Provider '{config.provider}' is not supported for Pydantic-based agents."
        )

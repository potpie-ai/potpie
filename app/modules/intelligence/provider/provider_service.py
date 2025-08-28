import logging
import os
from enum import Enum
from typing import List, Dict, Any, Union, AsyncGenerator, Optional
import uuid
from anthropic import AsyncAnthropic
from crewai import LLM
from pydantic import BaseModel
from pydantic_ai.models import Model
from litellm import litellm, AsyncOpenAI, acompletion
import instructor
import httpx
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL

from app.modules.key_management.secret_manager import SecretManager
from app.modules.users.user_preferences_model import UserPreferences
from app.modules.utils.posthog_helper import PostHogClient

logger = logging.getLogger(__name__)

from .provider_schema import (
    ProviderInfo,
    GetProviderResponse,
    AvailableModelsResponse,
    AvailableModelOption,
    SetProviderRequest,
    ModelInfo,
)
from .llm_config import LLMProviderConfig, build_llm_provider_config

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider
import litellm

import random
import time
import asyncio
from functools import wraps

litellm.num_retries = 5  # Number of retries for rate limited requests

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
    settings = RetrySettings(max_retries=litellm.num_retries)

    if not is_recoverable_error(exception, settings):
        # If it's not a recoverable error, don't retry
        return False

    delay = calculate_backoff_time(retry_count, settings)

    provider = identify_provider_from_error(exception)
    logging.warning(
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
                        logging.error(
                            f"Max retries ({settings.max_retries}) exceeded for {provider} API call. "
                            f"Last error: {str(e)}"
                        )
                        raise

                    delay = calculate_backoff_time(retries, settings)

                    logging.warning(
                        f"{provider.capitalize()} API error: {str(e)}. "
                        f"Retry {retries+1}/{settings.max_retries}, "
                        f"waiting {delay:.2f}s before next attempt..."
                    )

                    await asyncio.sleep(delay)
                    retries += 1

            # This should never be reached due to the raise in the loop,
            # but included for clarity
            raise last_exception

        return wrapper

    return decorator


class AgentProvider(Enum):
    CREWAI = "CREWAI"
    LANGCHAIN = "LANGCHAIN"
    PYDANTICAI = "PYDANTICAI"


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
        id="anthropic/claude-3-7-sonnet-20250219",
        name="Claude 3.7 Sonnet",
        description="Highest level of intelligence and capability with toggleable extended thinking",
        provider="anthropic",
        is_chat_model=True,
        is_inference_model=False,
    ),
    AvailableModelOption(
        id="anthropic/claude-3-5-haiku-20241022",
        name="Claude 3.5 Haiku",
        description="Faster, more efficient Claude model",
        provider="anthropic",
        is_chat_model=False,
        is_inference_model=True,
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
]

# Extract unique platform providers from the available models
PLATFORM_PROVIDERS = list({model.provider for model in AVAILABLE_MODELS})


class ProviderService:
    def __init__(self, db, user_id: str):
        litellm.modify_params = True
        self.db = db
        self.llm = None
        self.user_id = user_id
        self.portkey_api_key = os.environ.get("PORTKEY_API_KEY", None)

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
        """Get API key for the specified provider."""
        env_key = os.getenv("LLM_API_KEY", None)
        if env_key:
            return env_key

        try:
            secret = SecretManager.get_secret(provider, self.user_id, self.db)
            return secret
        except Exception as e:
            if "404" in str(e):
                env_key = os.getenv(f"{provider.upper()}_API_KEY")
                if env_key:
                    return env_key
                return None
            raise e

    def _build_llm_params(self, config: LLMProviderConfig) -> Dict[str, Any]:
        """Build a dictionary of parameters for LLM initialization."""
        api_key = self._get_api_key(config.model.split("/")[0])
        return config.get_llm_params(api_key)

    def get_extra_params_and_headers(
        self, routing_provider: Optional[str]
    ) -> tuple[dict[str, Any], Any]:
        """Get extra parameters and headers for API calls."""
        extra_params = {}
        headers = createHeaders(
            api_key=self.portkey_api_key,
            provider=routing_provider,
            trace_id=str(uuid.uuid4())[:8],
            custom_host=os.environ.get("LLM_API_BASE"),
            api_version=os.environ.get("LLM_API_VERSION"),
        )
        if self.portkey_api_key and routing_provider != "ollama":
            # ollama + portkey is not supported currently
            extra_params["base_url"] = PORTKEY_GATEWAY_URL
            extra_params["extra_headers"] = headers
        elif routing_provider == "azure":
            extra_params["api_base"] = os.environ.get("LLM_API_BASE")
            extra_params["api_version"] = os.environ.get("LLM_API_VERSION")
        return extra_params, headers

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
            logging.error(f"Error getting global AI provider: {e}")
            raise e

    def is_current_model_supported_by_pydanticai(
        self, config_type: str = "chat"
    ) -> bool:
        """Check if the current model is supported by PydanticAI."""
        config = self.chat_config if config_type == "chat" else self.inference_config
        return config.provider in ["openai", "anthropic", "openrouter"]

    @robust_llm_call()  # Apply the robust_llm_call decorator
    async def call_llm(
        self, messages: list, stream: bool = False, config_type: str = "chat"
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Call LLM with the specified messages with robust error handling."""
        # Select the appropriate config based on config_type
        config = self.chat_config if config_type == "chat" else self.inference_config

        # Build parameters using the config object
        params = self._build_llm_params(config)
        routing_provider = config.model.split("/")[0]

        # Get extra parameters and headers for API calls
        extra_params, _ = self.get_extra_params_and_headers(routing_provider)
        params.update(extra_params)

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
            logging.error(f"Error calling LLM: {e}, provider: {routing_provider}")
            raise e

    @robust_llm_call()
    async def call_llm_with_structured_output(
        self, messages: list, output_schema: BaseModel, config_type: str = "chat"
    ) -> Any:
        """Call LLM and parse the response into a structured output using a Pydantic model."""
        # Select the appropriate config
        config = self.chat_config if config_type == "chat" else self.inference_config

        # Build parameters
        params = self._build_llm_params(config)
        routing_provider = config.model.split("/")[0]

        # Get extra parameters and headers
        extra_params, _ = self.get_extra_params_and_headers(routing_provider)

        try:
            if config.provider == "ollama":
                # use openai client to call ollama because of https://github.com/BerriAI/litellm/issues/7355
                client = instructor.from_openai(
                    AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
                    mode=instructor.Mode.JSON,
                )
                response = await client.chat.completions.create(
                    model=params["model"].split("/")[-1],
                    messages=messages,
                    response_model=output_schema,
                    temperature=params.get("temperature", 0.3),
                    max_tokens=params.get("max_tokens"),
                    **extra_params,
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
                    api_key=params.get("api_key"),
                    **extra_params,
                )
            return response
        except Exception as e:
            logging.error(f"LLM call with structured output failed: {e}")
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
        # Select the appropriate config based on config_type
        config = self.chat_config if config_type == "chat" else self.inference_config

        # Build parameters using the config object
        params = self._build_llm_params(config)
        routing_provider = config.model.split("/")[0]

        # Get extra parameters and headers for API calls
        extra_params, _ = self.get_extra_params_and_headers(routing_provider)
        params.update(extra_params)

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
            logging.error(
                f"Error calling multimodal LLM: {e}, provider: {routing_provider}"
            )
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

            except Exception as e:
                logger.error(f"Error validating image {img_id}: {str(e)}")
                continue

        logger.info(
            f"Validated {len(validated_images)} out of {len(images)} images for multimodal processing"
        )
        return validated_images

    def is_vision_model(self, config_type: str = "chat") -> bool:
        """Check if the current model supports vision/multimodal inputs"""
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
            # Google models
            "gemini-pro-vision",
            "gemini-1.5",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-2.0",
            "gemini-2.0-flash",
            "gemini-2.5",
            "gemini-2.5-pro",
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

    def _initialize_llm(self, config: LLMProviderConfig, agent_type: AgentProvider):
        """Initialize LLM for the specified agent type."""
        params = self._build_llm_params(config)
        routing_provider = config.model.split("/")[0]

        # Get extra parameters and headers
        extra_params, headers = self.get_extra_params_and_headers(routing_provider)

        if agent_type == AgentProvider.CREWAI:
            crewai_params = {"model": params["model"], **params}
            if "default_headers" in params:
                crewai_params["headers"] = params["default_headers"]

            # Update with extra parameters
            crewai_params.update(extra_params)
            self.llm = LLM(**crewai_params)
        else:
            return None

    def get_llm(self, agent_type: AgentProvider, config_type: str = "chat"):
        """Get LLM for the specified agent type."""
        config = self.chat_config if config_type == "chat" else self.inference_config
        self._initialize_llm(config, agent_type)
        return self.llm

    def get_pydantic_model(
        self, provider: str | None = None, model: str | None = None
    ) -> Model | None:
        """Get the appropriate PydanticAI model based on the current provider."""
        config = self.chat_config
        model_name = config.model.split("/")[-1]

        if provider:
            config.provider = provider

        if model:
            model_name = model

        api_key = self._get_api_key(config.provider)

        if not api_key:
            return None

        # if portkey is enabled, use portkey gateway
        if self.portkey_api_key:
            match config.provider:
                case "openai":
                    return OpenAIModel(
                        model_name=model_name,
                        provider=OpenAIProvider(
                            api_key=api_key,
                            base_url=PORTKEY_GATEWAY_URL,
                            http_client=httpx.AsyncClient(
                                headers=createHeaders(
                                    api_key=self.portkey_api_key,
                                    provider=config.provider,
                                    trace_id=str(uuid.uuid4())[:8],
                                ),
                            ),
                        ),
                    )
                case "anthropic":
                    return AnthropicModel(
                        model_name=model_name,
                        provider=AnthropicProvider(
                            anthropic_client=AsyncAnthropic(
                                base_url=PORTKEY_GATEWAY_URL,
                                api_key=api_key,
                                default_headers=createHeaders(
                                    api_key=self.portkey_api_key,
                                    provider=config.provider,
                                    trace_id=str(uuid.uuid4())[:8],
                                ),
                            ),
                        ),
                    )
                case "openrouter":
                    # PORTKEY has a issue when used with openrouter here
                    return OpenAIModel(
                        model_name=config.model.split("/")[-2] + "/" + model_name,
                        provider=OpenAIProvider(
                            api_key=api_key,
                            base_url="https://openrouter.ai/api/v1",
                        ),
                    )

        match config.model.split("/")[0]:
            case "openai":
                return OpenAIModel(
                    model_name=model_name,
                    provider=OpenAIProvider(
                        api_key=api_key,
                    ),
                )
            case "anthropic":
                return AnthropicModel(
                    model_name=model_name,
                    provider=AnthropicProvider(
                        api_key=api_key,
                    ),
                )
            case "gemini":
                # OpenRouter uses OpenAI-compatible API
                return OpenAIModel(
                    model_name="google/" + model_name,
                    provider=OpenAIProvider(
                        api_key=api_key,
                        base_url="https://openrouter.ai/api/v1",
                    ),
                )

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
import litellm

litellm.num_retries = 5  # Number of retries for rate limited requests


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
        return config.provider in ["openai", "anthropic"]

    async def call_llm(
        self, messages: list, stream: bool = False, config_type: str = "chat"
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Call LLM with the specified messages."""
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
            logging.error(
                f"Error calling LLM: {e}, params: {params}, messages: {messages}"
            )
            raise e

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

    def get_pydantic_model(self) -> Model | None:
        """Get the appropriate PydanticAI model based on the current provider."""
        config = self.chat_config
        model_name = config.model.split("/")[-1]
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
                        anthropic_client=AsyncAnthropic(
                            base_url=PORTKEY_GATEWAY_URL,
                            api_key=api_key,
                            default_headers=createHeaders(
                                api_key=self.portkey_api_key,
                                provider=config.provider,
                                trace_id=str(uuid.uuid4())[:8],
                            ),
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
                    anthropic_client=AsyncAnthropic(
                        api_key=api_key,
                    ),
                )

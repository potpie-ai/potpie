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

from .provider_schema import ProviderInfo, GetProviderResponse, AvailableModelsResponse, AvailableModelOption
from .llm_config import LLMProviderConfig, build_llm_provider_config, CHAT_MODEL_CONFIG_MAP, INFERENCE_MODEL_CONFIG_MAP

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.openai import OpenAIProvider


class AgentProvider(Enum):
    CREWAI = "CREWAI"
    LANGCHAIN = "LANGCHAIN"
    PYDANTICAI = "PYDANTICAI"


PLATFORM_PROVIDERS = [
    "openai",
    "anthropic",
    "deepseek",
    "meta-llama",
    "gemini",
]


class ProviderService:
    def __init__(self, db, user_id: str):
        litellm.modify_params = True
        self.db = db
        self.llm = None
        self.user_id = user_id
        self.portkey_api_key = os.environ.get("PORTKEY_API_KEY", None)
        
        # Load user preferences
        user_pref = db.query(UserPreferences).filter_by(user_id=user_id).first()
        user_config = user_pref.preferences if user_pref and user_pref.preferences else {}
        
        # Create two configurations based on user input (or fallback defaults)
        self.chat_config = build_llm_provider_config(user_config, config_type="chat")
        self.inference_config = build_llm_provider_config(user_config, config_type="inference")

    @classmethod
    def create(cls, db, user_id: str):
        return cls(db, user_id)

    async def list_available_llms(self) -> List[ProviderInfo]:
        return [
            ProviderInfo(
                id="openai",
                name="OpenAI",
                description="A leading LLM provider, known for GPT models like GPT-3, GPT-4.",
            ),
            ProviderInfo(
                id="anthropic",
                name="Anthropic",
                description="An AI safety-focused company known for models like Claude.",
            ),
            ProviderInfo(
                id="deepseek",
                name="DeepSeek",
                description="An open-source AI company known for powerful chat and reasoning models.",
            ),
            ProviderInfo(
                id="meta-llama",
                name="Meta Llama",
                description="Meta's family of large language models.",
            ),
            ProviderInfo(
                id="gemini",
                name="Google Gemini",
                description="Google Gemini models.",
            ),
        ]
        
    async def list_available_models(self) -> AvailableModelsResponse:
        """List all available models for both chat and inference configurations."""
        chat_models = []
        inference_models = []
        
        # Convert chat models from mapping
        for model_id, config in CHAT_MODEL_CONFIG_MAP.items():
            chat_models.append(
                AvailableModelOption(
                    id=model_id,
                    name=model_id,  # Use the same ID as name for now
                    description=f"{config['provider'].capitalize()} model for chat/agent interactions",
                    provider=config["provider"],
                )
            )
            
        # Convert inference models from mapping
        for model_id, config in INFERENCE_MODEL_CONFIG_MAP.items():
            inference_models.append(
                AvailableModelOption(
                    id=model_id,
                    name=model_id,  # Use the same ID as name for now
                    description=f"{config['provider'].capitalize()} model for structured inference",
                    provider=config["provider"],
                )
            )
            
        return AvailableModelsResponse(
            chat_models=chat_models,
            inference_models=inference_models,
        )

    async def set_global_ai_provider(
        self,
        user_id: str,
        provider: str,
        low_reasoning_model: Optional[str] = None,
        high_reasoning_model: Optional[str] = None,
        config_type: str = "chat",
        selected_model: Optional[str] = None,
    ):
        provider = provider.lower()
        preferences = self.db.query(UserPreferences).filter_by(user_id=user_id).first()

        if not preferences:
            preferences = UserPreferences(
                user_id=user_id, preferences={}
            )
            self.db.add(preferences)
        else:
            if preferences.preferences is None:
                preferences.preferences = {}
                
        # Set provider for the specific configuration type
        provider_key = f"{config_type}_provider"
        preferences.preferences[provider_key] = provider
        
        # If a selected model was provided, store it
        if selected_model:
            selected_model_key = f"selected_{config_type}_model"
            preferences.preferences[selected_model_key] = selected_model

        if provider in PLATFORM_PROVIDERS:
            # For platform providers, allow model update only if API key is set
            api_key_set = await SecretManager.check_secret_exists_for_user(
                provider, user_id, self.db
            ) or (
                os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
            )  # check env keys too for platform providers
            if (
                low_reasoning_model or high_reasoning_model
            ):  # if user is trying to set custom models for platform provider
                if not api_key_set:
                    raise ValueError(
                        f"To set custom models for {provider}, please set your API key first."
                    )

        # Store model names with config type prefix
        if low_reasoning_model:
            low_model_key = f"low_reasoning_{config_type}_model"
            preferences.preferences[low_model_key] = low_reasoning_model
        if high_reasoning_model:
            high_model_key = f"high_reasoning_{config_type}_model"
            preferences.preferences[high_model_key] = high_reasoning_model

        self.db.query(UserPreferences).filter_by(user_id=user_id).update(
            {"preferences": preferences.preferences}
        )

        PostHogClient().send_event(
            user_id, f"{config_type}_provider_change_event", {"provider": provider}
        )
        self.db.commit()
        
        # Refresh the configuration object
        if config_type == "chat":
            self.chat_config = build_llm_provider_config(preferences.preferences, "chat")
        else:
            self.inference_config = build_llm_provider_config(preferences.preferences, "inference")
            
        return {"message": f"AI {config_type} provider set to {provider}"}

    def _get_provider_config(self, size: str, config_type: str = "chat") -> str:
        """
        Return the provider from environment variable LLM_PROVIDER if set;
        otherwise, fall back to user preferences, then default to 'openai'.
        """
        # Check for environment override
        env_var_name = f"{config_type.upper()}_LLM_PROVIDER"
        env_provider = os.environ.get(env_var_name) or os.environ.get("LLM_PROVIDER")
        if env_provider:
            return env_provider.lower()
            
        if self.user_id == "dummy":
            return "openai"
            
        # Get from user preferences
        user_pref = (
            self.db.query(UserPreferences)
            .filter(UserPreferences.user_id == self.user_id)
            .first()
        )
        
        if not user_pref or not user_pref.preferences:
            return "openai"
            
        # Try to get the provider specific to the config type
        provider_key = f"{config_type}_provider"
        if provider_key in user_pref.preferences:
            return user_pref.preferences[provider_key]
            
        # Fall back to global provider if specific not found
        return user_pref.preferences.get("llm_provider", "openai")

    def _get_reasoning_model_config(self, size: str, config_type: str = "chat") -> str:
        """
        Return the reasoning model from environment variables or user preferences,
        falling back to defaults if not set.
        """
        # Environment variable names based on config type
        env_prefix = f"{config_type.upper()}_"
        env_low_model = os.environ.get(f"{env_prefix}LOW_REASONING_MODEL") or os.environ.get("LOW_REASONING_MODEL")
        env_high_model = os.environ.get(f"{env_prefix}HIGH_REASONING_MODEL") or os.environ.get("HIGH_REASONING_MODEL")

        user_pref = (
            self.db.query(UserPreferences)
            .filter(UserPreferences.user_id == self.user_id)
            .first()
        )

        if size == "small":
            if env_low_model:
                return env_low_model
            elif user_pref and user_pref.preferences:
                # Try config-specific key first
                low_model_key = f"low_reasoning_{config_type}_model"
                if low_model_key in user_pref.preferences:
                    return user_pref.preferences[low_model_key]
                # Fall back to global key
                elif "low_reasoning_model" in user_pref.preferences:
                    return user_pref.preferences["low_reasoning_model"]
                
            # Use config object as last resort
            if config_type == "chat":
                return self.chat_config.low_reasoning_model
            else:
                return self.inference_config.low_reasoning_model
                
        elif size == "large":
            if env_high_model:
                return env_high_model
            elif user_pref and user_pref.preferences:
                # Try config-specific key first
                high_model_key = f"high_reasoning_{config_type}_model"
                if high_model_key in user_pref.preferences:
                    return user_pref.preferences[high_model_key]
                # Fall back to global key
                elif "high_reasoning_model" in user_pref.preferences:
                    return user_pref.preferences["high_reasoning_model"]
                
            # Use config object as last resort
            if config_type == "chat":
                return self.chat_config.high_reasoning_model
            else:
                return self.inference_config.high_reasoning_model
                
        return None

    def _get_api_key(self, provider: str) -> str:
        """Get API key for the specified provider."""

        env_key = os.getenv("LLM_API_KEY", None)
        if env_key:
            return env_key

        try:
            secret = SecretManager.get_secret(provider, self.user_id, self.db)
            return secret.get("api_key")
        except Exception as e:
            if "404" in str(e):
                env_key = os.getenv(f"{provider.upper()}_API_KEY")
                if env_key:
                    return env_key
                return None  # No API key found in secret manager or env for platform provider
            raise e

    def _build_llm_params(self, config: LLMProviderConfig, size: str) -> Dict[str, Any]:
        """
        Build a dictionary of parameters for LLM initialization.
        Takes an LLMProviderConfig object directly for clarity and correctness.
        """
        api_key = self._get_api_key(config.provider)
        return config.get_llm_params(size, api_key)

    def get_chat_llm_params(self, size: str) -> Dict[str, Any]:
        """Get parameters for chat LLM calls."""
        return self._build_llm_params(self.chat_config, size)
        
    def get_inference_llm_params(self, size: str) -> Dict[str, Any]:
        """Get parameters for inference LLM calls."""
        return self._build_llm_params(self.inference_config, size)

    async def call_llm(
        self, messages: list, size: str = "small", stream: bool = False, config_type: str = "inference"
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Call LLM with the specified messages and size."""
        # Select the appropriate config based on config_type
        config = self.chat_config if config_type == "chat" else self.inference_config
        
        # Build parameters using the config object
        params = self._build_llm_params(config, size)

        # Add Portkey headers if available
        if self.portkey_api_key:
            routing_provider = config.provider
            portkey_headers = createHeaders(
                api_key=self.portkey_api_key,
                provider=routing_provider,
                trace_id=str(uuid.uuid4())[:8],
                mode="litellm"
            )
            params["extra_headers"] = portkey_headers
            params["base_url"] = PORTKEY_GATEWAY_URL

        # Handle streaming response if requested
        if stream:
            try:
                async def generator() -> AsyncGenerator[str, None]:
                    response = await acompletion(
                        messages=messages, stream=True, **params
                    )
                    async for chunk in response:
                        yield chunk.choices[0].delta.content or ""

                return generator()
            except Exception as e:
                logging.error(
                    f"Error calling LLM with streaming: {e}, params: {params}, messages: {messages}"
                )
                raise e
        else:
            try:
                response = await acompletion(messages=messages, **params)
                return response.choices[0].message.content
            except Exception as e:
                logging.error(
                    f"Error calling LLM: {e}, params: {params}, messages: {messages}"
                )
                raise e

    async def call_llm_with_structured_output(
        self, messages: list, output_schema: BaseModel, size: str = "small", config_type: str = "inference"
    ) -> Any:
        """
        Call LLM with structured output using instructor.
        Uses the inference config by default.
        """
        # Select the appropriate config based on config_type (default to inference)
        config = self.inference_config if config_type == "inference" else self.chat_config
        
        # Build parameters using the config object
        params = self._build_llm_params(config, size)

        # Add Portkey headers if available
        if self.portkey_api_key:
            portkey_headers = createHeaders(
                api_key=self.portkey_api_key,
                provider=config.provider,
                trace_id=str(uuid.uuid4())[:8],
            )
            params["extra_headers"] = portkey_headers
            params["base_url"] = PORTKEY_GATEWAY_URL

        try:
            model_name = params["model"].split("/")[-1]

            # For openai provider, use the AsyncOpenAI client
            if config.provider == "openai":
                client = AsyncOpenAI(
                    api_key=params.get("api_key"),
                    base_url=params.get("base_url", None),
                    default_headers=params.get("extra_headers", None),
                )
                client = instructor.patch(client)
                response = await client.chat.completions.create(
                    messages=messages,
                    model=model_name,
                    temperature=params.get("temperature", 0.2),
                    response_model=output_schema,
                )
                return response
            # For anthropic provider, use the AsyncAnthropic client
            elif config.provider == "anthropic":
                client = AsyncAnthropic(api_key=params.get("api_key"))
                client = instructor.patch(client)
                response = await client.messages.create(
                    model=model_name,
                    messages=messages,
                    temperature=params.get("temperature", 0.2),
                    response_model=output_schema,
                    max_tokens=params.get("max_tokens", 4000),
                )
                return response
            else:
                # For other providers, fall back to litellm
                with instructor.patch(litellm):
                    response = await acompletion(
                        messages=messages,
                        response_model=output_schema,
                        **params,
                    )
                    return response
        except Exception as e:
            logging.error(
                f"Error calling LLM with structured output: {e}, params: {params}, messages: {messages}, schema: {output_schema}"
            )
            raise e

    def _initialize_llm(self, config: LLMProviderConfig, size: str, agent_type: AgentProvider):
        """Initialize LLM for the specified agent type."""
        params = self._build_llm_params(config, size)
        api_key = params.pop("api_key", None)

        if agent_type == AgentProvider.CREWAI:
            # For CrewAI
            if config.provider == "anthropic":
                self.llm = LLM(
                    provider="anthropic",
                    model=params["model"],
                    api_key=api_key,
                    temperature=params.get("temperature", 0.3),
                )
            else:  # default to OpenAI
                self.llm = LLM(
                    provider="openai",
                    model=params["model"],
                    api_key=api_key,
                    temperature=params.get("temperature", 0.3),
                    config={"base_url": params.get("api_base", None)},
                )
        # elif agent_type == AgentProvider.LANGCHAIN:
        #     # For LangChain
        #     if config.provider == "anthropic":
        #         self.llm = ChatAnthropic(
        #             model=params["model"],
        #             anthropic_api_key=api_key,
        #             temperature=params.get("temperature", 0.3),
        #         )
        #     else:  # default to OpenAI
        #         self.llm = ChatOpenAI(
        #             model=params["model"],
        #             openai_api_key=api_key,
        #             temperature=params.get("temperature", 0.3),
        #         )

    def get_large_llm(self, agent_type: AgentProvider, config_type: str = "chat"):
        """Get large LLM for the specified agent type."""
        config = self.chat_config if config_type == "chat" else self.inference_config
        self._initialize_llm(config, "large", agent_type)
        return self.llm

    def get_small_llm(self, agent_type: AgentProvider, config_type: str = "chat"):
        """Get small LLM for the specified agent type."""
        config = self.chat_config if config_type == "chat" else self.inference_config
        self._initialize_llm(config, "small", agent_type)
        return self.llm

    async def get_global_ai_provider(self, user_id: str, config_type: str = "chat") -> GetProviderResponse:
        """Get the current global AI provider configuration."""
        try:
            user_pref = (
                self.db.query(UserPreferences)
                .filter(UserPreferences.user_id == user_id)
                .first()
            )

            # Get provider based on config type
            provider_key = f"{config_type}_provider"
            preferred_llm = (
                user_pref.preferences.get(provider_key)
                if user_pref and user_pref.preferences and provider_key in user_pref.preferences
                else user_pref.preferences.get("llm_provider", "openai")
                if user_pref and user_pref.preferences
                else "openai"
            )

            # Get selected model name
            selected_model_key = f"selected_{config_type}_model"
            selected_model = (
                user_pref.preferences.get(selected_model_key)
                if user_pref and user_pref.preferences and selected_model_key in user_pref.preferences
                else None
            )

            # Model names for low and high reasoning based on config type
            low_model_key = f"low_reasoning_{config_type}_model"
            high_model_key = f"high_reasoning_{config_type}_model"

            # Get models from user preferences or use defaults from config objects
            if config_type == "chat":
                config = self.chat_config
            else:
                config = self.inference_config

            # Try to get models from user preferences first, fall back to config object
            low_reasoning_model = (
                user_pref.preferences.get(low_model_key)
                if user_pref and user_pref.preferences and low_model_key in user_pref.preferences
                else config.low_reasoning_model
            )
            
            high_reasoning_model = (
                user_pref.preferences.get(high_model_key)
                if user_pref and user_pref.preferences and high_model_key in user_pref.preferences
                else config.high_reasoning_model
            )

            return GetProviderResponse(
                preferred_llm=preferred_llm,
                model_type="global",
                low_reasoning_model=low_reasoning_model,
                high_reasoning_model=high_reasoning_model,
                config_type=config_type,
                selected_model=selected_model
            )
        except Exception as e:
            logging.error(f"Error getting global AI provider: {e}")
            raise e

    def is_current_model_supported_by_pydanticai(self) -> bool:
        """Check if the current model is supported by PydanticAI."""
        return self.inference_config.provider in ["openai", "anthropic"]

    def get_pydantic_model(self) -> Model | None:
        """Get the appropriate PydanticAI model based on the current provider."""
        config = self.chat_config
        model_name = config.high_reasoning_model.split("/")[-1]
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
        # No portkey, use direct provider integration
        if config.provider == "openai":
            return OpenAIModel(
                openai_provider=OpenAIProvider(api_key=api_key),
                model_name=model_name,
            )
        elif config.provider == "anthropic":
            return AnthropicModel(api_key=api_key, model_name=model_name)
        else:
            return None

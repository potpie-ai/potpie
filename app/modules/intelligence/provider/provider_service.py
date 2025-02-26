import logging
import os
from enum import Enum
from typing import List, Dict, Any, Union, AsyncGenerator, Optional
import uuid
from crewai import LLM
from pydantic import BaseModel
from litellm import litellm, AsyncOpenAI, acompletion
import instructor
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL

from app.modules.key_management.secret_manager import SecretManager
from app.modules.users.user_preferences_model import UserPreferences
from app.modules.utils.posthog_helper import PostHogClient

from .provider_schema import ProviderInfo, GetProviderResponse


class AgentProvider(Enum):
    CREWAI = "CREWAI"
    LANGCHAIN = "LANGCHAIN"


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

    async def set_global_ai_provider(
        self,
        user_id: str,
        provider: str,
        low_reasoning_model: Optional[str] = None,
        high_reasoning_model: Optional[str] = None,
    ):
        provider = provider.lower()
        preferences = self.db.query(UserPreferences).filter_by(user_id=user_id).first()

        if not preferences:
            preferences = UserPreferences(
                user_id=user_id, preferences={"llm_provider": provider}
            )
            self.db.add(preferences)
        else:
            if preferences.preferences is None:
                preferences.preferences = {}
            preferences.preferences["llm_provider"] = provider

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

        if low_reasoning_model:
            preferences.preferences["low_reasoning_model"] = low_reasoning_model
        if high_reasoning_model:
            preferences.preferences["high_reasoning_model"] = high_reasoning_model

        self.db.query(UserPreferences).filter_by(user_id=user_id).update(
            {"preferences": preferences.preferences}
        )

        PostHogClient().send_event(
            user_id, "provider_change_event", {"provider": provider}
        )
        self.db.commit()
        return {"message": f"AI provider set to {provider}"}

    # Model configurations
    MODEL_CONFIGS = {
        "openai": {
            "small": {"model": "openai/gpt-4o-mini"},
            "large": {"model": "openai/gpt-4o"},
        },
        "anthropic": {
            "small": {"model": "anthropic/claude-3-5-haiku-20241022"},
            "large": {"model": "anthropic/claude-3-7-sonnet-20250219"},
        },
        "deepseek": {
            "small": {"model": "openrouter/deepseek/deepseek-chat"},
            "large": {
                "model": "openrouter/deepseek/deepseek-chat"
            },  # r1 is slow and unstable rn
        },
        "meta-llama": {
            "small": {"model": "openrouter/meta-llama/llama-3.3-70b-instruct"},
            "large": {"model": "openrouter/meta-llama/llama-3.3-70b-instruct"},
        },
        "gemini": {
            "small": {"model": "openrouter/google/gemini-2.0-flash-001"},
            "large": {
                "model": "openrouter/google/gemini-2.0-flash-001"
            },  # TODO: add pro model after it moves out of experimentsl and gets higher rate
        },
    }

    def _get_provider_config(self, size: str) -> str:
        """
        Return the provider from environment variable LLM_PROVIDER if set;
        otherwise, fall back to user preferences, then default to 'openai'.
        """
        env_provider = os.environ.get("LLM_PROVIDER")
        if env_provider:
            return env_provider.lower()
        if self.user_id == "dummy":
            return "openai"
        user_pref = (
            self.db.query(UserPreferences)
            .filter(UserPreferences.user_id == self.user_id)
            .first()
        )
        return (
            user_pref.preferences.get("llm_provider", "openai")
            if user_pref and user_pref.preferences
            else "openai"
        )

    def _get_reasoning_model_config(self, size: str) -> str:
        """
        Return the reasoning model from environment variables or user preferences,
        falling back to defaults if not set.
        """
        env_low_model = os.environ.get("LOW_REASONING_MODEL")
        env_high_model = os.environ.get("HIGH_REASONING_MODEL")

        user_pref = (
            self.db.query(UserPreferences)
            .filter(UserPreferences.user_id == self.user_id)
            .first()
        )

        if size == "small":
            if env_low_model:
                return env_low_model
            elif user_pref and user_pref.preferences.get("low_reasoning_model"):
                return user_pref.preferences.get("low_reasoning_model")
            else:
                provider = self._get_provider_config(size)
                return self.MODEL_CONFIGS[provider]["small"]["model"]
        elif size == "large":
            if env_high_model:
                return env_high_model
            elif user_pref and user_pref.preferences.get("high_reasoning_model"):
                return user_pref.preferences.get("high_reasoning_model")
            else:
                provider = self._get_provider_config(size)
                return self.MODEL_CONFIGS[provider]["large"]["model"]
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

    def _build_llm_params(self, provider: str, size: str) -> Dict[str, Any]:
        """
        Build a dictionary of parameters for LLM initialization.
        Model is determined by _get_reasoning_model_config.
        """
        if (
            provider not in self.MODEL_CONFIGS and provider not in PLATFORM_PROVIDERS
        ):  # Allow non-platform providers
            # For non-platform providers, model names must be user specified, retrieve from user preferences
            user_pref = (
                self.db.query(UserPreferences)
                .filter(UserPreferences.user_id == self.user_id)
                .first()
            )
            if size == "small":
                low_reasoning_model = os.environ.get("LOW_REASONING_MODEL", None)
                model_name = (
                    low_reasoning_model
                    if low_reasoning_model
                    else (
                        user_pref.preferences.get("low_reasoning_model")
                        if user_pref and user_pref.preferences
                        else None
                    )
                )
            elif size == "large":
                high_reasoning_model = os.environ.get("HIGH_REASONING_MODEL", None)
                model_name = (
                    high_reasoning_model
                    if high_reasoning_model
                    else (
                        user_pref.preferences.get("high_reasoning_model")
                        if user_pref and user_pref.preferences
                        else None
                    )
                )
            if not model_name:
                raise ValueError(
                    f"Model name for {size} size for provider {provider} is not set in preferences."
                )
            params = {
                "temperature": 0.3,
                "api_key": self._get_api_key(provider),
                "model": model_name,
                "routing_provider": (
                    model_name.split("/")[0] if "/" in model_name else provider
                ),
            }

        elif (
            provider in self.MODEL_CONFIGS
        ):  # platform providers with default model configs
            params = {
                "temperature": 0.3,
                "api_key": self._get_api_key(provider),
                "model": self._get_reasoning_model_config(size),
                "routing_provider": self._get_reasoning_model_config(size).split("/")[
                    0
                ],
            }
        else:
            raise ValueError(f"Invalid LLM provider: {provider}")

        # For deepseek large model, add extra parameters.
        if provider == "deepseek":
            params.update({"max_tokens": 8000})

        if provider == "anthropic":
            params.update({"max_tokens": 8000})

        return params

    async def call_llm(
        self, messages: list, size: str = "small", stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Call LLM using LiteLLM's asynchronous completion.
        API key and model are dynamically configured.
        """
        provider = self._get_provider_config(size)
        params = self._build_llm_params(provider, size)
        routing_provider = params.pop("routing_provider", None)
        extra_params = {}
        if self.portkey_api_key and routing_provider != "ollama":
            # ollama + portkey is not supported currently
            extra_params["base_url"] = PORTKEY_GATEWAY_URL
            extra_params["extra_headers"] = createHeaders(
                api_key=self.portkey_api_key, provider=routing_provider
            )

        try:
            if stream:

                async def generator() -> AsyncGenerator[str, None]:
                    response = await acompletion(
                        model=params["model"],
                        messages=messages,
                        temperature=params.get("temperature", 0.3),
                        max_tokens=params.get("max_tokens"),
                        stream=True,
                        api_key=params.get("api_key"),
                        **extra_params,
                    )
                    async for chunk in response:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content

                return generator()
            else:
                response = await acompletion(
                    model=params["model"],
                    messages=messages,
                    temperature=params.get("temperature", 0.3),
                    max_tokens=params.get("max_tokens"),
                    stream=False,
                    api_key=params.get("api_key"),
                    **extra_params,
                )
                return response.choices[0].message.content
        except Exception as e:
            logging.error(f"LLM call failed: {e}")
            raise e

    async def call_llm_with_structured_output(
        self, messages: list, output_schema: BaseModel, size: str = "small"
    ) -> Any:
        """
        Call LLM and parse the response into a structured output using a Pydantic model.
        Uses Instructor's integration with LiteLLM for structured outputs.
        API key and model are dynamically configured.
        """
        provider = self._get_provider_config(size)
        params = self._build_llm_params(provider, size)
        routing_provider = params.pop("routing_provider", None)

        extra_params = {}
        if self.portkey_api_key and routing_provider != "ollama":
            # ollama + portkey is not supported currently
            extra_params["base_url"] = PORTKEY_GATEWAY_URL
            extra_params["extra_headers"] = createHeaders(
                api_key=self.portkey_api_key, provider=routing_provider
            )

        try:
            if provider == "ollama":
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
                    temperature=params.get("temperature", 0.3),
                    max_tokens=params.get("max_tokens"),
                    api_key=params.get("api_key"),
                    **extra_params,
                )
            return response
        except Exception as e:
            logging.error(f"LLM call with structured output failed: {e}")
            raise e

    def _initialize_llm(self, provider: str, size: str, agent_type: AgentProvider):
        """
        Initialize LLM based on provider, size, and agent type.
        Although agent_type and provider are passed, with simplified config, they are less relevant now.
        Kept for potential future differentiated initialization.
        """
        params = self._build_llm_params(provider, size)
        routing_provider = params.pop("routing_provider", None)
        if agent_type == AgentProvider.CREWAI:
            crewai_params = {"model": params["model"], **params}
            if "default_headers" in params:
                crewai_params["headers"] = params["default_headers"]
            if self.portkey_api_key and routing_provider != "ollama":
                # ollama + portkey is not supported currently
                headers = createHeaders(
                    api_key=self.portkey_api_key,
                    provider=routing_provider,
                    trace_id=str(uuid.uuid4())[:8],
                )
                crewai_params["extra_headers"] = headers
                crewai_params["base_url"] = PORTKEY_GATEWAY_URL
            return LLM(**crewai_params)
        else:
            return None

    def get_large_llm(self, agent_type: AgentProvider):
        provider = self._get_provider_config("large")
        logging.info(f"Initializing {provider.capitalize()} LLM")
        self.llm = self._initialize_llm(provider, "large", agent_type)
        return self.llm

    def get_small_llm(self, agent_type: AgentProvider):
        provider = self._get_provider_config("small")
        self.llm = self._initialize_llm(provider, "small", agent_type)
        return self.llm

    async def get_global_ai_provider(self, user_id: str) -> GetProviderResponse:
        user_pref = (
            self.db.query(UserPreferences)
            .filter(UserPreferences.user_id == user_id)
            .first()
        )
        provider = (
            user_pref.preferences.get("llm_provider", "openai")
            if user_pref and user_pref.preferences
            else "openai"
        )
        low_reasoning_model = (
            user_pref.preferences.get("low_reasoning_model")
            if user_pref and user_pref.preferences
            else None
        )
        high_reasoning_model = (
            user_pref.preferences.get("high_reasoning_model")
            if user_pref and user_pref.preferences
            else None
        )

        default_small_model = (
            self.MODEL_CONFIGS[provider]["small"]["model"]
            if provider in self.MODEL_CONFIGS
            else "openai/gpt-4o-mini"
        )  # Fallback in case provider is invalid in user_prefs
        default_large_model = (
            self.MODEL_CONFIGS[provider]["large"]["model"]
            if provider in self.MODEL_CONFIGS
            else "openai/gpt-4o"
        )  # Fallback in case provider is invalid in user_prefs

        return GetProviderResponse(
            preferred_llm=provider,
            model_type="global",  # or any other relevant type, if needed
            low_reasoning_model=(
                low_reasoning_model if low_reasoning_model else default_small_model
            ),
            high_reasoning_model=(
                high_reasoning_model if high_reasoning_model else default_large_model
            ),
        )

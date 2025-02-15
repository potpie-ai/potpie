import logging
import os
from enum import Enum
from typing import List, Tuple

from crewai import LLM
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_openai.chat_models import ChatOpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from app.modules.key_management.secret_manager import SecretManager
from app.modules.users.user_preferences_model import UserPreferences
from app.modules.utils.posthog_helper import PostHogClient

from .provider_schema import ProviderInfo


class AgentType(Enum):
    CREWAI = "CREWAI"
    LANGCHAIN = "LANGCHAIN"


class ProviderService:
    def __init__(self, db, user_id: str):
        self.db = db
        self.llm = None
        self.user_id = user_id
        if os.getenv("isDevelopmentMode") != "enabled":
            self.PORTKEY_API_KEY = os.environ.get("PORTKEY_API_KEY")
        self.openrouter_base_url = "https://openrouter.ai/api/v1"

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
        ]

    async def set_global_ai_provider(self, user_id: str, provider: str):
        provider = provider.lower()
        # First check if preferences exist
        preferences = self.db.query(UserPreferences).filter_by(user_id=user_id).first()

        if not preferences:
            # Create new preferences if they don't exist
            preferences = UserPreferences(
                user_id=user_id, preferences={"llm_provider": provider}
            )
            self.db.add(preferences)
        else:
            # Initialize preferences dict if None
            if preferences.preferences is None:
                preferences.preferences = {}

            # Update the provider in preferences
            preferences.preferences["llm_provider"] = provider

            # Explicit update query
            self.db.query(UserPreferences).filter_by(user_id=user_id).update(
                {"preferences": preferences.preferences}
            )

        PostHogClient().send_event(
            user_id, "provider_change_event", {"provider": provider}
        )

        self.db.commit()
        return {"message": f"AI provider set to {provider}"}

    # Model configurations for different providers and sizes
    MODEL_CONFIGS = {
        "openai": {
            "small": {
                "crewai": {"model": "openai/gpt-4o-mini"},
                "langchain": {
                    "model": "gpt-4o-mini",
                    "class": ChatOpenAI,
                },
            },
            "large": {
                "crewai": {"model": "openai/gpt-4o"},
                "langchain": {
                    "model": "gpt-4o",
                    "class": ChatOpenAI,
                },
            },
        },
        "anthropic": {
            "small": {
                "crewai": {"model": "anthropic/claude-3-5-haiku-20241022"},
                "langchain": {
                    "model": "claude-3-5-haiku-20241022",
                    "class": ChatAnthropic,
                },
            },
            "large": {
                "crewai": {"model": "anthropic/claude-3-5-sonnet-20241022"},
                "langchain": {
                    "model": "claude-3-5-sonnet-20241022",
                    "class": ChatAnthropic,
                },
            },
        },
        "deepseek": {
            "small": {
                "crewai": {"model": "deepseek/deepseek/deepseek-chat"},
                "langchain": {
                    "model": "deepseek/deepseek-chat",
                    "class": ChatDeepSeek,
                },
            },
            "large": {
                "crewai": {"model": "deepseek/deepseek/deepseek-r1"},
                "langchain": {
                    "model": "deepseek/deepseek-r1",
                    "class": ChatDeepSeek,
                },
            },
        },
    }

    def _get_provider_config(self, size: str) -> str:
        """Get the preferred provider and its configuration."""
        if self.user_id == "dummy":
            return "openai"

        user_pref = (
            self.db.query(UserPreferences)
            .filter(UserPreferences.user_id == self.user_id)
            .first()
        )
        return (
            user_pref.preferences.get("llm_provider", "openai")
            if user_pref
            else "openai"
        )

    def _get_api_key(self, provider: str) -> str:
        """Get API key for the specified provider."""
        if os.getenv("isDevelopmentMode") == "enabled":
            logging.info(
                "Development mode enabled. Using environment variable for API key."
            )
            return os.getenv(f"{provider.upper()}_API_KEY")

        try:
            secret = SecretManager.get_secret(provider, self.user_id)
            return secret.get("api_key")
        except Exception as e:
            if "404" in str(e):
                return os.getenv(f"{provider.upper()}_API_KEY")
            raise e

    def _get_portkey_headers(self, provider: str):
        """Get Portkey headers for the specified provider."""
        if os.getenv("isDevelopmentMode") == "enabled":
            return None

        return createHeaders(
            api_key=self.PORTKEY_API_KEY,
            provider=provider,
            metadata={
                "_user": self.user_id,
                "environment": os.environ.get("ENV"),
            },
        )

    def _initialize_llm(self, provider: str, size: str, agent_type: AgentType):
        """Initialize LLM based on provider, size, and agent type."""
        if provider not in self.MODEL_CONFIGS:
            raise ValueError(f"Invalid LLM provider: {provider}")

        config = self.MODEL_CONFIGS[provider][size]
        api_key = self._get_api_key(provider)
        portkey_headers = self._get_portkey_headers(provider)

        common_params = {
            "temperature": 0.3,
            "api_key": api_key,
        }

        if provider == "deepseek" and size == "large":
            common_params.update(
                {
                    "max_tokens": 8000,
                    "base_url": self.openrouter_base_url,
                    "api_base": self.openrouter_base_url,
                }
            )

        if provider == "anthropic":
            common_params.update(
                {
                    "max_tokens": 8000,
                }
            )

        if agent_type == AgentType.CREWAI:
            return LLM(model=config["crewai"]["model"], **common_params)
        else:
            model_class = config["langchain"]["class"]
            model_params = {"model_name": config["langchain"]["model"], **common_params}

            if not os.getenv("isDevelopmentMode") == "enabled":
                model_params.update(
                    {
                        "base_url": PORTKEY_GATEWAY_URL,
                        "default_headers": portkey_headers,
                    }
                )

            return model_class(**model_params)

    def get_large_llm(self, agent_type: AgentType):
        provider = self._get_provider_config("large")
        logging.info(f"Initializing {provider.capitalize()} LLM")
        self.llm = self._initialize_llm(provider, "large", agent_type)
        return self.llm

    def get_small_llm(self, agent_type: AgentType):
        provider = self._get_provider_config("small")
        if provider == "deepseek":
            # temporary
            provider = "openai"
        self.llm = self._initialize_llm(provider, "small", agent_type)
        return self.llm

    def get_llm_provider_name(self) -> str:
        """Returns the name of the LLM provider based on the LLM instance."""
        llm = self.get_small_llm(agent_type=AgentType.LANGCHAIN)

        # Check the type of the LLM to determine the provider
        if isinstance(llm, ChatOpenAI):
            return "OpenAI"
        elif isinstance(llm, ChatAnthropic):
            return "Anthropic"
        elif isinstance(llm, ChatDeepSeek):
            return "DeepSeek"
        elif isinstance(llm, LLM):
            if llm.model.split("/")[0] == "openai":
                return "OpenAI"
            elif llm.model.split("/")[0] == "anthropic":
                return "Anthropic"
            elif llm.model.split("/")[0] == "deepseek":
                return "DeepSeek"
        return "Unknown"

    async def get_global_ai_provider(self, user_id: str) -> str:
        user_pref = (
            self.db.query(UserPreferences)
            .filter(UserPreferences.user_id == user_id)
            .first()
        )

        return (
            user_pref.preferences.get("llm_provider", "openai")
            if user_pref
            else "openai"
        )

    async def get_preferred_llm(self, user_id: str) -> Tuple[str, str]:
        user_pref = (
            self.db.query(UserPreferences)
            .filter(UserPreferences.user_id == user_id)
            .first()
        )

        preferred_provider = (
            user_pref.preferences.get("llm_provider", "openai")
            if user_pref
            else "openai"
        )

        model_type = "gpt-4o"
        if preferred_provider == "anthropic":
            model_type = "claude-3-5-sonnet-20241022"
        elif preferred_provider == "deepseek":
            # update after custom agent r1 suppport
            model_type = "gpt-4o"
            preferred_provider = "openai"

        return preferred_provider, model_type

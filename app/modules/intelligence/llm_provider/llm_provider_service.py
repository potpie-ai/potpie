from functools import lru_cache
import logging
import os
from typing import List, Tuple

from crewai import LLM
from langchain_anthropic import ChatAnthropic
from langchain_openai.chat_models import ChatOpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from app.modules.intelligence.llm_provider.llm_provider_schema import LLMProviderInfo
from app.modules.intelligence.prompts_provider.agent_types import AgentLLMType
from app.modules.key_management.secret_manager import SecretManager
from app.modules.users.user_preferences_model import UserPreferences
from app.modules.utils.posthog_helper import PostHogClient


class LLMProviderService:
    def __init__(self, db, user_id: str):
        self.db = db
        self.llm = None
        self.user_id = user_id
        if os.getenv("isDevelopmentMode") != "enabled":
            self.PORTKEY_API_KEY = os.environ.get("PORTKEY_API_KEY")

    @classmethod
    def create(cls, db, user_id: str):
        return cls(db, user_id)

    async def list_available_llms(self) -> List[LLMProviderInfo]:
        return [
            LLMProviderInfo(
                id="openai",
                name="OpenAI",
                description="A leading LLM provider, known for GPT models like GPT-3, GPT-4.",
            ),
            LLMProviderInfo(
                id="anthropic",
                name="Anthropic",
                description="An AI safety-focused company known for models like Claude.",
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
    
    @lru_cache(maxsize=3)
    def get_large_llm(self, agent_type: AgentLLMType):
        # Get user preferences from the database
        user_pref = (
            self.db.query(UserPreferences)
            .filter(UserPreferences.user_id == self.user_id)
            .first()
        )

        # Determine preferred provider (default to 'openai')
        preferred_provider = (
            user_pref.preferences.get("llm_provider", "openai")
            if user_pref
            else "openai"
        )

        if preferred_provider == "openai":
            logging.info("Initializing OpenAI LLM")
            if os.getenv("isDevelopmentMode") == "enabled":
                logging.info(
                    "Development mode enabled. Using environment variable for API key."
                )
                openai_key = os.getenv("OPENAI_API_KEY")
                if agent_type == AgentLLMType.CREWAI:
                    self.llm = LLM(
                        model="openai/gpt-4o-mini",
                        api_key=openai_key,
                        temperature=0.3,
                    )
                else:
                    self.llm = ChatOpenAI(
                        model_name="gpt-4o",
                        api_key=openai_key,
                        temperature=0.3,
                    )
            else:
                try:
                    secret = SecretManager.get_secret("openai", self.user_id)
                    openai_key = secret.get("api_key")
                except Exception as e:
                    if "404" in str(e):
                        openai_key = os.getenv("OPENAI_API_KEY")
                    else:
                        raise e

                portkey_headers = createHeaders(
                    api_key=self.PORTKEY_API_KEY,
                    provider="openai",
                    metadata={
                        "_user": self.user_id,
                        "environment": os.environ.get("ENV"),
                    },
                )
                if agent_type == AgentLLMType.CREWAI:
                    self.llm = LLM(
                        model="openai/gpt-4o", api_key=openai_key, temperature=0.3
                    )
                else:
                    self.llm = ChatOpenAI(
                        model_name="gpt-4o",
                        api_key=openai_key,
                        temperature=0.3,
                        base_url=PORTKEY_GATEWAY_URL,
                        default_headers=portkey_headers,
                    )

        elif preferred_provider == "anthropic":
            logging.info("Initializing Anthropic LLM")
            if os.getenv("isDevelopmentMode") == "enabled":
                logging.info(
                    "Development mode enabled. Using environment variable for API key."
                )
                anthropic_key = os.getenv("ANTHROPIC_API_KEY")
                if agent_type == AgentLLMType.CREWAI:
                    self.llm = LLM(
                        model="anthropic/claude-3-5-sonnet-20241022",
                        temperature=0.3,
                        api_key=anthropic_key,
                    )
                else:
                    self.llm = ChatAnthropic(
                        model="claude-3-5-sonnet-20241022",
                        temperature=0.3,
                        api_key=anthropic_key,
                    )
            else:
                try:
                    secret = SecretManager.get_secret("anthropic", self.user_id)
                    anthropic_key = secret.get("api_key")
                except Exception as e:
                    if "404" in str(e):
                        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
                    else:
                        raise e

                portkey_headers = createHeaders(
                    api_key=self.PORTKEY_API_KEY,
                    provider="anthropic",
                    metadata={
                        "_user": self.user_id,
                        "environment": os.environ.get("ENV"),
                    },
                )

                if agent_type == AgentLLMType.CREWAI:
                    self.llm = LLM(
                        model="anthropic/claude-3-5-sonnet-20241022",
                        temperature=0.3,
                        api_key=anthropic_key,
                    )
                else:
                    self.llm = ChatAnthropic(
                        model="claude-3-5-sonnet-20241022",
                        temperature=0.3,
                        api_key=anthropic_key,
                        base_url=PORTKEY_GATEWAY_URL,
                        default_headers=portkey_headers,
                    )

        else:
            raise ValueError("Invalid LLM provider selected.")

        return self.llm

    def get_small_llm(self, agent_type: AgentLLMType):
        # Get user preferences from the database
        if self.user_id == "dummy":
            user_pref = UserPreferences(
                user_id=self.user_id, preferences={"llm_provider": "openai"}
            )

        user_pref = (
            self.db.query(UserPreferences)
            .filter(UserPreferences.user_id == self.user_id)
            .first()
        )

        # Determine preferred provider (default to 'openai')
        preferred_provider = (
            user_pref.preferences.get("llm_provider", "openai")
            if user_pref
            else "openai"
        )

        if preferred_provider == "openai":
            if os.getenv("isDevelopmentMode") == "enabled":
                logging.info(
                    "Development mode enabled. Using environment variable for API key."
                )
                openai_key = os.getenv("OPENAI_API_KEY")
                if agent_type == AgentLLMType.CREWAI:
                    self.llm = LLM(
                        model="openai/gpt-4o-mini",
                        api_key=openai_key,
                        temperature=0.3,
                    )
                else:
                    self.llm = ChatOpenAI(
                        model_name="gpt-4o-mini",
                        api_key=openai_key,
                        temperature=0.3,
                    )
            else:
                try:
                    secret = SecretManager.get_secret("openai", self.user_id)
                    openai_key = secret.get("api_key")
                except Exception as e:
                    if "404" in str(e):
                        openai_key = os.getenv("OPENAI_API_KEY")
                    else:
                        raise e

                portkey_headers = createHeaders(
                    api_key=self.PORTKEY_API_KEY,
                    provider="openai",
                    metadata={
                        "_user": self.user_id,
                        "environment": os.environ.get("ENV"),
                    },
                )
                if agent_type == AgentLLMType.CREWAI:
                    self.llm = LLM(
                        model="openai/gpt-4o-mini",
                        api_key=openai_key,
                        temperature=0.3,
                    )
                else:
                    self.llm = ChatOpenAI(
                        model_name="gpt-4o-mini",
                        api_key=openai_key,
                        temperature=0.3,
                        base_url=PORTKEY_GATEWAY_URL,
                        default_headers=portkey_headers,
                    )

        elif preferred_provider == "anthropic":
            if os.getenv("isDevelopmentMode") == "enabled":
                logging.info(
                    "Development mode enabled. Using environment variable for API key."
                )
                anthropic_key = os.getenv("ANTHROPIC_API_KEY")
                if agent_type == AgentLLMType.CREWAI:
                    self.llm = LLM(
                        model="anthropic/claude-3-haiku-20240307",
                        temperature=0.3,
                        api_key=anthropic_key,
                    )
                else:
                    self.llm = ChatAnthropic(
                        model="claude-3-haiku-20240307",
                        temperature=0.3,
                        api_key=anthropic_key,
                    )
            else:
                try:
                    secret = SecretManager.get_secret("anthropic", self.user_id)
                    anthropic_key = secret.get("api_key")
                except Exception as e:
                    if "404" in str(e):
                        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
                    else:
                        raise e

                portkey_headers = createHeaders(
                    api_key=self.PORTKEY_API_KEY,
                    provider="anthropic",
                    metadata={
                        "_user": self.user_id,
                        "environment": os.environ.get("ENV"),
                    },
                )

                if agent_type == AgentLLMType.CREWAI:
                    self.llm = LLM(
                        model="anthropic/claude-3-haiku-20240307",
                        temperature=0.3,
                        api_key=anthropic_key,
                    )
                else:
                    self.llm = ChatAnthropic(
                        model="claude-3-haiku-20240307",
                        temperature=0.3,
                        api_key=anthropic_key,
                        base_url=PORTKEY_GATEWAY_URL,
                        default_headers=portkey_headers,
                    )

        else:
            raise ValueError("Invalid LLM provider selected.")

        return self.llm

    def get_llm_provider_name(self) -> str:
        """Returns the name of the LLM provider based on the LLM instance."""
        llm = self.get_small_llm(agent_type=AgentLLMType.LANGCHAIN)

        # Check the type of the LLM to determine the provider
        if isinstance(llm, ChatOpenAI):
            return "OpenAI"
        elif isinstance(llm, ChatAnthropic):
            return "Anthropic"
        elif isinstance(llm, LLM):
            return "OpenAI" if llm.model.split("/")[0] == "openai" else "Anthropic"
        else:
            return "Unknown"

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

        model_type = (
            "gpt-4o" if preferred_provider == "openai" else "claude-3-5-sonnet-20241022"
        )

        return preferred_provider, model_type

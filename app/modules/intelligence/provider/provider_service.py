import logging
import os
from enum import Enum
from typing import List, Tuple

from langchain_community.chat_models import ChatLiteLLM

from app.modules.key_management.secret_manager import SecretManager
from app.modules.users.user_preferences_model import UserPreferences
from app.modules.utils.posthog_helper import PostHogClient

from .provider_schema import ProviderInfo
litellm_provider = os.getenv("LITELLM_PROVIDER", "openrouter")
class AgentType(Enum):
    CREWAI = "CREWAI"
    LANGCHAIN = "LANGCHAIN"

class ProviderService:
    def __init__(self, db, user_id: str):
        self.db = db
        self.llm = None
        self.user_id = user_id

    @classmethod
    def create(cls, db, user_id: str):
        return cls(db, user_id)

    async def list_available_llms(self) -> List[ProviderInfo]:
        return [
            ProviderInfo(
                id="litelm",
                name="LiteLLM",
                description="A provider for running open source models locally using LiteLLM.",
            )
        ]

    async def set_global_ai_provider(self, user_id: str, provider: str):
        provider = litellm_provider # Force all users to use LiteLLM
        
        preferences = self.db.query(UserPreferences).filter_by(user_id=user_id).first()

        if not preferences:
            preferences = UserPreferences(
                user_id=user_id, preferences={"llm_provider": litellm_provider}
            )
            self.db.add(preferences)
        else:
            if preferences.preferences is None:
                preferences.preferences = {}
            preferences.preferences["llm_provider"] = litellm_provider
            self.db.query(UserPreferences).filter_by(user_id=user_id).update(
                {"preferences": preferences.preferences}
            )

        PostHogClient().send_event(user_id, "provider_change_event", {"provider": litellm_provider})
        self.db.commit()
        return {"message": f"AI provider set to {litellm_provider}"}

    def _initialize_llm(self, size: str):
        """Initialize LiteLLM model with OpenRouter as the provider."""
        litellm_model = os.getenv("LITELLM_MODEL")
        provider = os.getenv("LITELLM_PROVIDER", "openrouter")
        logging.info(f"Initializing LiteLLM with model {litellm_model} and provider {litellm_provider}")
        self.llm = ChatLiteLLM(model=litellm_model)
        return self.llm


    def get_large_llm(self, agent_type: AgentType):
        self.llm = self._initialize_llm("large")
        return self.llm

    def get_small_llm(self, agent_type: AgentType):
        self.llm = self._initialize_llm("small")
        return self.llm

    def get_llm_provider_name(self) -> str:
        """Returns the name of the LLM provider, which is always LiteLLM."""
        return "openrouter"

    async def get_global_ai_provider(self, user_id: str) -> str:
        return "openrouter"

    async def get_preferred_llm(self, user_id: str) -> Tuple[str, str]:
        return "openrouter", os.getenv("LITELLM_MODEL")

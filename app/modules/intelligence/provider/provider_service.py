import logging
import os
from enum import Enum
from typing import List, Tuple

from langchain_ollama import ChatOllama

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

    @classmethod
    def create(cls, db, user_id: str):
        return cls(db, user_id)

    async def list_available_llms(self) -> List[ProviderInfo]:
        return [
            ProviderInfo(
                id="ollama",
                name="Ollama",
                description="A provider for running open source models locally.",
            )
        ]

    async def set_global_ai_provider(self, user_id: str, provider: str):
        provider = "ollama"  # Force all users to use Ollama
        
        preferences = self.db.query(UserPreferences).filter_by(user_id=user_id).first()

        if not preferences:
            preferences = UserPreferences(
                user_id=user_id, preferences={"llm_provider": "ollama"}
            )
            self.db.add(preferences)
        else:
            if preferences.preferences is None:
                preferences.preferences = {}
            preferences.preferences["llm_provider"] = provider
            self.db.query(UserPreferences).filter_by(user_id=user_id).update(
                {"preferences": preferences.preferences}
            )

        PostHogClient().send_event(user_id, "provider_change_event", {"provider": provider})
        self.db.commit()
        return {"message": f"AI provider set to {provider}"}

    def _initialize_llm(self, size: str):
        """Initialize local Ollama model."""
        ollama_endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")  # Default to a local model
        
        logging.info(f"Initializing Ollama LLM with model {ollama_model}")
        self.llm = ChatOllama(base_url=ollama_endpoint, model=ollama_model)
        return self.llm

    def get_large_llm(self, agent_type: AgentType):
        self.llm = self._initialize_llm("large")
        return self.llm

    def get_small_llm(self,agent_type: AgentType):
        self.llm = self._initialize_llm("small")
        return self.llm

    def get_llm_provider_name(self) -> str:
        """Returns the name of the LLM provider, which is always Ollama."""
        return "Ollama"

    async def get_global_ai_provider(self, user_id: str) -> str:
        return "ollama"

    async def get_preferred_llm(self, user_id: str) -> Tuple[str, str]:
        return "ollama", os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")

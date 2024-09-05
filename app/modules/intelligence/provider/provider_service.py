from typing import List
import os
from .provider_schema import ProviderInfo
from app.modules.users.user_preferences_model import UserPreferences
from app.modules.key_management.secret_manager import SecretManager
from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic

class ProviderService:
    def __init__(self, db, user_id):
        self.db = db
        self.user_id = user_id
        self.llm = None

    @classmethod
    def create(cls, db, user_id):
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
        ]
    
    async def set_global_ai_provider(self, user_id: str, provider: str):
        preferences = self.db.query(UserPreferences).filter_by(user_id=user_id).first()
        if not preferences:
            preferences = UserPreferences(user_id=user_id, preferences={})
            self.db.add(preferences)

        preferences.preferences["llm_provider"] = provider
        self.db.commit()

        return {"message": f"AI provider set to {provider}"}

    async def get_llm(self):
        # Get user preferences from the database
        user_pref = (
            self.db.query(UserPreferences)
            .filter(UserPreferences.user_id == self.user_id)
            .first()
        )
        
        # Determine preferred provider (default to 'openai')
        preferred_provider = user_pref.preferences.get("llm_provider", "openai") if user_pref else "openai"
        
        if preferred_provider == "openai":
            secret = SecretManager.get_secret("openai", self.user_id)
            openai_key = secret.get("api_key", os.getenv("OPENAI_API_KEY"))  # Fallback to env variable if no key
            self.llm = ChatOpenAI(api_key=openai_key, temperature=0.7, model_kwargs={"stream": True})
        
        elif preferred_provider == "anthropic":
            secret = SecretManager.get_secret("anthropic", self.user_id)
            anthropic_key = secret.get("api_key", os.getenv("ANTHROPIC_API_KEY"))  # Fallback to env variable if no key
            self.llm = ChatAnthropic(
                model="claude-3-sonnet-20240229",
                temperature=0,
                max_tokens=1024,
                api_key=anthropic_key,
            )

        else:
            raise ValueError("Invalid LLM provider selected.")
        
        return self.llm

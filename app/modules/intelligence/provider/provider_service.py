import os
from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_openai.chat_models import ChatOpenAI

from app.modules.key_management.secret_manager import SecretManager
from app.modules.users.user_preferences_model import UserPreferences

from .provider_schema import ProviderInfo


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

    def get_llm(self):
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
            try:
                # Try fetching the secret key from SecretManager
                secret = SecretManager.get_secret("openai", self.user_id)
                openai_key = secret.get("api_key")
            except Exception as e:
                # Log the exception if needed
                if "404" in str(e):
                    # If the secret is not found, fallback to environment variable
                    openai_key = os.getenv("OPENAI_API_KEY")
                else:
                    raise e  # Re-raise if it's a different error

            self.llm = ChatOpenAI(
                model_name="gpt-4o",
                api_key=openai_key,  # Use the key properly
                temperature=0.7,
                model_kwargs={"stream": True},
            )

        elif preferred_provider == "anthropic":
            try:
                # Try fetching the secret key from SecretManager
                secret = SecretManager.get_secret("anthropic", self.user_id)
                anthropic_key = secret.get("api_key")
            except Exception as e:
                # Log the exception if needed
                if "404" in str(e):
                    # If the secret is not found, fallback to environment variable
                    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
                else:
                    raise e  # Re-raise if it's a different error

            self.llm = ChatAnthropic(
                model="claude-3-5-sonnet-20240620",
                temperature=0,
                api_key=anthropic_key,  # Use the key properly
            )

        else:
            raise ValueError("Invalid LLM provider selected.")

        return self.llm
    def get_mini_llm(self):
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
            try:
                # Try fetching the secret key from SecretManager
                secret = SecretManager.get_secret("openai", self.user_id)
                openai_key = secret.get("api_key")
            except Exception as e:
                if "404" in str(e):
                    # If the secret is not found, fallback to environment variable
                    openai_key = os.getenv("OPENAI_API_KEY")
                else:
                    raise e  # Re-raise if it's a different error

            self.llm = ChatOpenAI(
                model_name="gpt-4o-mini", api_key=openai_key, temperature=0.7
            )

        elif preferred_provider == "anthropic":
            try:
                # Try fetching the secret key from SecretManager
                secret = SecretManager.get_secret("anthropic", self.user_id)
                anthropic_key = secret.get("api_key")
            except Exception as e:
                if "404" in str(e):
                    # If the secret is not found, fallback to environment variable
                    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
                else:
                    raise e  # Re-raise if it's a different error

            self.llm = ChatAnthropic(
                model="claude-3-haiku-20240307", temperature=0, api_key=anthropic_key
            )

        else:
            raise ValueError("Invalid LLM provider selected.")

        return self.llm
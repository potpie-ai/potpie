import asyncio
import json
import logging
import os
from enum import Enum
from typing import List, Tuple, Dict, Any, Type, get_origin, get_args, Union, AsyncGenerator
from pydantic import BaseModel, create_model
from crewai import LLM
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_openai.chat_models import ChatOpenAI
from portkey_ai import AsyncPortkey

from app.modules.key_management.secret_manager import SecretManager
from app.modules.users.user_preferences_model import UserPreferences
from app.modules.utils.posthog_helper import PostHogClient

from .provider_schema import ProviderInfo


class AgentProvider(Enum):
    CREWAI = "CREWAI"
    LANGCHAIN = "LANGCHAIN"


class ProviderService:
    def __init__(self, db, user_id: str):
        self.db = db
        self.llm = None
        self.user_id = user_id
        self.PORTKEY_API_KEY = os.environ.get("PORTKEY_API_KEY", None)
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
        self.portkey = None  # Initialize later when we know the provider

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

    def _initialize_portkey(self, provider: str):
        api_key = self._get_api_key(provider)
        if not api_key:
            return None

        if not self.PORTKEY_API_KEY:
            
            return AsyncPortkey(
                provider=provider,
                Authorization=f"Bearer {api_key}",
            )

        return AsyncPortkey(
            apiKey=self.PORTKEY_API_KEY,
            provider=provider,
            Authorization=f"Bearer {api_key}",
            metadata={
                "_user": self.user_id,
                "environment": os.environ.get("ENV"),
            }
        )

    def _build_llm_params(self, provider: str, size: str) -> Dict[str, Any]:
        """Build a dictionary of parameters for LLM initialization."""
        if provider not in self.MODEL_CONFIGS:
            raise ValueError(f"Invalid LLM provider: {provider}")

        config = self.MODEL_CONFIGS[provider][size]
        api_key = self._get_api_key(provider)

        params = {
            "temperature": 0.3,
            "api_key": api_key,
        }

        if provider == "deepseek" and size == "large":
            params.update({
                "max_tokens": 8000,
                "base_url": self.openrouter_base_url,
                "api_base": self.openrouter_base_url,
            })

        if provider == "anthropic":
            params.update({
                "max_tokens": 8000,
            })

        return params

    async def call_llm(self, messages: list, size: str = "small", stream: bool = False) -> Union[str, AsyncGenerator[str, None]]:
        provider = self._get_provider_config(size)
        params = self._build_llm_params(provider, size)
        config = self.MODEL_CONFIGS[provider][size]
        
        try:
            # Initialize Portkey with the correct provider
            self.portkey = self._initialize_portkey(provider)
            
            if self.portkey:
                if stream:
                    # Streaming case: wrap the synchronous iterator into an async generator
                    async def generator() -> AsyncGenerator[str, None]:
                        async for chunk in self.portkey.beta.chat.completions.stream(
                            model=config["langchain"]["model"],
                            messages=messages,
                            stream=True,
                            temperature=params.get("temperature", 0.3),
                            max_tokens=params.get("max_tokens"),
                            top_p=params.get("top_p", 1.0),
                        ):  
                            chunk = json.loads(chunk)
                            chunk = chunk["chunk"] if chunk["type"] == "chunk" else None
                            if chunk["choices"][0]["delta"]["content"]:
                                yield chunk["choices"][0]["delta"]["content"]
                    return generator()
                else:
                    # Non-streaming: gather the output into a string and return it
                    response = await self.portkey.chat.completions.create(
                        model=config["langchain"]["model"],
                        messages=messages,
                        temperature=params.get("temperature", 0.3),
                        max_tokens=params.get("max_tokens"),
                        stream=False
                    )
                    return response.choices[0].message.content
        except Exception as e:
            logging.error(f"LLM call failed: {e}")
            raise e

    async def call_llm_with_structured_output(self, messages: list, output_schema: BaseModel, size: str = "small") -> Any:
        """
        Call LLM and parse the response into a structured output using a Pydantic model.
        Uses Portkey's structured output feature for more reliable JSON responses.
        """
        provider = self._get_provider_config(size)
        params = self._build_llm_params(provider, size)
        config = self.MODEL_CONFIGS[provider][size]
        
        try:
            # Initialize Portkey with the correct provider
            self.portkey = self._initialize_portkey(provider)
            
            if self.portkey:

                # Get the Pydantic schema
                def transform_annotation(annotation: Any) -> Any:
                    """
                    Recursively transform type annotations: if the annotation is a Pydantic model,
                    replace it with its "no-defaults" copy; if it's a generic type (like Optional[T] or List[T]),
                    recursively process its arguments.
                    """
                    origin = get_origin(annotation)
                    if origin is None:
                        # Not a generic type: if it's a subclass of BaseModel, recursively copy it.
                        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                            return copy_without_defaults(annotation)
                        return annotation
                    else:
                        # Process generic arguments recursively.
                        args = get_args(annotation)
                        new_args = tuple(transform_annotation(arg) for arg in args)
                        try:
                            return origin[new_args]
                        except TypeError:
                            # Some types may not be subscriptable, so fallback.
                            return annotation
                    
                def copy_without_defaults(model_cls: Type[BaseModel]) -> Type[BaseModel]:
                    new_fields = {}
                    # Process each field's type annotation, replacing nested models recursively.
                    for name, annotation in model_cls.__annotations__.items():
                        new_annotation = transform_annotation(annotation)
                        # Mark the field as required (by using Ellipsis for its default).
                        new_fields[name] = (new_annotation, ...)
                    # Create a new model with the same configuration (by inheriting from model_cls).
                    NewModel = create_model(f"{model_cls.__name__}NoDefaults", __base__=model_cls, **new_fields)
                    return NewModel
                
                schema = copy_without_defaults(output_schema)
                # Use Portkey's beta.chat.completions.parse for structured output
                response = await self.portkey.beta.chat.completions.parse(
                    model=config["langchain"]["model"],
                    messages=messages,
                    response_format=schema,  # Pass the Pydantic model directly
                    temperature=params.get("temperature", 0.3),
                    max_tokens=params.get("max_tokens"),
                )
                return response.choices[0].message.parsed

        except Exception as e:
            logging.error(f"LLM call with structured output failed: {e}")
            raise

    def _initialize_llm(self, provider: str, size: str, agent_type: AgentProvider):
        """Initialize LLM based on provider, size, and agent type."""
        config = self.MODEL_CONFIGS[provider][size]
        params = self._build_llm_params(provider, size)

        if agent_type == AgentProvider.CREWAI:
            crewai_params = {
                "model": config["crewai"]["model"],
                **params
            }
            
            if "default_headers" in params:
                crewai_params["headers"] = params["default_headers"]
            
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
        if provider == "deepseek":
            # temporary
            provider = "openai"
        self.llm = self._initialize_llm(provider, "small", agent_type)
        return self.llm

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

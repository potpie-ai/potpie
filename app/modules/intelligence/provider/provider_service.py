import logging
import os
import asyncio
from enum import Enum
from typing import List, Tuple

from crewai import LLM
from langchain_anthropic import ChatAnthropic
from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from app.modules.key_management.secret_manager import SecretManager
from app.modules.users.user_preferences_model import UserPreferences
from app.modules.utils.posthog_helper import PostHogClient
from app.modules.utils.rate_limiter import RateLimiter

from .provider_schema import ProviderInfo

logger = logging.getLogger(__name__)

class AgentType(Enum):
    CREWAI = "CREWAI"
    LANGCHAIN = "LANGCHAIN"


class ProviderService:
    def __init__(self, db, user_id: str):
        """Initialize ProviderService with database session and user ID."""
        logger.info(f"Initializing ProviderService for user_id: {user_id}")
        self.db = db
        self.llm = None
        self.user_id = user_id
        if os.getenv("isDevelopmentMode") != "enabled":
            logger.debug("Production mode detected, setting up Portkey API key")
            self.PORTKEY_API_KEY = os.environ.get("PORTKEY_API_KEY")
            if not self.PORTKEY_API_KEY:
                logger.warning("PORTKEY_API_KEY not found in environment variables")

        # Initialize rate limiter
        self.rate_limiter = RateLimiter("provider")
        logger.debug("Rate limiter initialized for provider service")

    @classmethod
    def create(cls, db, user_id: str):
        """Factory method to create ProviderService instance."""
        logger.info(f"Creating new ProviderService instance for user_id: {user_id}")
        return cls(db, user_id)

    async def list_available_llms(self) -> List[ProviderInfo]:
        """List all available LLM providers."""
        logger.debug("Fetching list of available LLMs")
        providers = [
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
                id="google",
                name="Google Gemini",
                description="Google's latest language model offering.",
            ),
        ]
        logger.info(f"Retrieved {len(providers)} available LLM providers")
        return providers

    async def set_global_ai_provider(self, user_id: str, provider: str = None):
        """Set the global AI provider for a user."""
        logger.info(f"Setting global AI provider for user_id: {user_id}")
        
        try:
            if provider is None:
                provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
                logger.debug(f"No provider specified, using default: {provider}")
            
            provider = provider.lower()
            logger.debug(f"Normalized provider name: {provider}")
            
            # First check if preferences exist
            preferences = self.db.query(UserPreferences).filter_by(user_id=user_id).first()
            logger.debug(f"Current preferences found: {preferences is not None}")

            if not preferences:
                logger.info(f"Creating new preferences for user_id: {user_id}")
                preferences = UserPreferences(
                    user_id=user_id, preferences={"llm_provider": provider}
                )
                self.db.add(preferences)
            else:
                logger.debug("Updating existing preferences")
                # Initialize preferences dict if None
                if preferences.preferences is None:
                    preferences.preferences = {}

                # Update the provider in preferences
                preferences.preferences["llm_provider"] = provider

                # Explicit update query
                self.db.query(UserPreferences).filter_by(user_id=user_id).update(
                    {"preferences": preferences.preferences}
                )

            logger.debug("Sending analytics event for provider change")
            PostHogClient().send_event(
                user_id, "provider_change_event", {"provider": provider}
            )

            self.db.commit()
            logger.info(f"Successfully set AI provider to {provider} for user {user_id}")
            return {"message": f"AI provider set to {provider}"}
            
        except Exception as e:
            logger.error(f"Error setting global AI provider: {str(e)}", exc_info=True)
            raise

    def get_llm_provider_name(self) -> str:
        """Returns the name of the LLM provider based on the LLM instance."""
        logger.debug("Getting LLM provider name")
        try:
            llm = self.get_small_llm(agent_type=AgentType.LANGCHAIN)

            if isinstance(llm, ChatOpenAI):
                logger.debug("Identified OpenAI provider")
                return "OpenAI"
            elif isinstance(llm, ChatAnthropic):
                logger.debug("Identified Anthropic provider")
                return "Anthropic"
            elif isinstance(llm, ChatGoogleGenerativeAI):
                logger.debug("Identified Google provider")
                return "Google"
            elif isinstance(llm, LLM):
                provider = llm.model.split("/")[0]
                logger.debug(f"Identified provider from LLM model: {provider}")
                if provider == "openai":
                    return "OpenAI"
                elif provider == "anthropic":
                    return "Anthropic"
                elif provider == "google":
                    return "Google"
            
            logger.warning("Unable to identify LLM provider")
            return "Unknown"
            
        except Exception as e:
            logger.error(f"Error getting LLM provider name: {str(e)}", exc_info=True)
            raise

    async def get_large_llm(self, agent_type: AgentType):
        """Get large language model instance based on user preferences."""
        logger.info(f"Getting large LLM for agent type: {agent_type}")
        try:
            await self.rate_limiter.acquire()
            logger.debug("Rate limiter acquired")

            # Get user preferences from the database
            user_pref = (
                self.db.query(UserPreferences)
                .filter(UserPreferences.user_id == self.user_id)
                .first()
            )
            logger.debug(f"Retrieved user preferences: {user_pref is not None}")

            # Determine preferred provider (default to 'openai')
            preferred_provider = (
                user_pref.preferences.get("llm_provider", "openai")
                if user_pref
                else "openai"
            )

            if preferred_provider == "openai":
                logger.info("Initializing OpenAI LLM")
                if os.getenv("isDevelopmentMode") == "enabled":
                    logger.debug("Development mode enabled, using environment variables")
                    openai_key = os.getenv("OPENAI_API_KEY")
                    if not openai_key:
                        logger.error("OPENAI_API_KEY not found in environment variables")
                        raise ValueError("OPENAI_API_KEY not found")
                    
                    if agent_type == AgentType.CREWAI:
                        logger.debug("Creating CrewAI OpenAI instance")
                        self.llm = LLM(
                            model="openai/gpt-4",
                            api_key=openai_key,
                            temperature=0.3,
                        )
                    else:
                        logger.debug("Creating Langchain OpenAI instance")
                        self.llm = ChatOpenAI(
                            model_name="gpt-4",
                            api_key=openai_key,
                            temperature=0.3,
                        )
                else:
                    logger.debug("Production mode, fetching API key from secret manager")
                    try:
                        secret = SecretManager.get_secret("openai", self.user_id)
                        openai_key = secret.get("api_key")
                        logger.debug("Successfully retrieved OpenAI API key")
                    except Exception as e:
                        if "404" in str(e):
                            logger.warning("API key not found in secrets, using environment variable")
                            openai_key = os.getenv("OPENAI_API_KEY")
                        else:
                            logger.error(f"Error retrieving OpenAI API key: {str(e)}")
                            raise

                    portkey_headers = createHeaders(
                        api_key=self.PORTKEY_API_KEY,
                        provider="openai",
                        metadata={
                            "_user": self.user_id,
                            "environment": os.environ.get("ENV"),
                        },
                    )
                    logger.debug("Created Portkey headers for OpenAI")

                    if agent_type == AgentType.CREWAI:
                        logger.debug("Creating CrewAI OpenAI instance with Portkey")
                        self.llm = LLM(
                            model="openai/gpt-4",
                            api_key=openai_key,
                            temperature=0.3,
                        )
                    else:
                        logger.debug("Creating Langchain OpenAI instance with Portkey")
                        self.llm = ChatOpenAI(
                            model_name="gpt-4",
                            api_key=openai_key,
                            temperature=0.3,
                            base_url=PORTKEY_GATEWAY_URL,
                            default_headers=portkey_headers,
                        )

            elif preferred_provider == "anthropic":
                logger.info("Initializing Anthropic LLM")
                if os.getenv("isDevelopmentMode") == "enabled":
                    logger.debug("Development mode enabled, using environment variables")
                    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
                    if not anthropic_key:
                        logger.error("ANTHROPIC_API_KEY not found in environment variables")
                        raise ValueError("ANTHROPIC_API_KEY not found")
                    
                    if agent_type == AgentType.CREWAI:
                        logger.debug("Creating CrewAI Anthropic instance")
                        self.llm = LLM(
                            model="anthropic/claude-3-5-sonnet-20241022",
                            temperature=0.3,
                            api_key=anthropic_key,
                        )
                    else:
                        logger.debug("Creating Langchain Anthropic instance")
                        self.llm = ChatAnthropic(
                            model="claude-3-5-sonnet-20241022",
                            temperature=0.3,
                            api_key=anthropic_key,
                        )
                else:
                    logger.debug("Production mode, fetching API key from secret manager")
                    try:
                        secret = SecretManager.get_secret("anthropic", self.user_id)
                        anthropic_key = secret.get("api_key")
                        logger.debug("Successfully retrieved Anthropic API key")
                    except Exception as e:
                        if "404" in str(e):
                            logger.warning("API key not found in secrets, using environment variable")
                            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
                        else:
                            logger.error(f"Error retrieving Anthropic API key: {str(e)}")
                            raise

                    portkey_headers = createHeaders(
                        api_key=self.PORTKEY_API_KEY,
                        provider="anthropic",
                        metadata={
                            "_user": self.user_id,
                            "environment": os.environ.get("ENV"),
                        },
                    )
                    logger.debug("Created Portkey headers for Anthropic")

                    if agent_type == AgentType.CREWAI:
                        logger.debug("Creating CrewAI Anthropic instance with Portkey")
                        self.llm = LLM(
                            model="anthropic/claude-3-5-sonnet-20241022",
                            temperature=0.3,
                            api_key=anthropic_key,
                        )
                    else:
                        logger.debug("Creating Langchain Anthropic instance with Portkey")
                        self.llm = ChatAnthropic(
                            model="claude-3-5-sonnet-20241022",
                            temperature=0.3,
                            api_key=anthropic_key,
                            base_url=PORTKEY_GATEWAY_URL,
                            default_headers=portkey_headers,
                        )

            elif preferred_provider == "google":
                logger.info("Initializing Google LLM")
                if os.getenv("isDevelopmentMode") == "enabled":
                    logger.debug("Development mode enabled, using environment variables")
                    google_key = os.getenv("GOOGLE_API_KEY")
                    if not google_key:
                        logger.error("GOOGLE_API_KEY not found in environment variables")
                        raise ValueError("GOOGLE_API_KEY not found")
                    
                    if agent_type == AgentType.CREWAI:
                        logger.debug("Creating CrewAI Google instance")
                        self.llm = LLM(
                            model="gemini-pro",
                            api_key=google_key,
                            temperature=0.3,
                        )
                    else:
                        logger.debug("Creating Langchain Google instance")
                        self.llm = ChatGoogleGenerativeAI(
                            model="gemini-pro",
                            model_name="gemini-pro",
                            temperature=0.3,
                            google_api_key=google_key,
                        )
                else:
                    logger.debug("Production mode, fetching API key from secret manager")
                    try:
                        secret = SecretManager.get_secret("google", self.user_id)
                        google_key = secret.get("api_key")
                        logger.debug("Successfully retrieved Google API key")
                    except Exception as e:
                        if "404" in str(e):
                            logger.warning("API key not found in secrets, using environment variable")
                            google_key = os.getenv("GOOGLE_API_KEY")
                        else:
                            logger.error(f"Error retrieving Google API key: {str(e)}")
                            raise

                    portkey_headers = createHeaders(
                        api_key=self.PORTKEY_API_KEY,
                        provider="google",
                        metadata={
                            "_user": self.user_id,
                            "environment": os.environ.get("ENV"),
                        },
                    )
                    logger.debug("Created Portkey headers for Google")

                    if agent_type == AgentType.CREWAI:
                        logger.debug("Creating CrewAI Google instance with Portkey")
                        self.llm = LLM(
                            model="gemini-pro",
                            api_key=google_key,
                            temperature=0.3,
                        )
                    else:
                        logger.debug("Creating Langchain Google instance with Portkey")
                        self.llm = ChatGoogleGenerativeAI(
                            model="gemini-pro",
                            model_name="gemini-pro",
                            temperature=0.3,
                            google_api_key=google_key,
                            base_url=PORTKEY_GATEWAY_URL,
                            default_headers=portkey_headers,
                        )

            else:
                logger.error(f"Unsupported provider: {preferred_provider}")
                raise ValueError(f"Unsupported provider: {preferred_provider}")

            logger.info(f"Successfully initialized LLM with provider: {preferred_provider}")
            return self.llm

        except Exception as e:
            logger.error(f"Error initializing large LLM: {str(e)}", exc_info=True)
            raise

    async def get_small_llm(self, agent_type: AgentType):
        """Get small language model instance based on user preferences."""
        logger.info(f"Getting small LLM for user_id: {self.user_id}")
        logger.debug(f"Environment DEFAULT_LLM_PROVIDER: {os.getenv('DEFAULT_LLM_PROVIDER')}")
        
        try:
            await self.rate_limiter.acquire()
            logger.debug("Rate limiter acquired")

            # Get user preferences from the database
            if self.user_id == "dummy":
                default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
                logger.info(f"Setting dummy user provider to: {default_provider}")
                user_pref = UserPreferences(
                    user_id=self.user_id, 
                    preferences={"llm_provider": default_provider}
                )
            else:
                logger.debug("Fetching user preferences from database")
                user_pref = (
                    self.db.query(UserPreferences)
                    .filter(UserPreferences.user_id == self.user_id)
                    .first()
                )

            # Determine preferred provider
            preferred_provider = (
                user_pref.preferences.get("llm_provider", os.getenv("DEFAULT_LLM_PROVIDER", "openai"))
                if user_pref
                else os.getenv("DEFAULT_LLM_PROVIDER", "openai")
            )
            preferred_provider = preferred_provider.lower()
            logger.info(f"Selected provider: {preferred_provider}")

            if preferred_provider == "openai":
                logger.info("Initializing OpenAI small LLM")
                if os.getenv("isDevelopmentMode") == "enabled":
                    logger.debug("Development mode enabled, using environment variables")
                    openai_key = os.getenv("OPENAI_API_KEY")
                    if not openai_key:
                        logger.error("OPENAI_API_KEY not found in environment variables")
                        raise ValueError("OPENAI_API_KEY not found")
                    
                    if agent_type == AgentType.CREWAI:
                        logger.debug("Creating CrewAI OpenAI instance")
                        self.llm = LLM(
                            model="openai/gpt-4o-mini",
                            api_key=openai_key,
                            temperature=0.3,
                        )
                    else:
                        logger.debug("Creating Langchain OpenAI instance")
                        self.llm = ChatOpenAI(
                            model_name="gpt-4o-mini",
                            api_key=openai_key,
                            temperature=0.3,
                        )
                else:
                    logger.debug("Production mode, fetching API key from secret manager")
                    try:
                        secret = SecretManager.get_secret("openai", self.user_id)
                        openai_key = secret.get("api_key")
                        logger.debug("Successfully retrieved OpenAI API key")
                    except Exception as e:
                        if "404" in str(e):
                            logger.warning("API key not found in secrets, using environment variable")
                            openai_key = os.getenv("OPENAI_API_KEY")
                        else:
                            logger.error(f"Error retrieving OpenAI API key: {str(e)}")
                            raise

                    portkey_headers = createHeaders(
                        api_key=self.PORTKEY_API_KEY,
                        provider="openai",
                        metadata={
                            "_user": self.user_id,
                            "environment": os.environ.get("ENV"),
                        },
                    )
                    logger.debug("Created Portkey headers for OpenAI")

                    if agent_type == AgentType.CREWAI:
                        logger.debug("Creating CrewAI OpenAI instance with Portkey")
                        self.llm = LLM(
                            model="openai/gpt-4o-mini",
                            api_key=openai_key,
                            temperature=0.3,
                        )
                    else:
                        logger.debug("Creating Langchain OpenAI instance with Portkey")
                        self.llm = ChatOpenAI(
                            model_name="gpt-4o-mini",
                            api_key=openai_key,
                            temperature=0.3,
                            base_url=PORTKEY_GATEWAY_URL,
                            default_headers=portkey_headers,
                        )

            elif preferred_provider == "anthropic":
                logger.info("Initializing Anthropic small LLM")
                if os.getenv("isDevelopmentMode") == "enabled":
                    logger.debug("Development mode enabled, using environment variables")
                    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
                    if not anthropic_key:
                        logger.error("ANTHROPIC_API_KEY not found in environment variables")
                        raise ValueError("ANTHROPIC_API_KEY not found")
                    
                    if agent_type == AgentType.CREWAI:
                        logger.debug("Creating CrewAI Anthropic instance")
                        self.llm = LLM(
                            model="anthropic/claude-3-haiku-20240307",
                            temperature=0.3,
                            api_key=anthropic_key,
                        )
                    else:
                        logger.debug("Creating Langchain Anthropic instance")
                        self.llm = ChatAnthropic(
                            model="claude-3-haiku-20240307",
                            temperature=0.3,
                            api_key=anthropic_key,
                        )
                else:
                    logger.debug("Production mode, fetching API key from secret manager")
                    try:
                        secret = SecretManager.get_secret("anthropic", self.user_id)
                        anthropic_key = secret.get("api_key")
                        logger.debug("Successfully retrieved Anthropic API key")
                    except Exception as e:
                        if "404" in str(e):
                            logger.warning("API key not found in secrets, using environment variable")
                            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
                        else:
                            logger.error(f"Error retrieving Anthropic API key: {str(e)}")
                            raise

                    portkey_headers = createHeaders(
                        api_key=self.PORTKEY_API_KEY,
                        provider="anthropic",
                        metadata={
                            "_user": self.user_id,
                            "environment": os.environ.get("ENV"),
                        },
                    )
                    logger.debug("Created Portkey headers for Anthropic")

                    if agent_type == AgentType.CREWAI:
                        logger.debug("Creating CrewAI Anthropic instance with Portkey")
                        self.llm = LLM(
                            model="anthropic/claude-3-haiku-20240307",
                            temperature=0.3,
                            api_key=anthropic_key,
                        )
                    else:
                        logger.debug("Creating Langchain Anthropic instance with Portkey")
                        self.llm = ChatAnthropic(
                            model="claude-3-haiku-20240307",
                            temperature=0.3,
                            api_key=anthropic_key,
                            base_url=PORTKEY_GATEWAY_URL,
                            default_headers=portkey_headers,
                        )

            elif preferred_provider == "google":
                logger.info("Initializing Google small LLM")
                if os.getenv("isDevelopmentMode") == "enabled":
                    logger.debug("Development mode enabled, using environment variables")
                    google_key = os.getenv("GOOGLE_API_KEY")
                    if not google_key:
                        logger.error("GOOGLE_API_KEY not found in environment variables")
                        raise ValueError("GOOGLE_API_KEY not found")
                    
                    if agent_type == AgentType.CREWAI:
                        logger.debug("Creating CrewAI Google instance")
                        self.llm = LLM(
                            model="gemini-pro",
                            api_key=google_key,
                            temperature=0.3,
                        )
                    else:
                        logger.debug("Creating Langchain Google instance")
                        self.llm = ChatGoogleGenerativeAI(
                            model="gemini-pro", 
                            model_name="gemini-pro",
                            temperature=0.3,
                            google_api_key=google_key,
                        )
                else:
                    logger.debug("Production mode, fetching API key from secret manager")
                    try:
                        secret = SecretManager.get_secret("google", self.user_id)
                        google_key = secret.get("api_key")
                        logger.debug("Successfully retrieved Google API key")
                    except Exception as e:
                        if "404" in str(e):
                            logger.warning("API key not found in secrets, using environment variable")
                            google_key = os.getenv("GOOGLE_API_KEY")
                        else:
                            logger.error(f"Error retrieving Google API key: {str(e)}")
                            raise

                    portkey_headers = createHeaders(
                        api_key=self.PORTKEY_API_KEY,
                        provider="google",
                        metadata={
                            "_user": self.user_id,
                            "environment": os.environ.get("ENV"),
                        },
                    )
                    logger.debug("Created Portkey headers for Google")

                    if agent_type == AgentType.CREWAI:
                        logger.debug("Creating CrewAI Google instance with Portkey")
                        self.llm = LLM(
                            model="gemini-pro",
                            api_key=google_key,
                            temperature=0.3,
                        )
                    else:
                        logger.debug("Creating Langchain Google instance with Portkey")
                        self.llm = ChatGoogleGenerativeAI(
                            model="gemini-pro", 
                            model_name="gemini-pro",
                            temperature=0.3,
                            google_api_key=google_key,
                            base_url=PORTKEY_GATEWAY_URL,
                            default_headers=portkey_headers,
                        )
            else:
                logger.error(f"Unsupported provider: {preferred_provider}")
                raise ValueError(f"Unsupported provider: {preferred_provider}")

            logger.info(f"Successfully initialized small LLM with provider: {preferred_provider}")
            return self.llm

        except Exception as e:
            logger.error(f"Error initializing small LLM: {str(e)}", exc_info=True)
            raise
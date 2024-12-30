import logging
import os
import subprocess

import sentry_sdk
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.base_model import Base
from app.core.database import SessionLocal, engine
from app.core.models import *  # noqa #necessary for models to not give import errors
from app.modules.auth.auth_router import auth_router
from app.modules.code_provider.github.github_router import router as github_router
from app.modules.conversations.conversations_router import (
    router as conversations_router,
)
from app.modules.intelligence.agents.agents_router import router as agent_router
from app.modules.intelligence.prompts.prompt_router import router as prompt_router
from app.modules.intelligence.prompts.system_prompt_setup import SystemPromptSetup
from app.modules.intelligence.provider.provider_router import router as provider_router
from app.modules.intelligence.tools.tool_router import router as tool_router
from app.modules.key_management.secret_manager import router as secret_manager_router
from app.modules.parsing.graph_construction.parsing_router import (
    router as parsing_router,
)
from app.modules.projects.projects_router import router as projects_router
from app.modules.search.search_router import router as search_router
from app.modules.usage.usage_router import router as usage_router
from app.modules.users.user_router import router as user_router
from app.modules.users.user_service import UserService
from app.modules.utils.firebase_setup import FirebaseSetup

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MainApp:
    def __init__(self):
        load_dotenv(override=True)
        if (
            os.getenv("isDevelopmentMode") == "enabled"
            and os.getenv("ENV") != "development"
        ):
            logging.error(
                "Development mode enabled but ENV is not set to development. Exiting."
            )
            exit(1)
        self.setup_sentry()
        self.app = FastAPI()
        self.setup_cors()
        self.initialize_database()
        self.check_and_set_env_vars()
        self.setup_data()
        self.include_routers()

    def setup_sentry(self):
        if os.getenv("ENV") == "production":
            sentry_sdk.init(
                dsn=os.getenv("SENTRY_DSN"),
                traces_sample_rate=0.25,
                profiles_sample_rate=1.0,
            )

    def setup_cors(self):
        origins = ["*"]
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def setup_data(self):
        if os.getenv("isDevelopmentMode") == "enabled":
            logging.info("Development mode enabled. Skipping Firebase setup.")
            # Setup dummy user for development mode
            db = SessionLocal()
            user_service = UserService(db)
            user_service.setup_dummy_user()
            db.close()
            logging.info("Dummy user created")
        else:
            FirebaseSetup.firebase_init()

    def initialize_database(self):
        # Initialize database tables
        Base.metadata.create_all(bind=engine)

    def check_and_set_env_vars(self):
        provider_keys = {
            "OPENAI": ["OPENAI_API_KEY"],
            "ANTHROPIC": ["ANTHROPIC_API_KEY"],
            "GOOGLE": ["GOOGLE_API_KEY"]
        }
        
        # Check which provider keys are set in the environment
        available_providers = []
        for provider, keys in provider_keys.items():
            if all(key in os.environ and os.environ[key] for key in keys):
                available_providers.append(provider)
                logging.info(f"{provider} API key found")
        
        # If no provider is configured, prompt for at least one
        if not available_providers:
            logging.warning("No API keys found for any provider")
            print("You need to configure at least one AI provider.")
            for provider, keys in provider_keys.items():
                configure = input(f"Would you like to configure {provider}? (y/n): ")
                if configure.lower() == 'y':
                    for key in keys:
                        if key not in os.environ:
                            value = input(f"Enter value for {key}: ")
                            os.environ[key] = value
                    available_providers.append(provider)
                    break
        
        # Set DEFAULT_LLM_PROVIDER if not already set
        if "DEFAULT_LLM_PROVIDER" not in os.environ and available_providers:
            os.environ["DEFAULT_LLM_PROVIDER"] = available_providers[0]
            logging.info(f"Default LLM provider automatically set to {available_providers[0]}")

    def include_routers(self):
        self.app.include_router(auth_router, prefix="/api/v1", tags=["Auth"])
        self.app.include_router(user_router, prefix="/api/v1", tags=["User"])
        self.app.include_router(parsing_router, prefix="/api/v1", tags=["Parsing"])
        self.app.include_router(
            conversations_router, prefix="/api/v1", tags=["Conversations"]
        )
        self.app.include_router(prompt_router, prefix="/api/v1", tags=["Prompts"])
        self.app.include_router(projects_router, prefix="/api/v1", tags=["Projects"])
        self.app.include_router(search_router, prefix="/api/v1", tags=["Search"])
        self.app.include_router(github_router, prefix="/api/v1", tags=["Github"])
        self.app.include_router(agent_router, prefix="/api/v1", tags=["Agents"])
        self.app.include_router(provider_router, prefix="/api/v1", tags=["Providers"])
        self.app.include_router(tool_router, prefix="/api/v1", tags=["Tools"])
        self.app.include_router(usage_router, prefix="/api/v1/usage", tags=["Usage"])
        if os.getenv("isDevelopmentMode") != "enabled":
            self.app.include_router(
                secret_manager_router, prefix="/api/v1", tags=["Secret Manager"]
            )

    def add_health_check(self):
        @self.app.get("/health", tags=["Health"])
        def health_check():
            return {
                "status": "ok",
                "version": subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"]
                )
                .strip()
                .decode("utf-8"),
            }

    async def startup_event(self):
        db = SessionLocal()
        try:
            system_prompt_setup = SystemPromptSetup(db)
            await system_prompt_setup.initialize_system_prompts()
            logging.info("System prompts initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize system prompts: {str(e)}")
            raise
        finally:
            db.close()

    def run(self):
        self.add_health_check()
        self.app.add_event_handler("startup", self.startup_event)
        return self.app


# Create an instance of MainApp and run it
main_app = MainApp()
app = main_app.run()

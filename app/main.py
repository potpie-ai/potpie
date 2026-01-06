import os
import subprocess

# Set TOKENIZERS_PARALLELISM before any tokenizer imports to prevent fork warnings
# This must be set before sentence-transformers or any HuggingFace tokenizers are used
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sentry_sdk
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router as potpie_api_router
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
from app.modules.media.media_router import router as media_router
from app.modules.integrations.integrations_router import router as integrations_router
from app.modules.parsing.graph_construction.parsing_router import (
    router as parsing_router,
)
from app.modules.projects.projects_router import router as projects_router
from app.modules.search.search_router import router as search_router
from app.modules.usage.usage_router import router as usage_router
from app.modules.users.user_router import router as user_router
from app.modules.users.user_service import UserService
from app.modules.utils.firebase_setup import FirebaseSetup
from app.modules.utils.logger import configure_logging, setup_logger
from app.modules.utils.logging_middleware import LoggingContextMiddleware

configure_logging()
logger = setup_logger(__name__)


class MainApp:
    def __init__(self):
        load_dotenv(override=True)
        if (
            os.getenv("isDevelopmentMode") == "enabled"
            and os.getenv("ENV") != "development"
        ):
            logger.error(
                "Development mode enabled but ENV is not set to development. Exiting."
            )
            exit(1)
        self.setup_sentry()
        self.setup_phoenix_tracing()
        self.app = FastAPI()
        self.setup_cors()
        self.setup_logging_middleware()
        self.include_routers()

    def setup_sentry(self):
        if os.getenv("ENV") == "production":
            try:
                # Explicitly configure integrations to avoid auto-enabling Strawberry
                # which causes crashes when Strawberry is not installed
                from sentry_sdk.integrations.fastapi import FastApiIntegration
                from sentry_sdk.integrations.logging import LoggingIntegration
                from sentry_sdk.integrations.stdlib import StdlibIntegration

                sentry_sdk.init(
                    dsn=os.getenv("SENTRY_DSN"),
                    traces_sample_rate=0.25,
                    profiles_sample_rate=1.0,
                    default_integrations=False,
                    integrations=[
                        FastApiIntegration(),
                        LoggingIntegration(),
                        StdlibIntegration(),
                    ],
                )
            except Exception:
                logger.exception(
                    "Sentry initialization failed (non-fatal but should be investigated)"
                )

    def setup_phoenix_tracing(self):
        try:
            from app.modules.intelligence.tracing.phoenix_tracer import (
                initialize_phoenix_tracing,
            )

            initialize_phoenix_tracing()
        except Exception:
            logger.exception(
                "Phoenix tracing initialization failed (non-fatal but should be investigated)"
            )

    def setup_cors(self):
        # Get allowed origins from environment variable, default to localhost:3000 for development
        allowed_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000")
        # Split by comma if multiple origins are provided
        origins = [origin.strip() for origin in allowed_origins_env.split(",")]

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info(f"CORS configured with allowed origins: {origins}")

    def setup_logging_middleware(self):
        """
        Add logging context middleware to automatically inject request-level context.

        This ensures all logs within a request automatically include:
        - request_id: Unique identifier for tracing
        - path: API endpoint path
        - user_id: Authenticated user (if available)

        Domain-specific IDs (conversation_id, project_id) should be added
        manually using log_context() in routes where available.
        """
        self.app.add_middleware(LoggingContextMiddleware)
        logger.info("Logging context middleware configured")

    def setup_data(self):
        if os.getenv("isDevelopmentMode") == "enabled":
            logger.info("Development mode enabled. Skipping Firebase setup.")
            # Setup dummy user for development mode
            db = SessionLocal()
            user_service = UserService(db)
            user_service.setup_dummy_user()
            db.close()
            logger.info("Dummy user created")
        else:
            FirebaseSetup.firebase_init()

    def initialize_database(self):
        # Initialize database tables
        Base.metadata.create_all(bind=engine)

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
        self.app.include_router(
            potpie_api_router, prefix="/api/v2", tags=["Potpie API"]
        )
        self.app.include_router(
            secret_manager_router, prefix="/api/v1", tags=["Secret Manager"]
        )
        self.app.include_router(media_router, prefix="/api/v1", tags=["Media"])
        self.app.include_router(
            integrations_router, prefix="/api/v1", tags=["Integrations"]
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
        # Database initialization moved here (runs on app start, not import)
        logger.info("Initializing database...")
        self.initialize_database()
        logger.info("Database initialized successfully")

        # Setup data (Firebase or dummy user)
        logger.info("Setting up application data...")
        self.setup_data()
        logger.info("Application data setup complete")

        # System prompts initialization
        db = SessionLocal()
        try:
            system_prompt_setup = SystemPromptSetup(db)
            await system_prompt_setup.initialize_system_prompts()
            logger.info("System prompts initialized successfully")
        except Exception:
            logger.exception("Failed to initialize system prompts")
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

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import Base, SessionLocal, engine
from app.modules.auth.auth_router import auth_router
from app.modules.conversations.conversations_router import (
    router as conversations_router,
)
from app.modules.github.github_router import router as github_router
from app.modules.intelligence.agents.agents_router import router as agent_router
from app.modules.intelligence.prompts.prompt_router import router as prompt_router
from app.modules.intelligence.prompts.system_prompt_setup import SystemPromptSetup
from app.modules.intelligence.provider.provider_router import router as provider_router
from app.modules.key_management.secret_manager import router as secret_manager_router
from app.modules.parsing.graph_construction.parsing_router import (
    router as parsing_router,
)
from app.modules.parsing.knowledge_graph.query_router import router as query_router
from app.modules.projects.projects_router import router as projects_router
from app.modules.search.search_router import router as search_router
from app.modules.users.user_router import router as user_router
from app.modules.utils.firebase_setup import FirebaseSetup

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
import nltk
nltk.download()

class MainApp:
    def __init__(self):
        load_dotenv(override=True)
        self.app = FastAPI()
        self.setup_cors()
        self.initialize_database()
        self.check_and_set_env_vars()
        if os.getenv("isDevelopmentMode") == "enabled":
            self.setup_data()
        else:
            FirebaseSetup.firebase_init()
        self.include_routers()


    def setup_cors(self):
        origins = ["*"]
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def initialize_database(self):
        # Initialize database tables
        Base.metadata.create_all(bind=engine)

    def check_and_set_env_vars(self):
        required_env_vars = [
            "OPENAI_API_KEY",
            "OPENAI_MODEL_REASONING",
        ]
        for env_var in required_env_vars:
            if env_var not in os.environ:
                value = input(f"Enter value for {env_var}: ")
                os.environ[env_var] = value

    def include_routers(self):
        self.app.include_router(auth_router, prefix="/api/v1", tags=["Auth"])
        self.app.include_router(user_router, prefix="/api/v1", tags=["User"])
        self.app.include_router(parsing_router, prefix="/api/v1", tags=["Parsing"])
        self.app.include_router(
            conversations_router, prefix="/api/v1", tags=["Conversations"]
        )
        self.app.include_router(prompt_router, prefix="/api/v1", tags=["Prompts"])
        self.app.include_router(
            secret_manager_router, prefix="/api/v1", tags=["Secret Manager"]
        )
        self.app.include_router(projects_router, prefix="/api/v1", tags=["Projects"])
        self.app.include_router(search_router, prefix="/api/v1", tags=["Search"])
        self.app.include_router(github_router, prefix="/api/v1", tags=["Github"])
        self.app.include_router(agent_router, prefix="/api/v1", tags=["Agents"])

        self.app.include_router(provider_router, prefix="/api/v1", tags=["Providers"])
        self.app.include_router(query_router, prefix="/api/v1", tags=["query"])

    def add_health_check(self):
        @self.app.get("/health", tags=["Health"])
        def health_check():
            return {"status": "ok"}

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

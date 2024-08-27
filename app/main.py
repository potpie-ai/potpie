import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import Base, engine
from app.core.mongo_manager import MongoManager
from app.modules.auth.auth_router import auth_router
from app.modules.conversations.conversations_router import (
    router as conversations_router,
)
from app.modules.key_management.secret_manager import router as secret_manager_router
from app.modules.parsing.graph_construction.parsing_router import (
    router as parsing_router,
)
from app.modules.search.search_router import router as search_router
from app.modules.users.user_router import router as user_router
from app.modules.projects.projects_router import router as projects_router
from app.modules.github.github_router import router as github_router
from app.modules.utils.dummy_setup import DummyDataSetup
from app.modules.utils.firebase_setup import FirebaseSetup

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
        self.verify_mongodb_connection()

    def verify_mongodb_connection(self):
        try:
            mongo_manager = MongoManager.get_instance()
            mongo_manager.verify_connection()
            logging.info("MongoDB connection verified successfully")
        except Exception as e:
            logging.error(f"Failed to verify MongoDB connection: {str(e)}")
            raise

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

    def setup_data(self):
        # Setup dummy user and project during application startup
        dummy_data_setup = DummyDataSetup()
        dummy_data_setup.setup_dummy_user()
        dummy_data_setup.setup_dummy_project()

    def include_routers(self):
        self.app.include_router(user_router, prefix="/api/v1", tags=["User"])
        self.app.include_router(
            conversations_router, prefix="/api/v1", tags=["Conversations"]
        )
        self.app.include_router(parsing_router, prefix="/api/v1", tags=["Parsing"])
        self.app.include_router(auth_router, prefix="/api/v1", tags=["Auth"])
        self.app.include_router(
            secret_manager_router, prefix="/api/v1", tags=["Secret Manager"]
        )

        self.app.include_router(projects_router, prefix="/api/v1", tags=["Projects"])
        self.app.include_router(search_router, prefix="/api/v1/search", tags=["search"])
        self.app.include_router(github_router, prefix="/api/v1", tags=["Github"])

    def add_health_check(self):
        @self.app.get("/health", tags=["Health"])
        def health_check():
            return {"status": "ok"}

    def run(self):
        self.add_health_check()
        return self.app


# Create an instance of MainApp and run it
main_app = MainApp()
app = main_app.run()


@app.on_event("shutdown")
def shutdown_event():
    MongoManager.close_connection()

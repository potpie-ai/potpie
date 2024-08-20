import os
import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import Base, engine

from app.modules.conversations.conversations_router import router as conversations_router
from app.modules.users.user_router import router as user_router

from app.modules.utils.dummy_setup import DummyDataSetup


class MainApp:
    def __init__(self):
        load_dotenv(override=True)
        self.app = FastAPI()
        self.setup_cors()
        self.initialize_database()
        self.check_and_set_env_vars()
        self.setup_data()
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

    def setup_data(self):

        # Setup dummy user and project during application startup
        dummy_data_setup = DummyDataSetup()
        dummy_data_setup.setup_dummy_user()
        dummy_data_setup.setup_dummy_project()

    def include_routers(self):
        self.app.include_router(user_router, prefix="/api/v1", tags=["User"])
        self.app.include_router(conversations_router, prefix="/api/v1", tags=["Conversations"])


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

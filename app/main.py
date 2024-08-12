import os
import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import Base, engine

from app.modules.conversations.router import router as conversations_router
from app.modules.users.router import router as user_router


load_dotenv(override=True)

# Initialize database tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_and_set_env_vars():
    required_env_vars = [
        "OPENAI_API_KEY",
        "OPENAI_MODEL_REASONING",
    ]
    for env_var in required_env_vars:
        if env_var not in os.environ:
            value = input(f"Enter value for {env_var}: ")
            os.environ[env_var] = value

    
check_and_set_env_vars()


@app.get("/health" ,tags=["Health"])
def health_check():
    return {"status": "ok"}


app.include_router(user_router,  prefix="/api/v1", tags=["User"])
app.include_router(conversations_router,  prefix="/api/v1", tags=["Conversations"])
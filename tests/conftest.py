import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the project root directory to Python path
load_dotenv(dotenv_path=".env.test")
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set development mode for testing
os.environ["isDevelopmentMode"] = "enabled"
os.environ["defaultUsername"] = "test-user"

import pytest
import httpx
from httpx import AsyncClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.core.base_model import Base
from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.usage.usage_service import UsageService
import pytest_asyncio

# Database setup (remains the same)
TEST_DATABASE_URL = "postgresql://postgres:mysecretpassword@localhost:5432/momentum_test"
engine = create_engine(TEST_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """
    Handles the safe creation and teardown of a dedicated test database
    for the entire test session. Includes safety checks to prevent deleting
    the development database.
    """
    TEST_DATABASE_URL = os.getenv("DATABASE_URL")
    if not TEST_DATABASE_URL:
        raise ValueError("DATABASE_URL for tests is not set in .env.test")

    db_name = TEST_DATABASE_URL.split('/')[-1]
    
    # --- SAFETY CHECKS ---
    # 1. Ensure the test database name is not the main database.
    if db_name == "momentum": # <-- IMPORTANT: Put your main DB name here
        raise ValueError(
            "FATAL: Test database name cannot be the same as the development database ('momentum')."
        )
    # 2. Enforce that the test database name must end with '_test'.
    if not db_name.endswith("_test"):
        raise ValueError(
            f"FATAL: To prevent accidental data loss, the test database name must end with '_test'. Got: '{db_name}'"
        )

    # Connect to the default 'postgres' database to create/drop the test DB
    default_db_url = TEST_DATABASE_URL.replace(f'/{db_name}', '/postgres')
    default_engine = create_engine(default_db_url, isolation_level="AUTOCOMMIT")

    # --- Teardown and Create ---
    with default_engine.connect() as conn:
        print(f"\n--- Dropping test database '{db_name}' (if it exists) ---")
        conn.execute(text(f"DROP DATABASE IF EXISTS {db_name} WITH (FORCE)"))
        print(f"--- Creating test database '{db_name}' ---")
        conn.execute(text(f"CREATE DATABASE {db_name}"))
    
    # --- Connect to the new test database to create tables ---
    engine = create_engine(TEST_DATABASE_URL)
    print(f"--- Creating tables in '{db_name}' ---")
    Base.metadata.create_all(bind=engine)

    # Yield control to the test session
    yield

    # --- Teardown: Drop the test database ---
    with default_engine.connect() as conn:
        print(f"\n--- Dropping test database '{db_name}' ---")
        conn.execute(text(f"DROP DATABASE {db_name} WITH (FORCE)"))

@pytest.fixture
def db_session():
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()

@pytest_asyncio.fixture
async def client(db_session):
    def override_get_db():
        yield db_session

    # Override only the properly injected services.
    # The test function will handle MediaService directly.
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[AuthService.check_auth] = lambda: {"user_id": "test-user", "email": "test@example.com"}
    app.dependency_overrides[UsageService.check_usage_limit] = lambda: True
    
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client
    
    app.dependency_overrides.clear()
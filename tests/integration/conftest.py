import pytest
from fastapi.testclient import TestClient
from fastapi import Depends
from sqlalchemy import create_engine, JSON, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Import the Base class that models inherit from (for production models)
from app.core.base_model import Base
from app.core.database import get_db
from app.main import app
from app.modules.auth.auth_service import AuthService

# Import production models (if needed in tests)
from app.modules.users.user_model import User
from app.modules.projects.projects_model import Project
from app.modules.conversations.conversation.conversation_model import Conversation
from app.modules.intelligence.prompts.prompt_model import Prompt
from app.modules.users.user_preferences_model import UserPreferences
# Import any other models that inherit from Base

# Create SQLite in-memory database URL
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def setup_database():
    """
    Create all tables in the test database using a test-specific Base and TestUser model.
    """
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy import Column, String, Boolean, TIMESTAMP
    from sqlalchemy.sql import func

    # Create a separate Base for testing
    TestBase = declarative_base()

    class TestUser(TestBase):
        __tablename__ = "users"

        uid = Column(String(255), primary_key=True)
        email = Column(String(255), unique=True, nullable=False)
        display_name = Column(String(255))
        email_verified = Column(Boolean, default=False)
        created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
        last_login_at = Column(TIMESTAMP(timezone=True), default=func.now())
        provider_info = Column(JSON)  # Using JSON for SQLite testing
        provider_username = Column(String(255))

    print("Creating database tables...")
    TestBase.metadata.create_all(bind=engine)
    print("Tables created:", inspect(engine).get_table_names())

    # Store the TestUser class and metadata for later use (e.g. in fixtures and cleanup)
    setup_database.TestUser = TestUser
    setup_database.metadata = TestBase.metadata

@pytest.fixture(scope="function")
def db_session():
    """
    Create a fresh database session for each test.
    """
    setup_database()  # Create tables using the test-specific Base
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        # Drop all tables created by setup_database (using the correct metadata)
        if hasattr(setup_database, "metadata"):
            setup_database.metadata.drop_all(bind=engine)

def get_mock_user():
    """Return consistent mock user data."""
    return {
        "user_id": "test-user-id",
        "email": "test@example.com",
        "provider_username": "test-github-user"
    }

@pytest.fixture(scope="function")
def mock_auth_middleware():
    """Mock the authentication middleware by overriding AuthService.check_auth."""
    def mock_check_auth():
        return get_mock_user()

    app.dependency_overrides[AuthService.check_auth] = mock_check_auth

    yield

    # Cleanup: remove the override if it exists
    app.dependency_overrides.pop(AuthService.check_auth, None)

@pytest.fixture(scope="function")
def client(mock_auth_middleware, db_session):
    """
    Provide a TestClient with dependency overrides:
      - Override get_db to use the test database session.
      - Add a Bearer token to the headers.
    """
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    client_instance = TestClient(app)
    token = "testtoken"  # Use a test token (or generate dynamically)
    client_instance.headers.update({"Authorization": f"Bearer {token}"})

    try:
        yield client_instance
    finally:
        client_instance.close()
        app.dependency_overrides.clear()

@pytest.fixture(scope="function")
def mock_auth_user(db_session):
    """
    Create a mock authenticated user in the test database using the test-specific TestUser model.
    """
    print("Creating mock user...")
    test_user = setup_database.TestUser(
        uid="test-user-id",
        email="test@example.com",
        provider_username="test-github-user",
        provider_info={"access_token": "mock-github-token"}
    )
    db_session.add(test_user)
    db_session.commit()
    db_session.refresh(test_user)
    print(f"Mock user created with ID: {test_user.uid}")
    return test_user

@pytest.fixture(scope="function")
def client_without_auth(db_session):
    """
    Provide a TestClient without applying the authentication middleware override.
    """
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()

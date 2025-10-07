# In tests/conftest.py

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from unittest.mock import MagicMock

load_dotenv(dotenv_path=".env.test")
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ["isDevelopmentMode"] = "enabled"
os.environ["defaultUsername"] = "test-user"

import pytest
import pytest_asyncio
import httpx
from httpx import AsyncClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.main import app
from app.core.base_model import Base
from app.core.database import get_db, get_async_db
from app.modules.auth.auth_service import AuthService
from app.modules.usage.usage_service import UsageService
from app.modules.users.user_model import User
from app.modules.projects.projects_model import Project
from app.modules.conversations.conversation.conversation_model import Conversation, ConversationStatus
from app.modules.conversations.utils.redis_streaming import RedisStreamManager

# =================================================================
# 1. DATABASE SETUP FIXTURE
# =================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """ Handles the safe creation and teardown of the test database for the entire session. """
    TEST_DATABASE_URL = os.getenv("DATABASE_URL")
    if not TEST_DATABASE_URL or not TEST_DATABASE_URL.endswith("_test"):
        raise ValueError("FATAL: DATABASE_URL must be set in .env.test and end with '_test'.")
    
    db_name = TEST_DATABASE_URL.split('/')[-1]
    default_db_url = TEST_DATABASE_URL.replace(f'/{db_name}', '/postgres')
    
    with create_engine(default_db_url, isolation_level="AUTOCOMMIT").connect() as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS {db_name} WITH (FORCE)"))
        conn.execute(text(f"CREATE DATABASE {db_name}"))
    
    engine = create_engine(TEST_DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    yield
    with create_engine(default_db_url, isolation_level="AUTOCOMMIT").connect() as conn:
        conn.execute(text(f"DROP DATABASE {db_name} WITH (FORCE)"))

# =================================================================
# 2. SESSION FIXTURES (SYNC & ASYNC)
# =================================================================

@pytest.fixture(scope="function")
def db_session() -> Session:
    """ Provides a standard synchronous SQLAlchemy session for tests. """
    TEST_DATABASE_URL = os.getenv("DATABASE_URL")
    engine = create_engine(TEST_DATABASE_URL)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()

@pytest_asyncio.fixture(scope="function")
async def async_db_session() -> AsyncSession:
    """ Provides an async SQLAlchemy session for tests. """
    ASYNC_TEST_DATABASE_URL = os.getenv("DATABASE_URL").replace("postgresql://", "postgresql+asyncpg://")
    engine = create_async_engine(ASYNC_TEST_DATABASE_URL)
    AsyncTestingSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with AsyncTestingSessionLocal() as session:
        yield session
    await engine.dispose()

# =================================================================
# 3. MOCK FIXTURES (CELERY & REDIS) - RESTORED
# =================================================================

@pytest.fixture
def mock_celery_tasks(monkeypatch):
    """ Mocks the .delay() method of our Celery tasks. """
    mock_execute = MagicMock()
    mock_regenerate = MagicMock()
    
    monkeypatch.setattr("app.celery.tasks.agent_tasks.execute_agent_background.delay", mock_execute)
    monkeypatch.setattr("app.celery.tasks.agent_tasks.execute_regenerate_background.delay", mock_regenerate)
    
    return {"execute": mock_execute, "regenerate": mock_regenerate}

@pytest.fixture
def mock_redis_stream_manager(monkeypatch):
    """ Mocks the RedisStreamManager to prevent actual Redis connections. """
    mock_manager = MagicMock(spec=RedisStreamManager)
    mock_manager.wait_for_task_start.return_value = True
    mock_manager.redis_client = MagicMock()
    mock_manager.redis_client.exists.return_value = False
    
    monkeypatch.setattr(
        "app.modules.conversations.utils.redis_streaming.RedisStreamManager", 
        lambda: mock_manager
    )
    return mock_manager

# =================================================================
# 4. PREREQUISITE DATA FIXTURES
# =================================================================

@pytest.fixture(scope="function")
def setup_test_user_committed(db_session: Session):
    """ Creates a test user and commits it so it's visible to all transactions. """
    user = db_session.query(User).filter_by(uid="test-user").one_or_none()
    if not user:
        user = User(uid="test-user", email="test@example.com")
        db_session.add(user)
        db_session.commit()
    return user

@pytest.fixture(scope="function")
def setup_test_project_committed(db_session: Session, setup_test_user_committed: User):
    """ Creates a prerequisite Project record with a valid status. """
    project = db_session.query(Project).filter_by(id="project-id-123").one_or_none()
    if not project:
        project = Project(
            id="project-id-123",
            user_id=setup_test_user_committed.uid,
            repo_name="Test Project Repo",
            status="ready" # Valid status from your model's CheckConstraint
        )
        db_session.add(project)
        db_session.commit()
    return project

@pytest.fixture(scope="function")
def setup_test_conversation_committed(db_session: Session, setup_test_project_committed: Project):
    """ Creates a prerequisite Conversation record. """
    conversation = db_session.query(Conversation).filter_by(id="test-convo-123").one_or_none()
    if not conversation:
        conversation = Conversation(
            id="test-convo-123",
            user_id="test-user",
            project_ids=["project-id-123"],
            agent_ids=["default-chat-agent"],
            title="Initial Test Conversation",
            status=ConversationStatus.ACTIVE 
        )
        db_session.add(conversation)
        db_session.commit()
    return conversation

# =================================================================
# 5. HTTP TEST CLIENT FIXTURE
# =================================================================

@pytest_asyncio.fixture
async def client(db_session: Session, async_db_session: AsyncSession):
    """ The main test client, correctly overriding both sync and async DB dependencies. """
    def override_get_db():
        yield db_session
    async def override_get_async_db():
        yield async_db_session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_async_db] = override_get_async_db
    app.dependency_overrides[AuthService.check_auth] = lambda: {"user_id": "test-user", "email": "test@example.com"}
    app.dependency_overrides[UsageService.check_usage_limit] = lambda: True
    
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c
    
    app.dependency_overrides.clear()
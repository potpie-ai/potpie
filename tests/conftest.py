# In tests/conftest.py

import os
import sys
import time
import redis
import httpx
import urllib.parse
from pathlib import Path
from dotenv import load_dotenv
from unittest.mock import MagicMock

# --- Basic Setup ---
load_dotenv()  # Load .env for all environment variables
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.environ["isDevelopmentMode"] = "enabled"
os.environ["defaultUsername"] = "test-user"

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# --- App and Model Imports ---
from app.main import app
from app.core.base_model import Base
from app.core.database import get_db, get_async_db
from app.modules.auth.auth_service import AuthService
from app.modules.usage.usage_service import UsageService
from app.modules.users.user_model import User
from app.modules.projects.projects_model import Project
from app.modules.conversations.conversation.conversation_model import (
    Conversation,
    ConversationStatus,
)
from app.modules.conversations.utils.redis_streaming import RedisStreamManager
from app.modules.code_provider.github.github_service import GithubService


# =================================================================
# 1. PYTEST CONFIGURATION & TEST GATING
# =================================================================


def pytest_configure(config):
    """Registers the 'github_live' marker with pytest."""
    config.addinivalue_line(
        "markers", "github_live: marks tests that hit the live GitHub API"
    )


@pytest.fixture(scope="session", autouse=True)
def require_github_tokens():
    """Skips 'github_live' tests if the GH_TOKEN_LIST environment variable is not set."""
    # This fixture runs for all sessions but only skips tests with the 'github_live' marker.
    # We can check for the marker in the request if we want to be more specific, but this is fine.
    if not os.environ.get("GH_TOKEN_LIST"):
        pytest.skip(
            "Skipping live GitHub tests: GH_TOKEN_LIST environment variable not set."
        )


@pytest.fixture(scope="session")
def require_private_repo_secrets():
    """Gating fixture that skips tests needing private repo secrets if they are not configured."""
    if not os.environ.get("PRIVATE_TEST_REPO_NAME"):
        pytest.skip("Skipping private repo tests: PRIVATE_TEST_REPO_NAME not set.")


# =================================================================
# 2. DATABASE SETUP & SESSIONS
# =================================================================


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Handles the safe, automatic creation and teardown of a dedicated test database."""
    main_db_url = os.getenv("POSTGRES_SERVER")  # Use the standard DATABASE_URL
    if not main_db_url:
        raise ValueError("FATAL: POSTGRES_SERVER not found in your .env file.")

    parsed_url = urllib.parse.urlparse(main_db_url)
    main_db_name = parsed_url.path.lstrip("/")
    if not main_db_name or "test" in main_db_name:
        raise ValueError(f"FATAL: Main database '{main_db_name}' looks like a test DB.")

    test_db_name = f"{main_db_name}_test"
    test_db_url = parsed_url._replace(path=f"/{test_db_name}").geturl()
    default_db_url = parsed_url._replace(path="/postgres").geturl()

    with create_engine(default_db_url, isolation_level="AUTOCOMMIT").connect() as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS {test_db_name} WITH (FORCE)"))
        conn.execute(text(f"CREATE DATABASE {test_db_name}"))

    engine = create_engine(test_db_url)
    Base.metadata.create_all(bind=engine)
    os.environ["DATABASE_URL"] = test_db_url
    yield
    with create_engine(default_db_url, isolation_level="AUTOCOMMIT").connect() as conn:
        conn.execute(text(f"DROP DATABASE {test_db_name} WITH (FORCE)"))


@pytest.fixture(scope="function")
def db_session() -> Session:
    """Provides a standard synchronous SQLAlchemy session for tests."""
    engine = create_engine(os.getenv("DATABASE_URL"))
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest_asyncio.fixture(scope="function")
async def async_db_session() -> AsyncSession:
    """Provides an async SQLAlchemy session for tests."""
    ASYNC_DB_URL = os.getenv("DATABASE_URL").replace(
        "postgresql://", "postgresql+asyncpg://"
    )
    engine = create_async_engine(ASYNC_DB_URL)
    AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)
    async with AsyncSessionLocal() as session:
        yield session
    await engine.dispose()


# =================================================================
# 3. MOCKS & FAKES
# =================================================================


class FakeRedis:
    """A lightweight, in-memory fake of a Redis client for testing caching."""

    def __init__(self):
        self._store = {}

    def get(self, key):
        entry = self._store.get(key)
        if not entry:
            return None
        value, expires_at = entry
        if expires_at and time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def setex(self, key, ttl, val):
        self._store[key] = (
            val.encode("utf-8") if isinstance(val, str) else val,
            time.time() + ttl if ttl else None,
        )


@pytest.fixture
def mock_celery_tasks(monkeypatch):
    """Mocks the .delay() method of Celery tasks for conversation tests."""
    mock_execute = MagicMock()
    mock_regenerate = MagicMock()
    monkeypatch.setattr(
        "app.celery.tasks.agent_tasks.execute_agent_background.delay", mock_execute
    )
    monkeypatch.setattr(
        "app.celery.tasks.agent_tasks.execute_regenerate_background.delay",
        mock_regenerate,
    )
    return {"execute": mock_execute, "regenerate": mock_regenerate}


@pytest.fixture
def mock_redis_stream_manager(monkeypatch):
    """Mocks the RedisStreamManager for conversation streaming tests."""
    mock_manager = MagicMock(spec=RedisStreamManager)
    mock_manager.wait_for_task_start.return_value = True
    mock_manager.redis_client = MagicMock()
    mock_manager.redis_client.exists.return_value = False
    monkeypatch.setattr(
        "app.modules.conversations.utils.redis_streaming.RedisStreamManager",
        lambda: mock_manager,
    )
    return mock_manager


# =================================================================
# 4. PREREQUISITE DATA FIXTURES
# =================================================================


@pytest.fixture(scope="function")
def setup_test_user_committed(db_session: Session):
    """Creates a single test user, visible to all transactions."""
    user = db_session.query(User).filter_by(uid="test-user").one_or_none()
    if not user:
        user = User(uid="test-user", email="test@example.com")
        db_session.add(user)
        db_session.commit()
    return user


@pytest.fixture(scope="function")
def conversation_project(db_session: Session, setup_test_user_committed: User):
    """RENAMED: Creates a prerequisite Project record for conversation tests."""
    project = db_session.query(Project).filter_by(id="project-id-123").one_or_none()
    if not project:
        project = Project(
            id="project-id-123",
            user_id=setup_test_user_committed.uid,
            repo_name="Test Project Repo",
            status="ready",
        )
        db_session.add(project)
        db_session.commit()
    return project


@pytest.fixture(scope="function")
def setup_test_conversation_committed(
    db_session: Session, conversation_project: Project
):
    """Creates a prerequisite Conversation record for message/regenerate tests."""
    convo = db_session.query(Conversation).filter_by(id="test-convo-123").one_or_none()
    if not convo:
        convo = Conversation(
            id="test-convo-123",
            user_id="test-user",
            project_ids=[conversation_project.id],
            agent_ids=["default-chat-agent"],
            title="Initial Test Convo",
            status=ConversationStatus.ACTIVE,
        )
        db_session.add(convo)
        db_session.commit()
    return convo


@pytest.fixture
def hello_world_project(db_session: Session, setup_test_user_committed: User):
    """Creates a Project record pointing to the live 'octocat/Hello-World' repo."""
    project = (
        db_session.query(Project).filter_by(id="live-hello-world-proj").one_or_none()
    )
    if not project:
        project = Project(
            id="live-hello-world-proj",
            user_id=setup_test_user_committed.uid,
            repo_name="octocat/Hello-World",
            branch_name="master",
            status="ready",
        )
        db_session.add(project)
        db_session.commit()
        db_session.refresh(project)
    return project


@pytest.fixture
def private_project_committed(db_session: Session, setup_test_user_committed: User):
    """Creates a Project record pointing to the private test repo from env vars."""
    repo_name = os.environ.get("PRIVATE_TEST_REPO_NAME")
    project = db_session.query(Project).filter_by(repo_name=repo_name).one_or_none()
    if not project:
        project = Project(
            id="live-project-private",
            user_id=setup_test_user_committed.uid,
            repo_name=repo_name,
            branch_name="main",
            status="ready",
        )
        db_session.add(project)
        db_session.commit()
    return project


@pytest.fixture
def ensure_default_branch(db_session: Session, hello_world_project: Project):
    """Dynamically updates the hello_world_project's branch name to match the live repo."""
    from github import Github as PyGithub

    tokens = os.environ.get("GH_TOKEN_LIST", "").split(",")
    gh = PyGithub(tokens[0].strip())
    repo = gh.get_repo("octocat/Hello-World")
    if hello_world_project.branch_name != repo.default_branch:
        hello_world_project.branch_name = repo.default_branch
        db_session.commit()
    return repo.default_branch


# =================================================================
# 5. SERVICE & CLIENT FIXTURES
# =================================================================


@pytest.fixture
def github_service_with_fake_redis(monkeypatch, db_session: Session):
    """Provides a GithubService instance with FakeRedis and App Auth disabled."""
    monkeypatch.setattr(redis, "from_url", lambda *args, **kwargs: FakeRedis())
    monkeypatch.delenv("GITHUB_APP_ID", raising=False)
    service = GithubService(db_session)
    return service


@pytest_asyncio.fixture
async def client(db_session: Session, async_db_session: AsyncSession):
    """The main FastAPI test client with all necessary dependency overrides."""

    def override_get_db():
        yield db_session

    async def override_get_async_db():
        yield async_db_session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_async_db] = override_get_async_db
    app.dependency_overrides[AuthService.check_auth] = lambda: {
        "user_id": "test-user",
        "email": "test@example.com",
    }
    app.dependency_overrides[UsageService.check_usage_limit] = lambda: True
    async with AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    app.dependency_overrides.clear()

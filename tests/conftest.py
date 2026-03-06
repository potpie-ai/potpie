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
from starlette.requests import Request
from starlette.responses import Response
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# --- App and Model Imports ---
from datetime import datetime, timedelta, timezone

from app.main import app
from app.core.base_model import Base
from app.core.database import get_db, get_async_db
from app.modules.auth.auth_service import AuthService
from app.modules.auth.auth_provider_model import (
    UserAuthProvider,
    PendingProviderLink,
    OrganizationSSOConfig,
    AuthAuditLog,
)
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


# Markers (github_live, stress, unit, integration) are defined in pyproject.toml.


@pytest.fixture(scope="session")
def require_github_tokens():
    """Skips 'github_live' tests if the GH_TOKEN_LIST environment variable is not set.
    Not autouse: only request this fixture in test modules that use @pytest.mark.github_live
    (e.g. integration_tests/code_provider) so unit tests can run without tokens."""
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


@pytest.fixture(scope="session")
def setup_test_database():
    """Creates and teardowns a dedicated test database. Only used by tests that need DB.
    Not autouse: pure unit tests (e.g. parsing schema, content_hash) run without Postgres."""
    main_db_url = os.getenv("POSTGRES_SERVER")
    if not main_db_url:
        pytest.skip(
            "POSTGRES_SERVER not set. Skipping tests that require database."
        )

    parsed_url = urllib.parse.urlparse(main_db_url)
    main_db_name = parsed_url.path.lstrip("/")
    if not main_db_name or "test" in main_db_name:
        pytest.skip(
            f"Main database '{main_db_name}' looks like a test DB or is empty."
        )

    test_db_name = f"{main_db_name}_test"
    test_db_url = parsed_url._replace(path=f"/{test_db_name}").geturl()
    default_db_url = parsed_url._replace(path="/postgres").geturl()

    try:
        with create_engine(default_db_url, isolation_level="AUTOCOMMIT").connect() as conn:
            conn.execute(text(f"DROP DATABASE IF EXISTS {test_db_name} WITH (FORCE)"))
            conn.execute(text(f"CREATE DATABASE {test_db_name}"))
    except OperationalError as e:
        pytest.skip(
            f"Postgres unreachable (is it running?): {e}"
        )

    engine = create_engine(test_db_url)
    Base.metadata.create_all(bind=engine)
    os.environ["DATABASE_URL"] = test_db_url
    yield
    try:
        with create_engine(default_db_url, isolation_level="AUTOCOMMIT").connect() as conn:
            conn.execute(text(f"DROP DATABASE {test_db_name} WITH (FORCE)"))
    except OperationalError:
        pass


@pytest.fixture(scope="function")
def db_session(setup_test_database) -> Session:
    """Provides a standard synchronous SQLAlchemy session for tests."""
    engine = create_engine(os.getenv("DATABASE_URL"))
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest_asyncio.fixture(scope="function")
async def async_db_session(setup_test_database) -> AsyncSession:
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
    # Celery AsyncResult-like objects expose a string `id` used for Redis mapping.
    mock_execute.return_value.id = "test-task-id-execute"
    mock_regenerate.return_value.id = "test-task-id-regenerate"
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
    # End the stream immediately so StreamingResponse does not hang in tests.
    mock_manager.consume_stream.return_value = iter(
        [
            {"type": "queued"},
            {"type": "end"},
        ]
    )
    # Patch both the source module and the module where it's imported/used.
    monkeypatch.setattr(
        "app.modules.conversations.utils.redis_streaming.RedisStreamManager",
        lambda: mock_manager,
    )
    monkeypatch.setattr(
        "app.modules.conversations.utils.conversation_routing.RedisStreamManager",
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


# --- Auth fixtures (for tests/unit/auth/, single conftest) ---


@pytest.fixture(scope="function")
def test_user(db_session: Session):
    """Create a canonical test user (uid=test-user)."""
    user = db_session.query(User).filter_by(uid="test-user").one_or_none()
    if not user:
        user = User(
            uid="test-user",
            email="test@example.com",
            display_name="Test User",
            email_verified=True,
            created_at=datetime.now(timezone.utc),
            last_login_at=datetime.now(timezone.utc),
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
    return user


@pytest.fixture(scope="function")
def test_user_with_github(db_session: Session, test_user: User):
    """Create a test user with GitHub provider for auth unit tests."""
    existing = (
        db_session.query(UserAuthProvider)
        .filter_by(user_id=test_user.uid, provider_type="firebase_github")
        .first()
    )
    if not existing:
        provider = UserAuthProvider(
            user_id=test_user.uid,
            provider_type="firebase_github",
            provider_uid="github-123",
            provider_data={"login": "testuser"},
            is_primary=True,
            linked_at=datetime.now(timezone.utc),
            last_used_at=datetime.now(timezone.utc),
        )
        db_session.add(provider)
        db_session.commit()
        db_session.refresh(test_user)
    return test_user


@pytest.fixture(scope="function")
def test_user_with_multiple_providers(db_session: Session, test_user_with_github: User):
    """Create a test user with multiple providers for auth unit tests."""
    existing = (
        db_session.query(UserAuthProvider)
        .filter_by(user_id=test_user_with_github.uid, provider_type="sso_google")
        .first()
    )
    if not existing:
        google_provider = UserAuthProvider(
            user_id=test_user_with_github.uid,
            provider_type="sso_google",
            provider_uid="google-456",
            provider_data={"email": "test@example.com"},
            is_primary=False,
            linked_at=datetime.now(timezone.utc),
            last_used_at=datetime.now(timezone.utc),
        )
        db_session.add(google_provider)
        db_session.commit()
        db_session.refresh(test_user_with_github)
    return test_user_with_github


@pytest.fixture(scope="function")
def pending_link(db_session: Session, test_user: User):
    """Create a pending provider link for auth unit tests."""
    link = db_session.query(PendingProviderLink).filter_by(token="test-linking-token-123").first()
    if not link:
        link = PendingProviderLink(
            user_id=test_user.uid,
            provider_type="sso_google",
            provider_uid="google-789",
            provider_data={"email": "test@example.com"},
            token="test-linking-token-123",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
            ip_address="127.0.0.1",
        )
        db_session.add(link)
        db_session.commit()
        db_session.refresh(link)
    return link


@pytest.fixture(scope="function")
def org_sso_config(db_session: Session):
    """Create organization SSO config for auth unit tests."""
    config = db_session.query(OrganizationSSOConfig).filter_by(domain="company.com").first()
    if not config:
        config = OrganizationSSOConfig(
            domain="company.com",
            organization_name="Test Company",
            sso_provider="google",
            sso_config={"client_id": "test-client-id"},
            enforce_sso=True,
            allow_other_providers=False,
            configured_at=datetime.now(timezone.utc),
            is_active=True,
        )
        db_session.add(config)
        db_session.commit()
        db_session.refresh(config)
    return config


@pytest.fixture
def auth_token():
    """Mock auth token for tests that pass Authorization header (e.g. auth router)."""
    return "mock-firebase-token"


# =================================================================
# NEO4J FIXTURES (for real_parse tests)
# =================================================================


@pytest.fixture(scope="session")
def neo4j_config():
    """Returns Neo4j config from env vars; skips if not available or unreachable."""
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri or not username or not password:
        pytest.skip(
            "NEO4J_URI, NEO4J_USERNAME, or NEO4J_PASSWORD not set. Skipping real_parse tests."
        )

    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable

    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        driver.close()
    except ServiceUnavailable as e:
        pytest.skip(f"Neo4j unreachable at {uri}: {e}")
    except Exception as e:
        pytest.skip(f"Neo4j connection error: {e}")

    return {"uri": uri, "username": username, "password": password}


@pytest.fixture(scope="session")
def test_repo_path(tmp_path_factory):
    """Creates a small, deterministic test repository for real parse tests.
    Contains a few Python files so language detection returns 'python'."""
    repo_dir = tmp_path_factory.mktemp("test_repo")

    # Create a minimal Python project
    (repo_dir / "main.py").write_text(
        '''"""Main module for test repo."""

def greet(name: str) -> str:
    """Return a greeting."""
    return f"Hello, {name}!"


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


if __name__ == "__main__":
    print(greet("World"))
'''
    )

    (repo_dir / "utils.py").write_text(
        '''"""Utility functions."""

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def is_even(n: int) -> bool:
    """Check if a number is even."""
    return n % 2 == 0
'''
    )

    (repo_dir / "models.py").write_text(
        '''"""Data models."""

class User:
    """Simple user model."""

    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

    def __repr__(self):
        return f"User(name={self.name!r}, email={self.email!r})"
'''
    )

    # Create a subdirectory with more files
    subdir = repo_dir / "lib"
    subdir.mkdir()
    (subdir / "__init__.py").write_text('"""Lib package."""\n')
    (subdir / "helpers.py").write_text(
        '''"""Helper functions."""

def clamp(value: int, min_val: int, max_val: int) -> int:
    """Clamp a value to a range."""
    return max(min_val, min(value, max_val))
'''
    )

    # Initialize as a Git repo so clone_or_copy_repository can open it with GitPython
    import subprocess
    subprocess.run(
        ["git", "init"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "add", "."],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )

    return str(repo_dir)


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
    async def override_check_auth(
        request: Request,
        res: Response,
        credential=None,
    ):
        # Same signature as AuthService.check_auth so FastAPI can call as drop-in replacement.
        return {
            "user_id": "test-user",
            "email": "test@example.com",
        }

    app.dependency_overrides[AuthService.check_auth] = override_check_auth
    app.dependency_overrides[UsageService.check_usage_limit] = lambda: True
    async with AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        follow_redirects=True,
    ) as c:
        yield c
    app.dependency_overrides.clear()

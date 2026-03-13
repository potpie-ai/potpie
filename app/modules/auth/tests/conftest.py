"""
Pytest fixtures for auth tests.

Uses PostgreSQL (from POSTGRES_SERVER in .env) because app models use
PostgreSQL-specific types (e.g. JSONB). SQLite cannot run these tests.
"""

import os
import urllib.parse

import pytest
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from fastapi.testclient import TestClient

from dotenv import load_dotenv

load_dotenv()

# Import app so all models are registered on Base (required for create_all + relationships)
from app.main import app
from app.core.database import get_db
from app.modules.auth.auth_service import AuthService

from app.core.base_model import Base
from app.modules.users.user_model import User
from app.modules.auth.auth_provider_model import (
    UserAuthProvider,
    PendingProviderLink,
    OrganizationSSOConfig,
    AuthAuditLog,
)


@pytest.fixture(scope="session")
def auth_test_database():
    """Create and tear down a dedicated Postgres test database for auth tests."""
    main_db_url = os.getenv("POSTGRES_SERVER")
    if not main_db_url:
        pytest.skip(
            "POSTGRES_SERVER not set. Auth tests require PostgreSQL (models use JSONB)."
        )

    parsed = urllib.parse.urlparse(main_db_url)
    main_db_name = parsed.path.lstrip("/")
    if not main_db_name or "test" in main_db_name:
        pytest.skip(
            f"POSTGRES_SERVER database name looks like a test DB: {main_db_name!r}"
        )

    test_db_name = f"{main_db_name}_test"
    test_db_url = parsed._replace(path=f"/{test_db_name}").geturl()
    default_db_url = parsed._replace(path="/postgres").geturl()

    with create_engine(default_db_url, isolation_level="AUTOCOMMIT").connect() as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS {test_db_name} WITH (FORCE)"))
        conn.execute(text(f"CREATE DATABASE {test_db_name}"))

    engine = create_engine(test_db_url)
    Base.metadata.create_all(bind=engine)
    os.environ["DATABASE_URL"] = test_db_url
    yield test_db_url
    with create_engine(default_db_url, isolation_level="AUTOCOMMIT").connect() as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS {test_db_name} WITH (FORCE)"))


@pytest.fixture(scope="function")
def db_session(auth_test_database) -> Session:
    """Create a test database session using the Postgres test database.
    Each test runs in a transaction that is rolled back, so tests are isolated.
    """
    test_db_url = os.getenv("DATABASE_URL")
    engine = create_engine(test_db_url)
    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(autocommit=False, autoflush=False, bind=connection)()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def client(db_session):
    """FastAPI test client using the full app with DB and auth overrides.
    Auth router only uses get_db; we override that and check_auth for protected routes.
    """
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[AuthService.check_auth] = lambda: {
        "user_id": "test-user-123",
        "email": "test@example.com",
    }
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def auth_token(test_user):
    """Token string for Authorization header; check_auth is overridden to return test_user."""
    return "mock-firebase-token"


@pytest.fixture
def test_user(db_session):
    """Create a test user"""
    user = User(
        uid="test-user-123",
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


@pytest.fixture
def test_user_with_github(db_session, test_user):
    """Create a test user with GitHub provider"""
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
    db_session.refresh(provider)
    return test_user


@pytest.fixture
def test_user_with_multiple_providers(db_session, test_user_with_github):
    """Create a test user with multiple providers"""
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
    return test_user_with_github


@pytest.fixture
def pending_link(db_session, test_user):
    """Create a pending provider link"""
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


@pytest.fixture
def org_sso_config(db_session):
    """Create organization SSO config"""
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

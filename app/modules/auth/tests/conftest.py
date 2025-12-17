"""
Pytest fixtures for auth tests
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.base_model import Base
from app.modules.users.user_model import User
from app.modules.auth.auth_provider_model import (
    UserAuthProvider,
    PendingProviderLink,
    OrganizationSSOConfig,
    AuthAuditLog,
)


@pytest.fixture(scope="function")
def db_session():
    """Create a test database session"""
    # Use in-memory SQLite for testing
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create session
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    
    yield session
    
    # Cleanup
    session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_user(db_session):
    """Create a test user"""
    user = User(
        uid="test-user-123",
        email="test@example.com",
        display_name="Test User",
        email_verified=True,
        created_at=datetime.utcnow(),
        last_login_at=datetime.utcnow(),
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
        linked_at=datetime.utcnow(),
        last_used_at=datetime.utcnow(),
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
        linked_at=datetime.utcnow(),
        last_used_at=datetime.utcnow(),
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
        expires_at=datetime.utcnow() + timedelta(minutes=15),
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
        configured_at=datetime.utcnow(),
        is_active=True,
    )
    db_session.add(config)
    db_session.commit()
    db_session.refresh(config)
    return config


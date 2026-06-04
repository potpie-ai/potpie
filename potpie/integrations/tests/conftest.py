"""Shared fixtures for potpie-integrations HTTP integration tests.

Mounts the legacy FastAPI app because integrations routers depend on
``app.core`` and ``app.modules.auth``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.orm import Session
from starlette.requests import Request
from starlette.responses import Response

LEGACY_ROOT = Path(__file__).resolve().parents[3] / "legacy"
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

os.environ.setdefault("ENV", "development")
os.environ.setdefault("isDevelopmentMode", "enabled")
os.environ.setdefault("defaultUsername", "test-user")
os.environ.setdefault("LINEAR_CLIENT_ID", "test-linear-client-id")
os.environ.setdefault("LINEAR_CLIENT_SECRET", "test-linear-client-secret")
os.environ.setdefault("SENTRY_CLIENT_ID", "test-sentry-client-id")
os.environ.setdefault("SENTRY_CLIENT_SECRET", "test-sentry-client-secret")
os.environ.setdefault("OAUTH_STATE_SECRET", "test-oauth-state-secret")


@pytest.fixture(scope="session")
def _db_engine():
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import OperationalError

    from app.core.base_model import Base
    import app.core.models  # noqa: F401

    url = os.environ.get(
        "POSTGRES_SERVER", "postgresql://postgres:postgres@localhost:5432/postgres"
    )
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as exc:
        pytest.skip(f"Postgres not available for integrations tests: {exc}")
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(_db_engine) -> Session:
    from sqlalchemy.orm import sessionmaker

    connection = _db_engine.connect()
    transaction = connection.begin()
    SessionLocal = sessionmaker(bind=connection)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def app():
    from app.main import app as _app

    return _app


@pytest_asyncio.fixture
async def client(app, db_session: Session):
    from unittest.mock import AsyncMock, MagicMock

    from app.core.database import get_db
    from app.modules.auth.auth_service import AuthService
    from app.modules.usage.usage_service import UsageService

    def override_get_db():
        yield db_session

    async def override_check_auth(request: Request, res: Response, credential=None):
        return {"user_id": "test-user", "email": "test@example.com"}

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[AuthService.check_auth] = override_check_auth
    app.dependency_overrides[UsageService.check_usage_limit] = lambda: True

    if getattr(app.state, "async_redis_stream_manager", None) is None:
        mock_async_redis = MagicMock()
        mock_async_redis.wait_for_task_start = AsyncMock(return_value=True)
        app.state.async_redis_stream_manager = mock_async_redis

    async with AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        follow_redirects=True,
    ) as http_client:
        yield http_client
    app.dependency_overrides.clear()

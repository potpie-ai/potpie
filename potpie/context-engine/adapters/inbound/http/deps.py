"""FastAPI dependencies for standalone HTTP mode."""

from __future__ import annotations

import os
from collections.abc import Generator
from functools import lru_cache

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from adapters.outbound.postgres.session import database_url, make_session_factory
from bootstrap.container import ContextEngineContainer
from bootstrap.standalone_container import build_standalone_context_engine_container

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(_api_key_header)) -> None:
    expected = os.getenv("CONTEXT_ENGINE_API_KEY", "").strip()
    if not expected:
        return None
    if not key or key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return None


@lru_cache
def get_container() -> ContextEngineContainer:
    return build_standalone_context_engine_container()


def get_container_or_503() -> ContextEngineContainer:
    try:
        return get_container()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@lru_cache
def _session_factory():
    return make_session_factory()


def get_db() -> Generator[Session, None, None]:
    factory = _session_factory()
    if factory is None:
        raise HTTPException(
            status_code=503,
            detail="DATABASE_URL (or POSTGRES_URL) is required for context HTTP API",
        )
    session = factory()
    try:
        yield session
    finally:
        session.close()


def get_db_optional() -> Generator[Session | None, None, None]:
    """Yield a SQLAlchemy session when ``DATABASE_URL`` is set, else ``None``."""
    factory = _session_factory()
    if factory is None:
        yield None
        return
    session = factory()
    try:
        yield session
    finally:
        session.close()


def require_database_configured() -> None:
    if not database_url():
        raise HTTPException(
            status_code=503,
            detail="DATABASE_URL (or POSTGRES_URL) is required",
        )

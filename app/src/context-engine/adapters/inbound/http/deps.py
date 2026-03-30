"""FastAPI dependencies for standalone HTTP mode."""

from __future__ import annotations

import os
from collections.abc import Generator
from functools import lru_cache

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from adapters.outbound.postgres.session import database_url, make_session_factory
from bootstrap.container import ContextEngineContainer, build_container_with_github_token
from bootstrap.http_projects import ExplicitProjectResolution, project_map_from_env

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
    token = (os.getenv("CONTEXT_ENGINE_GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN") or "").strip()
    mapping = project_map_from_env()
    if not token:
        raise RuntimeError("CONTEXT_ENGINE_GITHUB_TOKEN or GITHUB_TOKEN is required for HTTP server")
    if not mapping:
        raise RuntimeError("CONTEXT_ENGINE_PROJECTS env JSON is required, e.g. {\"proj-id\":\"owner/repo\"}")
    projects = ExplicitProjectResolution(mapping)
    return build_container_with_github_token(token=token, projects=projects)


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


def require_database_configured() -> None:
    if not database_url():
        raise HTTPException(
            status_code=503,
            detail="DATABASE_URL (or POSTGRES_URL) is required",
        )

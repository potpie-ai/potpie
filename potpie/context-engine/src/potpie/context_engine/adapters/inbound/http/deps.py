"""FastAPI dependencies for standalone HTTP mode."""

from __future__ import annotations

import hmac
import logging
import os
from collections.abc import Generator
from functools import lru_cache

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from potpie.context_engine.adapters.outbound.postgres.session import database_url, make_session_factory
from potpie.context_engine.bootstrap.ingestion_server import IngestionServerContainer
from potpie.context_engine.bootstrap.standalone_container import build_standalone_context_engine_container

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Single dev-only escape hatch shared with the policy layer: when set, the
# standalone HTTP surface runs with no API key AND no per-actor pot scoping.
# Never set this in a network-reachable / multi-tenant deployment.
ALLOW_NO_AUTH_ENV = "CONTEXT_ENGINE_ALLOW_NO_AUTH"

_no_auth_warned = False


def allow_no_auth() -> bool:
    """True only when the operator explicitly opted into the no-auth dev mode."""
    return os.getenv(ALLOW_NO_AUTH_ENV, "").strip().lower() in {"1", "true", "yes"}


def require_api_key(key: str | None = Security(_api_key_header)) -> None:
    """Authenticate the standalone HTTP surface — fail **closed** by default.

    The host-mounted path supplies its own ``require_auth``; this dependency
    only guards the standalone listener. When ``CONTEXT_ENGINE_API_KEY`` is
    unset the listener refuses every request (503) unless the operator set
    the loud, dev-only ``CONTEXT_ENGINE_ALLOW_NO_AUTH`` escape hatch.
    """
    expected = os.getenv("CONTEXT_ENGINE_API_KEY", "").strip()
    if not expected:
        if allow_no_auth():
            global _no_auth_warned
            if not _no_auth_warned:
                logger.warning(
                    "SECURITY: %s is set — the context-engine HTTP API is "
                    "running with NO authentication and NO per-actor pot "
                    "scoping. This must never be used in a network-reachable "
                    "or multi-tenant deployment.",
                    ALLOW_NO_AUTH_ENV,
                )
                _no_auth_warned = True
            return None
        raise HTTPException(
            status_code=503,
            detail=(
                "CONTEXT_ENGINE_API_KEY is not configured. Refusing to serve "
                "unauthenticated requests. Set an API key, or set "
                f"{ALLOW_NO_AUTH_ENV}=1 for local dev only."
            ),
        )
    if not key or not hmac.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return None


@lru_cache
def get_container() -> IngestionServerContainer:
    return build_standalone_context_engine_container()


def get_container_or_503() -> IngestionServerContainer:
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

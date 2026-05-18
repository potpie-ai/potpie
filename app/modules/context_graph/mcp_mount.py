"""Wire embedded HTTP MCP into the Potpie app (session factory + ASGI app)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from adapters.inbound.mcp.http_server import default_mcp_client_name
from adapters.inbound.mcp.session import McpSession, register_mcp_session_factory
from app.core.database import AsyncSessionLocal, SessionLocal
from app.modules.context_graph.context_engine_http import resolve_user_from_api_key
from app.modules.context_graph.wiring import build_container_for_user_session
from domain.actor import Actor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _mcp_session_from_api_key(api_key: str) -> AsyncIterator[McpSession]:
    db = SessionLocal()
    try:
        async with AsyncSessionLocal() as async_db:
            user = await resolve_user_from_api_key(
                api_key, db, async_db=async_db
            )
            if user is None:
                raise ValueError("Invalid API key")
            container = build_container_for_user_session(db, user["user_id"])
            client_name = None
            try:
                from adapters.inbound.mcp.auth_context import get_mcp_client_name

                client_name = get_mcp_client_name()
            except Exception:
                client_name = None
            actor = Actor(
                user_id=str(user["user_id"]),
                surface="mcp",
                client_name=client_name or default_mcp_client_name(),
                auth_method="api_key",
            )
            yield container, db, user, actor
    finally:
        db.close()


def init_embedded_mcp() -> None:
    """Register the MCP session factory (idempotent)."""
    register_mcp_session_factory(_mcp_session_from_api_key)
    logger.info("Potpie embedded HTTP MCP session factory registered")


def get_embedded_mcp_asgi_app():
    """Return the Streamable HTTP MCP ASGI app (call init_embedded_mcp first)."""
    from adapters.inbound.mcp.http_server import build_embedded_mcp_asgi_app

    init_embedded_mcp()
    return build_embedded_mcp_asgi_app()

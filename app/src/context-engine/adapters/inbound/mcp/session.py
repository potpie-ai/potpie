"""Host-injected session factory for embedded HTTP MCP."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from bootstrap.container import ContextEngineContainer
from domain.actor import Actor
from sqlalchemy.orm import Session

McpSession = tuple[ContextEngineContainer, Session, dict[str, Any], Actor]

SessionFactory = Callable[[str], AsyncIterator[McpSession]]

_factory: SessionFactory | None = None


def register_mcp_session_factory(factory: SessionFactory) -> None:
    global _factory
    _factory = factory


def mcp_session_factory_registered() -> bool:
    return _factory is not None


@asynccontextmanager
async def mcp_request_session(api_key: str) -> AsyncIterator[McpSession]:
    if _factory is None:
        raise RuntimeError(
            "Potpie MCP session factory is not registered. "
            "Mount MCP from app.main after importing app.modules.context_graph.mcp_mount."
        )
    async with _factory(api_key) as session:
        yield session

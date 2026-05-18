"""Embedded Streamable HTTP MCP on the Potpie FastAPI app (X-API-Key auth)."""

from __future__ import annotations

import asyncio
import functools
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from adapters.inbound.mcp.auth_context import get_mcp_api_key
from adapters.inbound.mcp.session import mcp_request_session
from adapters.inbound.mcp.tool_impl import (
    run_context_ingest,
    run_context_record,
    run_context_resolve,
    run_context_search,
    run_context_status,
)

logger = logging.getLogger(__name__)

# streamable_http_path="/" so FastAPI mount at /mcp yields https://host/mcp (not /mcp/mcp).
mcp_http = FastMCP(
    "potpie",
    streamable_http_path="/",
    instructions=(
        "Potpie context graph MCP. All tools require pot_id for a pot you can access "
        "with your API key. Prefer context_resolve for task context; use context_status "
        "before broad work; context_search for narrow follow-ups; context_record for "
        "durable learnings; context_ingest for raw episodic episodes."
    ),
)


def _auth_error(exc: Exception) -> dict:
    return {"ok": False, "error": "auth_error", "detail": str(exc)}


@mcp_http.tool()
async def context_search(
    pot_id: str,
    query: str,
    limit: int = 8,
    node_labels: str | None = None,
    repo_name: str | None = None,
    source_description: str | None = None,
    include_invalidated: bool = False,
    as_of: str | None = None,
) -> dict:
    """Narrow follow-up memory search. Prefer context_resolve first for task context wraps."""
    try:
        api_key = get_mcp_api_key()
    except ValueError as exc:
        return _auth_error(exc)
    async with mcp_request_session(api_key) as (container, _db, _user, actor):
        return await run_context_search(
            container=container,
            actor=actor,
            pot_id=pot_id,
            query=query,
            limit=limit,
            node_labels=node_labels,
            repo_name=repo_name,
            source_description=source_description,
            include_invalidated=include_invalidated,
            as_of=as_of,
        )


@mcp_http.tool()
async def context_resolve(
    pot_id: str,
    query: str,
    consumer_hint: str | None = None,
    intent: str | None = None,
    repo_name: str | None = None,
    branch: str | None = None,
    file_path: str | None = None,
    function_name: str | None = None,
    symbol: str | None = None,
    pr_number: int | None = None,
    services: str | None = None,
    features: str | None = None,
    environment: str | None = None,
    ticket_ids: str | None = None,
    user: str | None = None,
    source_refs: str | None = None,
    include: str | None = None,
    exclude: str | None = None,
    mode: str = "fast",
    source_policy: str = "references_only",
    max_items: int = 12,
    max_tokens: int | None = None,
    timeout_ms: int = 4000,
    freshness: str = "prefer_fresh",
    as_of: str | None = None,
) -> dict:
    """Primary agent context tool: resolve a bounded task context wrap with evidence, freshness, and fallbacks."""
    try:
        api_key = get_mcp_api_key()
    except ValueError as exc:
        return _auth_error(exc)
    async with mcp_request_session(api_key) as (container, _db, _user, actor):
        return await run_context_resolve(
            container=container,
            actor=actor,
            pot_id=pot_id,
            query=query,
            consumer_hint=consumer_hint,
            intent=intent,
            repo_name=repo_name,
            branch=branch,
            file_path=file_path,
            function_name=function_name,
            symbol=symbol,
            pr_number=pr_number,
            services=services,
            features=features,
            environment=environment,
            ticket_ids=ticket_ids,
            user=user,
            source_refs=source_refs,
            include=include,
            exclude=exclude,
            mode=mode,
            source_policy=source_policy,
            max_items=max_items,
            max_tokens=max_tokens,
            timeout_ms=timeout_ms,
            freshness=freshness,
            as_of=as_of,
        )


@mcp_http.tool()
async def context_ingest(
    pot_id: str,
    name: str,
    episode_body: str,
    source_description: str,
    reference_time: str | None = None,
    idempotency_key: str | None = None,
    sync: bool = False,
) -> dict:
    """Ingest a raw episodic episode into the context graph for a pot.
    Async by default (queued); pass sync=True for inline apply.
    Idempotency key deduplicates re-submissions."""
    try:
        api_key = get_mcp_api_key()
    except ValueError as exc:
        return _auth_error(exc)
    async with mcp_request_session(api_key) as (container, db, _user, actor):
        return await asyncio.to_thread(
            functools.partial(
                run_context_ingest,
                container=container,
                db=db,
                actor=actor,
                pot_id=pot_id,
                name=name,
                episode_body=episode_body,
                source_description=source_description,
                reference_time=reference_time,
                idempotency_key=idempotency_key,
                sync=sync,
            )
        )


@mcp_http.tool()
async def context_record(
    pot_id: str,
    record_type: str,
    summary: str,
    repo_name: str | None = None,
    source_refs: str | None = None,
    confidence: float = 0.7,
    visibility: str = "project",
    idempotency_key: str | None = None,
    details: str | None = None,
    sync: bool = False,
) -> dict:
    """Record durable project memory: decisions, fixes, preferences, workflows, feature notes, or incidents."""
    try:
        api_key = get_mcp_api_key()
    except ValueError as exc:
        return _auth_error(exc)
    async with mcp_request_session(api_key) as (container, db, _user, actor):
        return await asyncio.to_thread(
            functools.partial(
                run_context_record,
                container=container,
                db=db,
                actor=actor,
                pot_id=pot_id,
                record_type=record_type,
                summary=summary,
                repo_name=repo_name,
                source_refs=source_refs,
                confidence=confidence,
                visibility=visibility,
                idempotency_key=idempotency_key,
                details=details,
                sync=sync,
            )
        )


@mcp_http.tool()
async def context_status(
    pot_id: str,
    repo_name: str | None = None,
    source_refs: str | None = None,
    intent: str | None = None,
) -> dict:
    """Return cheap pot readiness plus the recommended context_resolve recipe for an intent."""
    try:
        api_key = get_mcp_api_key()
    except ValueError as exc:
        return _auth_error(exc)
    async with mcp_request_session(api_key) as (container, db, _user, actor):
        return await asyncio.to_thread(
            functools.partial(
                run_context_status,
                container=container,
                db=db,
                actor=actor,
                pot_id=pot_id,
                repo_name=repo_name,
                source_refs=source_refs,
                intent=intent,
            )
        )


_embedded_mcp_asgi_app = None


@asynccontextmanager
async def mcp_http_lifespan() -> AsyncIterator[None]:
    """Run Streamable HTTP MCP session manager (required when mounted on FastAPI)."""
    async with mcp_http._session_manager.run():
        yield


def build_embedded_mcp_asgi_app():
    """ASGI app for ``app.mount('/mcp', ...)`` with API-key middleware."""
    global _embedded_mcp_asgi_app
    if _embedded_mcp_asgi_app is not None:
        return _embedded_mcp_asgi_app
    from adapters.inbound.mcp.auth_context import PotpieMcpApiKeyMiddleware

    inner = mcp_http.streamable_http_app()
    _embedded_mcp_asgi_app = PotpieMcpApiKeyMiddleware(inner)
    return _embedded_mcp_asgi_app


def default_mcp_client_name() -> str:
    for var in (
        "POTPIE_CLIENT_NAME",
        "MCP_CLIENT_NAME",
        "CONTEXT_ENGINE_CLIENT_NAME",
    ):
        v = os.getenv(var)
        if v and v.strip():
            return v.strip()
    return "mcp-unknown"

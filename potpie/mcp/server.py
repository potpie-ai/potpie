"""Protocol-native MCP binding for Potpie's four public context tools.

The MCP process is a product surface. It obtains the same ``PotpieRuntime`` as
the CLI and daemon UI, then routes engine work through ``runtime.engine``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from mcp.server.fastmcp import FastMCP

from potpie.mcp.project_access import assert_mcp_pot_allowed
from potpie.runtime.composition import get_runtime
from potpie.runtime.contracts import (
    CapabilityNotImplemented,
    RecordRequest,
    ResolveRequest,
    SearchRequest,
)
from potpie.runtime.errors import RuntimeBoundaryError
from potpie.runtime.sync_view import await_engine

mcp = FastMCP("potpie")


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_as_of_iso(value: str | None) -> datetime | None:
    if not value or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    return datetime.fromisoformat(normalized)


def _scope(**fields: Any) -> dict[str, Any]:
    return {key: value for key, value in fields.items() if value not in (None, [], ())}


def _error(exc: Exception) -> dict[str, Any]:
    return {"ok": False, "error": type(exc).__name__, "detail": str(exc)}


def _envelope_dict(envelope: Any) -> dict[str, Any]:
    payload = envelope.to_dict()
    payload["ok"] = True
    return payload


@mcp.tool()
def context_resolve(
    pot_id: str,
    query: str,
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
    as_of: str | None = None,
) -> dict:
    """Resolve a bounded task context wrap with evidence and coverage."""
    try:
        assert_mcp_pot_allowed(pot_id)
        envelope = await_engine(
            get_runtime().engine.context.resolve(
                ResolveRequest(
                    pot_id=pot_id,
                    task=query,
                    intent=intent,
                    include=_split_csv(include),
                    exclude=_split_csv(exclude),
                    scope=_scope(
                        repo_name=repo_name,
                        branch=branch,
                        file_path=file_path,
                        function_name=function_name,
                        symbol=symbol,
                        pr_number=pr_number,
                        services=list(_split_csv(services)),
                        features=list(_split_csv(features)),
                        environment=environment,
                        ticket_ids=list(_split_csv(ticket_ids)),
                        user=user,
                        source_refs=list(_split_csv(source_refs)),
                    ),
                    mode=mode,
                    source_policy=source_policy,
                    max_items=max_items,
                    as_of=_parse_as_of_iso(as_of),
                )
            )
        )
        return _envelope_dict(envelope)
    except (ValueError, CapabilityNotImplemented, RuntimeBoundaryError) as exc:
        return _error(exc)


@mcp.tool()
def context_search(
    pot_id: str,
    query: str,
    include: str | None = None,
    repo_name: str | None = None,
    max_items: int = 8,
) -> dict:
    """Search narrowly for a known phrase or entity."""
    try:
        assert_mcp_pot_allowed(pot_id)
        envelope = await_engine(
            get_runtime().engine.context.search(
                SearchRequest(
                    pot_id=pot_id,
                    query=query,
                    include=_split_csv(include),
                    scope=_scope(repo_name=repo_name),
                    max_items=max_items,
                )
            )
        )
        return _envelope_dict(envelope)
    except (ValueError, CapabilityNotImplemented, RuntimeBoundaryError) as exc:
        return _error(exc)


@mcp.tool()
def context_record(
    pot_id: str,
    record_type: str,
    summary: str,
    repo_name: str | None = None,
    source_refs: str | None = None,
    details: str | None = None,
    confidence: float = 0.7,
    visibility: str = "project",
    idempotency_key: str | None = None,
) -> dict:
    """Record durable project memory."""
    try:
        assert_mcp_pot_allowed(pot_id)
        detail_payload: dict[str, Any] = {
            "confidence": confidence,
            "visibility": visibility,
        }
        if details:
            detail_payload["text"] = details
        receipt = await_engine(
            get_runtime().engine.context.record(
                RecordRequest(
                    pot_id=pot_id,
                    record_type=record_type,
                    summary=summary,
                    details=detail_payload,
                    scope=_scope(repo_name=repo_name),
                    source_refs=_split_csv(source_refs),
                    idempotency_key=idempotency_key,
                )
            )
        )
        return {
            "ok": receipt.accepted,
            "status": receipt.status,
            "record_id": receipt.record_id,
            "mutations_applied": receipt.mutations_applied,
            "detail": receipt.detail,
        }
    except (ValueError, CapabilityNotImplemented, RuntimeBoundaryError) as exc:
        return _error(exc)


@mcp.tool()
def context_status(
    pot_id: str,
    intent: str | None = None,
    harness: str | None = None,
) -> dict:
    """Return the flat, root-composed Potpie product status."""
    del intent  # Retained only to preserve the established MCP argument schema.
    try:
        assert_mcp_pot_allowed(pot_id)
        runtime = get_runtime()
        return runtime.status.get(
            pot_id=pot_id,
            harness=harness or "claude",
        ).to_dict()
    except (ValueError, RuntimeBoundaryError) as exc:
        return _error(exc)


__all__ = [
    "context_record",
    "context_resolve",
    "context_search",
    "context_status",
    "mcp",
]

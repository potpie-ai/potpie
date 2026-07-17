"""MCP server: the four-tool context surface, served in-process via HostShell.

CLI, HTTP, and MCP all bind to the same ``AgentContextPort``; this module is the
MCP binding. It runs the engine **in-process** (``build_host_shell``) rather than
as an HTTP client to a managed API — a local agent (claude-code, cursor) talks
straight to the local context graph through the same services the CLI uses.

    context_resolve   -> host.agent_context.resolve
    context_search    -> host.agent_context.search
    context_record    -> host.agent_context.record
    context_status    -> host.agent_context.status
"""

from __future__ import annotations

import logging
from datetime import datetime
from functools import lru_cache
from typing import Any

from mcp.server.fastmcp import FastMCP

from potpie.mcp.project_access import assert_mcp_pot_allowed
from potpie.services.host_wiring import build_host_shell
from potpie_context_core.errors import CapabilityNotImplemented
from potpie_context_core.ports.agent_context import (
    RecordRequest,
    ResolveRequest,
    SearchRequest,
    StatusRequest,
)
from potpie.daemon.shell import HostShell

logger = logging.getLogger(__name__)
mcp = FastMCP("potpie")


@lru_cache(maxsize=1)
def _host() -> HostShell:
    """One in-process ``HostShell`` per server process (the embedded backend is
    persistent, so this survives across tool calls within a session)."""
    return build_host_shell()


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_as_of_iso(value: str | None) -> datetime | None:
    if not value or not value.strip():
        return None
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _scope(**fields: Any) -> dict[str, Any]:
    """Drop unset (None / empty list) scope axes so the reader sees only what
    the caller actually constrained."""
    return {key: value for key, value in fields.items() if value not in (None, [], ())}


def _error(exc: Exception) -> dict[str, Any]:
    return {"ok": False, "error": type(exc).__name__, "detail": str(exc)}


def _envelope_dict(env: Any) -> dict[str, Any]:
    payload = env.to_dict()
    payload["ok"] = True
    return payload


def _nudge_dict(nudge: Any) -> dict[str, Any] | None:
    if nudge is None:
        return None
    return {
        "agent": nudge.agent,
        "missing": list(nudge.missing),
        "outdated": list(nudge.outdated),
        "install_command": nudge.install_command,
    }


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
    """Primary agent context tool: resolve a bounded task context wrap with evidence and coverage."""
    try:
        assert_mcp_pot_allowed(pot_id)
        env = _host().agent_context.resolve(
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
        return _envelope_dict(env)
    except (ValueError, CapabilityNotImplemented) as exc:
        return _error(exc)


@mcp.tool()
def context_search(
    pot_id: str,
    query: str,
    include: str | None = None,
    repo_name: str | None = None,
    max_items: int = 8,
) -> dict:
    """Narrow follow-up memory search on a known phrase or entity. Prefer context_resolve for task wraps."""
    try:
        assert_mcp_pot_allowed(pot_id)
        env = _host().agent_context.search(
            SearchRequest(
                pot_id=pot_id,
                query=query,
                include=_split_csv(include),
                scope=_scope(repo_name=repo_name),
                max_items=max_items,
            )
        )
        return _envelope_dict(env)
    except (ValueError, CapabilityNotImplemented) as exc:
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
    """Record durable project memory: decisions, fixes, preferences, workflows, feature notes, or incidents."""
    try:
        assert_mcp_pot_allowed(pot_id)
        detail_payload: dict[str, Any] = {
            "confidence": confidence,
            "visibility": visibility,
        }
        if details:
            detail_payload["text"] = details
        receipt = _host().agent_context.record(
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
        return {
            "ok": receipt.accepted,
            "status": receipt.status,
            "record_id": receipt.record_id,
            "mutations_applied": receipt.mutations_applied,
            "detail": receipt.detail,
        }
    except (ValueError, CapabilityNotImplemented) as exc:
        return _error(exc)


@mcp.tool()
def context_status(
    pot_id: str,
    intent: str | None = None,
    harness: str | None = None,
) -> dict:
    """Return cheap pot readiness plus the recommended next action for an intent."""
    try:
        assert_mcp_pot_allowed(pot_id)
        report = _host().agent_context.status(
            StatusRequest(pot_id=pot_id, intent=intent, harness=harness)
        )
        return {
            "ok": True,
            "pot_id": report.pot_id,
            "profile": report.profile,
            "daemon_up": report.daemon_up,
            "active_pot": report.active_pot,
            "backend_ready": report.backend_ready,
            "data_plane": dict(report.data_plane),
            "pot_summary": dict(report.pot_summary),
            "skills": _nudge_dict(report.skills),
            "recommended_next_action": report.recommended_next_action,
        }
    except (ValueError, CapabilityNotImplemented) as exc:
        return _error(exc)


def main() -> None:
    from potpie_context_engine.bootstrap.logging_setup import configure_logging

    configure_logging()
    mcp.run()


if __name__ == "__main__":
    main()

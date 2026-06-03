"""MCP server: context graph tools via Potpie POST /api/v2/context (X-API-Key)."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

from adapters.inbound.cli.potpie_api_config import (
    resolve_potpie_api_base_url,
    resolve_potpie_api_key,
)
from adapters.inbound.mcp.project_access import assert_mcp_pot_allowed
from adapters.outbound.http.potpie_context_api_client import (
    IngestRejectedError,
    PotpieContextApiClient,
    PotpieContextApiError,
)

logger = logging.getLogger(__name__)
mcp = FastMCP("potpie")


def _mcp_client_name() -> str:
    """Identify the enclosing MCP client (e.g. ``claude-code``, ``cursor``)."""
    for var in (
        "POTPIE_CLIENT_NAME",
        "MCP_CLIENT_NAME",
        "CONTEXT_ENGINE_CLIENT_NAME",
    ):
        v = os.getenv(var)
        if v and v.strip():
            return v.strip()
    return "mcp-unknown"


def _client() -> PotpieContextApiClient:
    return PotpieContextApiClient(
        resolve_potpie_api_base_url(),
        resolve_potpie_api_key(),
        client_surface="mcp",
        client_name=_mcp_client_name(),
    )


def _api_err(e: PotpieContextApiError) -> dict:
    return {
        "ok": False,
        "error": "api_error",
        "status_code": e.status_code,
        "detail": e.detail,
    }


def _parse_as_of_iso(value: str | None) -> datetime | None:
    if not value or not value.strip():
        return None
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


@mcp.tool()
def context_search(
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
    assert_mcp_pot_allowed(pot_id)
    labels = None
    if node_labels:
        labels = [x.strip() for x in node_labels.split(",") if x.strip()]
    as_of_dt = None
    if as_of:
        try:
            as_of_dt = _parse_as_of_iso(as_of)
        except ValueError as exc:
            return {"ok": False, "error": "invalid_as_of", "detail": str(exc)}
    body = {
        "pot_id": pot_id,
        "query": query,
        "goal": "retrieve",
        "strategy": "semantic",
        "limit": limit,
        "node_labels": labels,
        "scope": {"repo_name": repo_name} if repo_name else {},
        "source_descriptions": [source_description] if source_description else [],
        "include_invalidated": include_invalidated,
        "as_of": as_of_dt,
    }
    try:
        out = _client().context_graph_query(body)
        rows = out.get("result") if isinstance(out, dict) else []
        rows = rows if isinstance(rows, list) else []
        return {
            "ok": True,
            "answer": {"summary": f"Found {len(rows)} context search result(s)."},
            "evidence": rows,
            "source_refs": [],
            "coverage": {
                "status": "complete" if rows else "empty",
                "available": ["semantic_search"] if rows else [],
                "missing": [] if rows else ["semantic_search"],
                "missing_reasons": {} if rows else {"semantic_search": "empty_result"},
            },
            "freshness": {
                "status": "unknown",
                "last_graph_update": None,
                "last_source_verification": None,
                "stale_refs": [],
                "needs_verification_refs": [],
            },
            "fallbacks": [],
            "recommended_next_actions": [],
        }
    except PotpieContextApiError as e:
        return _api_err(e)


def context_ingest_episode(
    pot_id: str,
    name: str,
    episode_body: str,
    source_description: str,
    sync: bool = False,
    reference_time: str | None = None,
    idempotency_key: str | None = None,
) -> dict:
    """Ingest a raw episode (Potpie POST /api/v2/context/ingest)."""
    assert_mcp_pot_allowed(pot_id)
    if reference_time and reference_time.strip():
        try:
            s = reference_time.strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            ref = datetime.fromisoformat(s)
        except ValueError as exc:
            return {"ok": False, "error": "invalid_reference_time", "detail": str(exc)}
    else:
        ref = datetime.now(timezone.utc)

    body: dict = {
        "pot_id": pot_id,
        "name": name,
        "episode_body": episode_body,
        "source_description": source_description,
        "reference_time": ref,
    }
    if idempotency_key and idempotency_key.strip():
        body["idempotency_key"] = idempotency_key.strip()
    try:
        status_code, data = _client().ingest(body, sync=sync)
    except IngestRejectedError as exc:
        out = {"ok": False, **exc.body}
        if out.get("status") is None:
            out["status"] = "reconciliation_rejected"
        return out
    except PotpieContextApiError as exc:
        return {
            "ok": False,
            "error": "api_error",
            "status_code": exc.status_code,
            "detail": exc.detail,
        }

    if status_code == 202:
        return {
            "ok": True,
            "status": "queued",
            "episode_uuid": data.get("episode_uuid"),
            "event_id": data.get("event_id"),
            "job_id": data.get("job_id"),
        }
    return {
        "ok": True,
        "status": "applied",
        "episode_uuid": data.get("episode_uuid"),
        "event_id": data.get("event_id"),
        "job_id": data.get("job_id"),
    }


@mcp.tool()
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
    assert_mcp_pot_allowed(pot_id)
    as_of_dt = None
    if as_of:
        try:
            as_of_dt = _parse_as_of_iso(as_of)
        except ValueError as exc:
            return {"ok": False, "error": "invalid_as_of", "detail": str(exc)}
    scope = {
        "repo_name": repo_name,
        "branch": branch,
        "file_path": file_path,
        "function_name": function_name,
        "symbol": symbol,
        "pr_number": pr_number,
        "services": _split_csv(services),
        "features": _split_csv(features),
        "environment": environment,
        "ticket_ids": _split_csv(ticket_ids),
        "user": user,
        "source_refs": _split_csv(source_refs),
    }
    body = {
        "pot_id": pot_id,
        "query": query,
        "consumer_hint": consumer_hint,
        "intent": intent,
        "scope": {
            key: value for key, value in scope.items() if value not in (None, [])
        },
        "include": _split_csv(include),
        "exclude": _split_csv(exclude),
        "goal": "answer",
        "strategy": "hybrid" if mode == "balanced" else "auto",
        "source_policy": source_policy,
        "budget": {
            "max_items": max_items,
            "max_tokens": max_tokens,
            "timeout_ms": timeout_ms,
            "freshness": freshness,
        },
        "as_of": as_of_dt,
    }
    client = _client()
    try:
        out = await client.context_graph_query_async(body)
        result = out.get("result") if isinstance(out, dict) else None
        return result if isinstance(result, dict) else out
    except PotpieContextApiError as e:
        return _api_err(e)


@mcp.tool()
def context_record(
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
    assert_mcp_pot_allowed(pot_id)
    body = {
        "pot_id": pot_id,
        "record": {
            "type": record_type,
            "summary": summary,
            "details": {"text": details} if details else {},
            "source_refs": _split_csv(source_refs),
            "confidence": confidence,
            "visibility": visibility,
        },
        "scope": {"repo_name": repo_name} if repo_name else {},
        "idempotency_key": idempotency_key,
    }
    try:
        return _client().record(body, sync=sync)
    except PotpieContextApiError as e:
        return _api_err(e)


@mcp.tool()
def context_status(
    pot_id: str,
    repo_name: str | None = None,
    source_refs: str | None = None,
    intent: str | None = None,
) -> dict:
    """Return cheap pot readiness plus the recommended context_resolve recipe for an intent."""
    assert_mcp_pot_allowed(pot_id)
    scope = {
        "repo_name": repo_name,
        "source_refs": _split_csv(source_refs),
    }
    body = {
        "pot_id": pot_id,
        "intent": intent,
        "scope": {
            key: value for key, value in scope.items() if value not in (None, [])
        },
    }
    try:
        return _client().status(body)
    except PotpieContextApiError as e:
        return _api_err(e)


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()

"""In-process MCP tool implementations (shared by HTTP and future transports)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from application.use_cases.record_durable_context import (
    DurableContextPayload,
    record_durable_context,
)
from application.use_cases.report_status import report_status
from application.use_cases.submit_raw_episode import (
    RawEpisodeSubmissionResult,
    submit_raw_episode,
)
from bootstrap.container import ContextEngineContainer
from domain.actor import Actor
from domain.graph_query import (
    ContextGraphBudget,
    ContextGraphGoal,
    ContextGraphQuery,
    ContextGraphScope,
    ContextGraphStrategy,
)
from domain.ports.policy import (
    ACTION_POT_INGEST_EPISODE,
    ACTION_POT_READ,
    ACTION_POT_RECORD,
    RESOURCE_POT,
)
from sqlalchemy.orm import Session

from adapters.inbound.mcp.policy import UNKNOWN_POT_DETAIL, policy_error_response


def _parse_as_of_iso(value: str | None) -> datetime | None:
    if not value or not value.strip():
        return None
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _scope_from_mcp(
    *,
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
) -> dict[str, Any]:
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
    return {k: v for k, v in scope.items() if v not in (None, [], "")}


def _context_graph_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return str(isoformat())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _context_graph_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_context_graph_jsonable(v) for v in value]
    return str(value)


async def run_context_search(
    *,
    container: ContextEngineContainer,
    actor: Actor,
    pot_id: str,
    query: str,
    limit: int = 8,
    node_labels: str | None = None,
    repo_name: str | None = None,
    source_description: str | None = None,
    include_invalidated: bool = False,
    as_of: str | None = None,
) -> dict[str, Any]:
    denied = policy_error_response(
        container,
        actor=actor,
        resource=RESOURCE_POT,
        action=ACTION_POT_READ,
        pot_id=pot_id,
    )
    if denied:
        return denied

    as_of_dt = None
    if as_of:
        try:
            as_of_dt = _parse_as_of_iso(as_of)
        except ValueError as exc:
            return {"ok": False, "error": "invalid_as_of", "detail": str(exc)}

    labels = None
    if node_labels:
        labels = [x.strip() for x in node_labels.split(",") if x.strip()]

    if container.context_graph is None:
        return {
            "ok": False,
            "error": "api_error",
            "status_code": 503,
            "detail": "Unified context graph query port is not configured.",
        }

    body = ContextGraphQuery(
        pot_id=pot_id,
        query=query,
        goal=ContextGraphGoal.RETRIEVE,
        strategy=ContextGraphStrategy.SEMANTIC,
        limit=limit,
        node_labels=labels or [],
        scope=ContextGraphScope(repo_name=repo_name) if repo_name else ContextGraphScope(),
        source_descriptions=[source_description] if source_description else [],
        include_invalidated=include_invalidated,
        as_of=as_of_dt,
    )
    try:
        result = await container.context_graph.query_async(body)
        dumped = result.model_dump()
        rows = dumped.get("result") if isinstance(dumped, dict) else []
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
    except Exception as exc:
        return {"ok": False, "error": "api_error", "detail": str(exc)}


async def run_context_resolve(
    *,
    container: ContextEngineContainer,
    actor: Actor,
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
) -> dict[str, Any]:
    denied = policy_error_response(
        container,
        actor=actor,
        resource=RESOURCE_POT,
        action=ACTION_POT_READ,
        pot_id=pot_id,
    )
    if denied:
        return denied

    as_of_dt = None
    if as_of:
        try:
            as_of_dt = _parse_as_of_iso(as_of)
        except ValueError as exc:
            return {"ok": False, "error": "invalid_as_of", "detail": str(exc)}

    if container.context_graph is None:
        return {
            "ok": False,
            "error": "api_error",
            "status_code": 503,
            "detail": "Unified context graph query port is not configured.",
        }

    scope_dict = _scope_from_mcp(
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
    )
    body = ContextGraphQuery(
        pot_id=pot_id,
        query=query,
        consumer_hint=consumer_hint,
        intent=intent,
        scope=ContextGraphScope(**scope_dict) if scope_dict else ContextGraphScope(),
        include=_split_csv(include),
        exclude=_split_csv(exclude),
        goal=ContextGraphGoal.ANSWER,
        strategy=ContextGraphStrategy.HYBRID if mode == "balanced" else ContextGraphStrategy.AUTO,
        source_policy=source_policy,
        as_of=as_of_dt,
        budget=ContextGraphBudget(
            max_items=max_items,
            max_tokens=max_tokens,
            timeout_ms=timeout_ms,
            freshness=freshness,
        ),
    )
    try:
        result = await container.context_graph.query_async(body)
        dumped = _context_graph_jsonable(result.model_dump())
        inner = dumped.get("result") if isinstance(dumped, dict) else None
        return inner if isinstance(inner, dict) else dumped
    except Exception as exc:
        return {"ok": False, "error": "api_error", "detail": str(exc)}


def run_context_record(
    *,
    container: ContextEngineContainer,
    db: Session,
    actor: Actor,
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
) -> dict[str, Any]:
    denied = policy_error_response(
        container,
        actor=actor,
        resource=RESOURCE_POT,
        action=ACTION_POT_RECORD,
        pot_id=pot_id,
    )
    if denied:
        return denied

    scope = {"repo_name": repo_name} if repo_name else {}
    try:
        receipt, record_type_out, source_id = record_durable_context(
            container.ingestion_submission(db),
            pot_id=pot_id,
            record=DurableContextPayload(
                record_type=record_type,
                summary=summary,
                details={"text": details} if details else {},
                source_refs=tuple(_split_csv(source_refs)),
                confidence=confidence,
                visibility=visibility,
            ),
            scope={**scope, "source_refs": _split_csv(source_refs)},
            actor=actor,
            idempotency_key=idempotency_key,
            occurred_at=None,
            sync=sync,
        )
        db.commit()
    except ValueError as exc:
        db.rollback()
        err = str(exc)
        if err == "unknown_pot_id":
            return {
                "ok": False,
                "error": "api_error",
                "status_code": 404,
                "detail": UNKNOWN_POT_DETAIL,
            }
        if err in {"context_graph_disabled", "async_requires_database"}:
            return {"ok": False, "error": "api_error", "status_code": 503, "detail": err}
        return {"ok": False, "error": "api_error", "status_code": 400, "detail": err}
    except Exception as exc:
        db.rollback()
        return {"ok": False, "error": "api_error", "detail": str(exc)}

    return {
        "ok": receipt.error is None,
        "status": "duplicate" if receipt.duplicate else receipt.status,
        "event_id": receipt.event_id,
        "job_id": receipt.job_id,
        "record_type": record_type_out,
        "source_id": source_id,
        "fallbacks": [
            {
                "code": "record_queued",
                "message": "The context record was accepted and queued for reconciliation.",
                "impact": "It may not appear in graph reads until the worker applies it.",
            }
        ]
        if receipt.status == "queued" and not receipt.duplicate
        else [],
        "error": receipt.error,
    }


def _map_raw_episode_submission_result(
    result: RawEpisodeSubmissionResult,
) -> dict[str, Any]:
    """Map submit_raw_episode outcome to the MCP ingest envelope."""
    downgrades = list(result.downgrades or [])
    if result.status == "duplicate":
        return {
            "ok": True,
            "status": "duplicate",
            "event_id": result.event_id,
            "job_id": result.job_id,
            "episode_uuid": result.episode_uuid,
            "detail": "Already ingested.",
            "duplicate_reason": result.duplicate_reason,
            "downgrades": downgrades,
            "fallbacks": [],
        }
    if result.status == "reconciliation_rejected":
        return {
            "ok": False,
            "error": "reconciliation_rejected",
            "status": result.status,
            "event_id": result.event_id,
            "job_id": result.job_id,
            "errors": list(result.reconciliation_errors or []),
            "downgrades": downgrades,
            "fallbacks": [],
        }
    if result.status == "queued" and result.ok:
        return {
            "ok": True,
            "status": "queued",
            "event_id": result.event_id,
            "job_id": result.job_id,
            "episode_uuid": result.episode_uuid,
            "downgrades": downgrades,
            "fallbacks": [
                {
                    "code": "ingest_queued",
                    "message": "The episode was accepted and queued for ingestion.",
                    "impact": "It may not appear in context_resolve or context_search until the worker applies it.",
                }
            ],
        }
    if result.status == "applied" and result.ok:
        return {
            "ok": True,
            "status": "applied",
            "event_id": result.event_id,
            "job_id": result.job_id,
            "episode_uuid": result.episode_uuid,
            "downgrades": downgrades,
            "fallbacks": [],
        }
    err = result.error or "ingest_failed"
    if err == "unknown_pot_id":
        return {
            "ok": False,
            "error": "unknown_pot_id",
            "status_code": 404,
            "detail": UNKNOWN_POT_DETAIL,
        }
    if err == "async_requires_database":
        return {
            "ok": False,
            "error": "async_requires_database",
            "status_code": 503,
            "detail": "Pass sync=True or configure Postgres for async ingest.",
        }
    if err == "context_graph_disabled":
        return {
            "ok": False,
            "error": "context_graph_disabled",
            "status_code": 503,
            "detail": (
                "Context graph is disabled (unset CONTEXT_GRAPH_ENABLED or set to true)."
            ),
        }
    if err == "context_graph_unavailable":
        return {
            "ok": False,
            "error": "context_graph_unavailable",
            "status_code": 503,
            "detail": "Unified context graph query port is not configured.",
        }
    if err == "graphiti_returned_no_uuid":
        return {
            "ok": False,
            "error": "graphiti_returned_no_uuid",
            "status_code": 502,
            "detail": "Graph write completed without an episode UUID.",
        }
    return {
        "ok": False,
        "error": "api_error",
        "status": result.status,
        "event_id": result.event_id,
        "job_id": result.job_id,
        "detail": err,
        "downgrades": downgrades,
        "fallbacks": [],
    }


def run_context_ingest(
    *,
    container: ContextEngineContainer,
    db: Session,
    actor: Actor,
    pot_id: str,
    name: str,
    episode_body: str,
    source_description: str,
    reference_time: str | None = None,
    idempotency_key: str | None = None,
    sync: bool = False,
) -> dict[str, Any]:
    denied = policy_error_response(
        container,
        actor=actor,
        resource=RESOURCE_POT,
        action=ACTION_POT_INGEST_EPISODE,
        pot_id=pot_id,
    )
    if denied:
        return denied

    if not name.strip():
        return {
            "ok": False,
            "error": "validation_error",
            "detail": "name must not be empty",
        }
    if not episode_body.strip():
        return {
            "ok": False,
            "error": "validation_error",
            "detail": "episode_body must not be empty",
        }

    ref_dt: datetime
    if reference_time is not None and reference_time.strip():
        try:
            parsed = _parse_as_of_iso(reference_time)
        except ValueError as exc:
            return {
                "ok": False,
                "error": "invalid_reference_time",
                "detail": str(exc),
            }
        ref_dt = parsed if parsed is not None else datetime.now(timezone.utc)
    else:
        ref_dt = datetime.now(timezone.utc)

    try:
        result = submit_raw_episode(
            container=container,
            db=db,
            pot_id=pot_id,
            name=name.strip(),
            episode_body=episode_body,
            source_description=source_description,
            reference_time=ref_dt,
            idempotency_key=idempotency_key,
            sync=sync,
            source_channel=actor.surface,
            actor=actor,
        )
        if result.ok:
            db.commit()
        else:
            db.rollback()
    except Exception as exc:
        db.rollback()
        return {"ok": False, "error": "api_error", "detail": str(exc)}

    return _map_raw_episode_submission_result(result)


def run_context_status(
    *,
    container: ContextEngineContainer,
    db: Session | None,
    actor: Actor,
    pot_id: str,
    repo_name: str | None = None,
    source_refs: str | None = None,
    intent: str | None = None,
) -> dict[str, Any]:
    denied = policy_error_response(
        container,
        actor=actor,
        resource=RESOURCE_POT,
        action=ACTION_POT_READ,
        pot_id=pot_id,
    )
    if denied:
        return denied

    scope = _scope_from_mcp(repo_name=repo_name, source_refs=source_refs)
    report = report_status(
        container,
        pot_id=pot_id,
        scope=scope,
        intent=intent,
        db=db,
    )
    if report.unknown_pot:
        return {
            "ok": False,
            "error": "api_error",
            "status_code": 404,
            "detail": UNKNOWN_POT_DETAIL,
        }
    return report.payload

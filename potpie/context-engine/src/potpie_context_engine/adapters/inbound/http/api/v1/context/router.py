"""Context graph HTTP API (standalone service + host-injected dependencies)."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from potpie_context_engine.adapters.inbound.http.api.v1.context.event_payload import (
    ingestion_event_to_payload,
)
from potpie_context_engine.adapters.inbound.http.deps import (
    get_container_or_503,
    get_db,
    get_db_optional,
    require_api_key,
)
from potpie_context_engine.adapters.outbound.postgres.ledger import SqlAlchemyIngestionLedger
from potpie_context_engine.adapters.outbound.postgres.reconciliation_ledger import (
    SqlAlchemyReconciliationLedger,
)
from potpie_context_engine.application.use_cases.hard_reset_pot import hard_reset_pot
from potpie_context_engine.application.use_cases.record_durable_context import (
    DurableContextPayload,
    record_durable_context,
)
from potpie_context_engine.application.use_cases.report_status import report_status
from potpie_context_engine.application.use_cases.submit_raw_episode import submit_raw_episode
from potpie_context_engine.bootstrap.ingestion_server import IngestionServerContainer
from potpie_context_engine.domain.actor import Actor, ActorSurface, normalize_surface
from potpie_context_engine.domain.ingestion_event_models import (
    EventListFilters,
    IngestionEventStatus,
    IngestionSubmissionRequest,
)
from potpie_context_engine.domain.ingestion_kinds import INGESTION_KIND_AGENT_RECONCILIATION
from potpie_context_engine.domain.ports.policy import (
    ACTION_POT_INGEST_EPISODE,
    ACTION_POT_READ,
    ACTION_POT_RECORD,
    ACTION_POT_RESET,
    ACTION_POT_SUBMIT_EVENT,
    RESOURCE_POT,
    REASON_UNKNOWN_POT,
    PolicyDecision,
)
from potpie_context_engine.domain.ports.agent_context import ResolveRequest

# Surfaces a client may self-declare via the untrusted ``X-Potpie-Client``
# header. ``system``/``webhook`` are deliberately excluded — those are
# server-stamped (internal jobs, signature-verified webhooks) and must never
# be assertable by a request.
_CLIENT_DECLARABLE_SURFACES: frozenset[str] = frozenset({"cli", "mcp", "http"})

UNKNOWN_POT_DETAIL = (
    "Unknown pot_id for this user (create with POST /api/v2/context/pots "
    "and attach at least one repository)."
)


def _ndjson_line(event: dict[str, Any]) -> bytes:
    """Serialize one stream event as a UTF-8 NDJSON line.

    Matches the chat streaming client contract: one JSON object per line so
    the consumer can parse line-by-line without needing SSE framing.
    """
    import json as _json

    return (_json.dumps(event, default=str) + "\n").encode("utf-8")


_logger = logging.getLogger(__name__)


# Operator/admin routes (``/reset``, ``/maintenance/*``) are
# grouped under a distinct OpenAPI tag so product docs and SDKs do not conflate
# them with everyday agent flows. Every destructive or graph-mutating call goes
# through ``_audit_operator_action`` so there is a single structured log line
# per action that operators and on-call can scrape.
OPERATOR_TAG = "context:operator"

_AUDIT_LOGGER_NAME = "context_engine.operator_audit"
_audit_logger = logging.getLogger(_AUDIT_LOGGER_NAME)


def _actor_identity(actor: Any) -> str:
    """Best-effort extraction of an actor label for audit logs."""

    if actor is None:
        return "anonymous"
    for attr in ("email", "username", "sub", "id"):
        v = getattr(actor, attr, None)
        if v:
            return str(v)
    if isinstance(actor, dict):
        for key in ("email", "username", "sub", "id"):
            if actor.get(key):
                return str(actor[key])
    return repr(actor)[:64]


def _audit_operator_action(
    *,
    action: str,
    pot_id: str,
    actor: Any = None,
    dry_run: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit a structured audit log for a destructive or admin operator action."""

    fields: dict[str, Any] = {
        "action": action,
        "pot_id": pot_id,
        "actor": _actor_identity(actor),
    }
    if dry_run is not None:
        fields["dry_run"] = bool(dry_run)
    if extra:
        for k, v in extra.items():
            if k in fields:
                continue
            fields[k] = v
    _audit_logger.warning("operator_action %s", action, extra={"audit": fields})


def _ingest_rejection_returns_422() -> bool:
    """When true, sync ingest surfaces structured reconciliation failures as HTTP 422.

    Set ``CONTEXT_ENGINE_INGEST_422=0`` to restore legacy 503 + plain detail string
    for operators that treat any 2xx as success monitors.
    """
    v = os.getenv("CONTEXT_ENGINE_INGEST_422", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


class BatchRetryEventsRequest(BaseModel):
    """Bulk-retry request body. Capped at 200 to avoid a runaway batch.

    Caller passes only event_ids; the route enforces that all belong to the
    URL's ``pot_id``. Larger bulk operations should rely on windowed
    batching instead of a single jumbo retry.
    """

    model_config = ConfigDict(extra="forbid")
    event_ids: list[str] = Field(..., min_length=1, max_length=200)


class IngestionConfigBody(BaseModel):
    """PUT /pots/{pot_id}/ingestion-config body.

    ``mode`` is constrained at the schema level so pydantic returns 422
    before we hit the adapter — this is the failure mode our integration
    tests pin down (otherwise an unknown mode would silently no-op when
    the adapter is mocked).
    """

    model_config = ConfigDict(extra="forbid")
    mode: Literal["immediate", "windowed"] = Field(
        ..., description="'immediate' or 'windowed'"
    )
    window_minutes: int = Field(5, ge=1, le=1440)
    min_batch_size: int | None = Field(None, ge=1)


class HardResetRequest(BaseModel):
    """Hard-delete all context-graph data for a pot."""

    model_config = ConfigDict(populate_by_name=True)

    pot_id: str = Field(description="Pot scope id (Neo4j partition / group_id).")
    skip_ledger: bool = Field(
        default=False,
        description="If true, only clear the canonical Neo4j graph; do not delete Postgres ledger rows.",
    )


class IngestEpisodeRequest(BaseModel):
    """Raw episode admitted through the async reconciliation pipeline."""

    model_config = ConfigDict(populate_by_name=True)

    pot_id: str
    name: str
    episode_body: str
    source_description: str
    reference_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event time for the episode (defaults to UTC now).",
    )
    idempotency_key: Optional[str] = Field(
        default=None,
        description="Optional dedupe key for raw episodic ingest (requires DATABASE_URL).",
    )


class ResolveScopeBody(BaseModel):
    repo_name: Optional[str] = None
    branch: Optional[str] = None
    file_path: Optional[str] = None
    function_name: Optional[str] = None
    symbol: Optional[str] = None
    pr_number: Optional[int] = Field(default=None, ge=1)
    services: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    environment: Optional[str] = None
    ticket_ids: list[str] = Field(default_factory=list)
    user: Optional[str] = None
    source_refs: list[str] = Field(default_factory=list)


class ContextRecordPayload(BaseModel):
    type: str
    summary: str = Field(min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    visibility: str = Field(default="project")


class ContextRecordRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pot_id: str
    record: ContextRecordPayload
    scope: ResolveScopeBody = Field(default_factory=ResolveScopeBody)
    idempotency_key: Optional[str] = None
    occurred_at: Optional[datetime] = None


class ContextStatusRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pot_id: str
    scope: ResolveScopeBody = Field(default_factory=ResolveScopeBody)
    intent: Optional[str] = Field(
        default=None,
        description="Optional task intent used to return the recommended context_resolve recipe.",
    )


class ContextEventHttpBody(BaseModel):
    """Normalized context event (persisted, then reconciled when agent is configured)."""

    model_config = ConfigDict(populate_by_name=True)

    event_id: Optional[str] = Field(
        default=None,
        description="Stable id for this event; generated if omitted.",
    )
    ingestion_kind: Optional[str] = Field(
        default=None,
        description="Defaults to ``agent_reconciliation`` (the canonical kind).",
    )
    source_system: str
    event_type: str
    action: str
    pot_id: str
    provider: Optional[str] = None
    provider_host: Optional[str] = None
    repo_name: Optional[str] = None
    source_id: str
    source_event_id: Optional[str] = None
    artifact_refs: list[str] = Field(default_factory=list)
    occurred_at: Optional[datetime] = None
    received_at: Optional[datetime] = None
    payload: dict[str, Any] = Field(default_factory=dict)


def _wants_sync(sync: bool, x_sync: str | None) -> bool:
    if sync:
        return True
    if x_sync and x_sync.strip().lower() in ("true", "1", "yes"):
        return True
    return False


def _enforce(
    container: IngestionServerContainer,
    *,
    actor: Any | None,
    resource: str,
    action: str,
    pot_id: str | None = None,
    **extra: Any,
) -> PolicyDecision:
    """Single policy decision call site for HTTP routes (Phase 5).

    Translates :class:`PolicyDecision` to ``HTTPException`` so the route body
    no longer contains ad-hoc ``settings.is_enabled() / pots.resolve_pot``
    chains. The ``UNKNOWN_POT_DETAIL`` string is preserved verbatim for
    backwards-compatible client error parsing.
    """
    decision = container.policy().authorize(
        actor=actor,
        resource=resource,
        action=action,
        context={"pot_id": pot_id, **extra} if pot_id is not None or extra else {},
    )
    if decision.allowed:
        return decision
    detail = (
        UNKNOWN_POT_DETAIL
        if decision.reason == REASON_UNKNOWN_POT
        else decision.detail or decision.reason
    )
    raise HTTPException(status_code=decision.status_code, detail=detail)


def _parse_event_status_filters(
    raw: list[str] | None,
) -> tuple[IngestionEventStatus, ...] | None:
    if not raw:
        return None
    allowed: frozenset[str] = frozenset({"queued", "processing", "done", "error"})
    bad = [x for x in raw if x not in allowed]
    if bad:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status filter(s): {bad}; use queued, processing, done, or error.",
        )
    return tuple(raw)  # type: ignore[return-value]


_WINDOW_RE = re.compile(r"^(\d+)\s*([mhdw])$")
_WINDOW_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def _parse_window(window: str | None) -> timedelta | None:
    """Parse a relative lookback like ``24h`` / ``7d`` / ``2w`` / ``30m``.

    Returns ``None`` for an unparseable value so the caller can fall back to
    an unbounded (or explicit ``since``) window rather than 500.
    """
    if not window:
        return None
    match = _WINDOW_RE.match(window.strip().lower())
    if not match:
        return None
    amount = int(match.group(1))
    return timedelta(**{_WINDOW_UNITS[match.group(2)]: amount})


def create_context_router(
    *,
    require_auth: Callable[..., Any],
    get_container: Callable[..., IngestionServerContainer],
    get_db: Callable[..., Any],
    get_db_optional: Callable[..., Any],
) -> APIRouter:
    """Build context API routes with injected FastAPI dependencies.

    Authorization is mediated by :class:`PolicyPort` for every route — no
    flag toggles enforcement. Hosts wire ``require_auth`` for principal
    resolution; pot-scoped access checks are inside the policy adapter.
    """
    router = APIRouter()

    def _resolve_actor(auth_user: Any, request: Request) -> Actor:
        """Derive Actor from the authenticated principal only.

        ``user_id`` comes solely from what ``require_auth`` resolved — never
        from request headers. ``X-Potpie-Client`` is an untrusted hint that
        may only select a non-privileged client surface (``cli``/``mcp``/
        ``http``); a caller can never assert the server-trusted ``system`` or
        ``webhook`` surfaces (those are stamped by internal jobs / the
        signature-verified webhook handler, which build :class:`Actor`
        directly). ``auth_method`` reflects the transport that actually
        authenticated this request, not a client claim.
        """
        user_id: str
        if isinstance(auth_user, dict):
            user_id = str(auth_user.get("user_id") or auth_user.get("id") or "unknown")
        elif hasattr(auth_user, "user_id"):
            user_id = str(getattr(auth_user, "user_id"))
        elif isinstance(auth_user, str):
            user_id = auth_user
        else:
            user_id = "unknown"
        declared = request.headers.get("x-potpie-client") if request else None
        norm = normalize_surface(declared)
        # Privileged surfaces are server-stamped only — a client header may
        # never escalate to "system"/"webhook".
        surface: ActorSurface = "http"
        if norm is not None and norm in _CLIENT_DECLARABLE_SURFACES:
            surface = norm
        client_name = request.headers.get("x-potpie-client-name") if request else None
        client_name = (
            client_name.strip() if client_name and client_name.strip() else None
        )
        if client_name:
            # Free-form, client-asserted metadata — clamp so a hostile value
            # cannot bloat audit logs / graph actor properties.
            client_name = client_name[:64]
        return Actor(
            user_id=user_id,
            surface=surface,
            client_name=client_name,
            auth_method="api_key",
        )

    @router.post(
        "/ingest",
        summary="Add episodic episode",
        description=(
            "Ingest a narrative event for the pot (group_id); routed through the reconciliation agent. "
            "When DATABASE_URL is set, defaults to async (202): event is persisted then applied by a worker. "
            "Use sync=true or header X-Context-Ingest-Sync for inline apply."
        ),
    )
    def post_ingest_episode(
        body: IngestEpisodeRequest,
        request: Request,
        auth_user: Any = Depends(require_auth),
        container: IngestionServerContainer = Depends(get_container),
        db: Session | None = Depends(get_db_optional),
        sync: bool = Query(
            False,
            description="Synchronous apply after persist (200). Default false = async (202) when DATABASE_URL is set.",
        ),
        x_context_ingest_sync: str | None = Header(
            None,
            alias="X-Context-Ingest-Sync",
            description="If true/1/yes, same as sync=true.",
        ),
    ):
        want_sync = _wants_sync(sync, x_context_ingest_sync)
        actor = _resolve_actor(auth_user, request)
        _enforce(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_INGEST_EPISODE,
            pot_id=body.pot_id,
        )
        try:
            result = submit_raw_episode(
                container=container,
                db=db,
                pot_id=body.pot_id,
                name=body.name,
                episode_body=body.episode_body,
                source_description=body.source_description,
                reference_time=body.reference_time,
                idempotency_key=body.idempotency_key,
                sync=want_sync,
                source_channel=actor.surface,
                actor=actor,
            )
            if db is not None:
                if result.ok:
                    db.commit()
                else:
                    db.rollback()
        except Exception:
            if db is not None:
                db.rollback()
            raise

        if not result.ok:
            if result.status == "duplicate":
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "duplicate_ingest",
                        "event_id": result.event_id,
                        "message": "Duplicate idempotency key or conflicting event.",
                    },
                )
            if result.status == "reconciliation_rejected":
                payload = {
                    "status": "reconciliation_rejected",
                    "event_id": result.event_id,
                    "mutation_id": None,
                    "errors": list(result.reconciliation_errors or []),
                    "downgrades": list(result.downgrades or []),
                }
                if _ingest_rejection_returns_422():
                    return JSONResponse(status_code=422, content=payload)
                raise HTTPException(
                    status_code=503,
                    detail=result.error or "reconciliation_rejected",
                )
            err = result.error or "ingest_failed"
            if err == "unknown_pot_id":
                raise HTTPException(status_code=404, detail=UNKNOWN_POT_DETAIL)
            if err == "async_requires_database":
                raise HTTPException(
                    status_code=503,
                    detail="Async raw ingest requires DATABASE_URL (or pass sync=true for inline context graph write).",
                )
            if err == "context_graph_disabled":
                raise HTTPException(
                    status_code=503,
                    detail="Context graph is disabled (opt in by unsetting CONTEXT_GRAPH_ENABLED or setting true).",
                )
            raise HTTPException(status_code=503, detail=err)

        if result.status == "applied":
            return {
                "status": "applied",
                "mutation_id": result.mutation_id,
                "event_id": result.event_id,
                "job_id": result.job_id,
                "errors": [],
                "downgrades": list(result.downgrades or []),
            }
        return JSONResponse(
            status_code=202,
            content={
                "status": "queued",
                "event_id": result.event_id,
                "job_id": result.job_id,
                "downgrades": list(result.downgrades or []),
            },
        )

    @router.post(
        "/reset",
        summary="[operator] Hard-reset pot context graph",
        description=(
            "**Operator/admin action — destructive and not part of the agent surface.** "
            "Deletes Postgres reconciliation/ingestion rows for the pot first (so async "
            "workers cannot re-apply after the graph is cleared), then the canonical "
            "Entity nodes and :RELATES_TO edges in the pot's Neo4j partition. "
            "There is no dry-run mode: callers must scope to exactly one ``pot_id``. "
            "Each successful call emits a ``context_engine.operator_audit`` log record."
        ),
        tags=[OPERATOR_TAG],
    )
    def post_hard_reset(
        body: HardResetRequest,
        actor: Any = Depends(require_auth),
        container: IngestionServerContainer = Depends(get_container),
        db: Session | None = Depends(get_db_optional),
    ) -> dict[str, Any]:
        _enforce(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_RESET,
            pot_id=body.pot_id,
        )
        use_ledger = db is not None and not body.skip_ledger
        ledger = SqlAlchemyIngestionLedger(db) if use_ledger else None
        reconciliation_ledger = (
            SqlAlchemyReconciliationLedger(db) if use_ledger else None
        )
        assert container.context_graph is not None  # policy guarantees this
        out = hard_reset_pot(
            container.context_graph,
            body.pot_id,
            ledger=ledger,
            reconciliation_ledger=reconciliation_ledger,
        )
        if not out.get("ok"):
            _audit_operator_action(
                action="hard_reset_pot",
                pot_id=body.pot_id,
                actor=actor,
                dry_run=False,
                extra={
                    "skip_ledger": bool(body.skip_ledger),
                    "outcome": "failed",
                    "error": out.get("error"),
                },
            )
            raise HTTPException(
                status_code=502, detail=out.get("error") or "reset_failed"
            )
        _audit_operator_action(
            action="hard_reset_pot",
            pot_id=body.pot_id,
            actor=actor,
            dry_run=False,
            extra={"skip_ledger": bool(body.skip_ledger), "outcome": "ok"},
        )
        return out

    @router.post(
        "/events/reconcile",
        summary="Record event and run ingestion agent (reconciliation)",
        description=(
            "Canonical event submission path. Persists a row in ``context_events`` "
            "(deduped by scope + source_system + source_id), then runs the configured "
            "ingestion agent and applies the plan asynchronously."
        ),
    )
    def post_events_reconcile(
        body: ContextEventHttpBody,
        actor: Any = Depends(require_auth),
        container: IngestionServerContainer = Depends(get_container),
        db: Session = Depends(get_db),
    ):
        _enforce(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_SUBMIT_EVENT,
            pot_id=body.pot_id,
        )

        svc = container.ingestion_submission(db)
        req = IngestionSubmissionRequest(
            pot_id=body.pot_id,
            ingestion_kind=body.ingestion_kind or INGESTION_KIND_AGENT_RECONCILIATION,
            source_channel="http",
            source_system=body.source_system,
            event_type=body.event_type,
            action=body.action,
            payload=dict(body.payload),
            event_id=body.event_id,
            source_id=body.source_id,
            provider=body.provider,
            provider_host=body.provider_host,
            repo_name=body.repo_name,
            source_event_id=body.source_event_id,
            artifact_refs=tuple(body.artifact_refs),
            occurred_at=body.occurred_at,
        )
        try:
            out = svc.submit(req)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if out.duplicate:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "duplicate_event",
                    "event_id": out.event_id,
                    "message": "Event already recorded for this scope and source_id.",
                },
            )
        return JSONResponse(
            status_code=202,
            content={
                "status": "queued",
                "event_id": out.event_id,
                "batch_id": out.job_id,
            },
        )

    @router.get("/events/{event_id}", summary="Fetch a persisted context event")
    def get_context_event(
        event_id: str,
        actor: Any = Depends(require_auth),
        container: IngestionServerContainer = Depends(get_container),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        q = container.event_query_service(db)
        ev = q.get_event(event_id)
        if ev is None:
            raise HTTPException(status_code=404, detail="Unknown event_id")
        _enforce(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_READ,
            pot_id=ev.pot_id,
        )
        reco = SqlAlchemyReconciliationLedger(db)
        run_payload = []
        for run in reco.list_runs_for_event(event_id):
            work_events = reco.list_work_events_for_run(run.id)
            run_payload.append(
                {
                    "id": run.id,
                    "attempt_number": run.attempt_number,
                    "status": run.status,
                    "agent_name": run.agent_name,
                    "agent_version": run.agent_version,
                    "toolset_version": run.toolset_version,
                    "plan_summary": run.plan_summary,
                    "episode_count": run.episode_count,
                    "entity_mutation_count": run.entity_mutation_count,
                    "edge_mutation_count": run.edge_mutation_count,
                    "error": run.error,
                    "started_at": run.started_at.isoformat()
                    if run.started_at
                    else None,
                    "completed_at": run.completed_at.isoformat()
                    if run.completed_at
                    else None,
                    "work_events": [
                        {
                            "id": w.id,
                            "sequence": w.sequence,
                            "event_kind": w.event_kind,
                            "title": w.title,
                            "body": w.body,
                            "payload": w.payload,
                            "created_at": w.created_at.isoformat(),
                        }
                        for w in work_events
                    ],
                }
            )
        out = ingestion_event_to_payload(ev)
        out["reconciliation_runs"] = run_payload
        return out

    @router.post(
        "/events/{event_id}/retry",
        summary="Re-enqueue an ingestion event",
        description=(
            "Resets the event back to ``queued`` and coalesces it into the pot's "
            "open batch so the reconciliation worker picks it up again. Works for "
            "events in any lifecycle (error, done, processing, queued); "
            "``upsert_open_batch_for_pot`` opens a fresh pending batch when one "
            "is already in-flight, so a retry never races a running batch."
        ),
    )
    def retry_context_event(
        event_id: str,
        actor: Any = Depends(require_auth),
        container: IngestionServerContainer = Depends(get_container),
        db: Session = Depends(get_db),
    ):
        events = container.ingestion_event_store(db)
        ev = events.get_event(event_id)
        if ev is None:
            raise HTTPException(status_code=404, detail="Unknown event_id")
        _enforce(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_SUBMIT_EVENT,
            pot_id=ev.pot_id,
        )

        reco = container.reconciliation_ledger(db)
        batches = container.batch_repository(db)
        jobs = container.jobs

        reco.mark_event_for_retry(event_id)
        batch_id = batches.upsert_open_batch_for_pot(ev.pot_id, event_id)
        reco.mark_event_queued(event_id)
        if jobs is not None:
            try:
                jobs.enqueue_batch(batch_id)
            except Exception:
                _logger.exception(
                    "retry_context_event: enqueue_batch failed for batch %s "
                    "(event %s); batch is durable, next event will re-enqueue",
                    batch_id,
                    event_id,
                )

        return JSONResponse(
            status_code=202,
            content={
                "status": "queued",
                "event_id": event_id,
                "batch_id": batch_id,
                "previous_lifecycle_status": ev.status,
            },
        )

    @router.post(
        "/pots/{pot_id}/events/batch-retry",
        summary="Re-enqueue a set of events as one open batch",
        description=(
            "Bulk variant of ``/events/{event_id}/retry``. Validates every "
            "event belongs to ``pot_id`` up-front (no partial apply on a bad "
            "set), then in three bulk statements marks the whole set for "
            "retry, drops it into the pot's open pending batch in one "
            "insert, and enqueues that batch once. The worker hands the "
            "entire batch to the agent in one chunked pass. Pre-existing "
            "in-flight batches are not interrupted — a fresh pending batch "
            "is opened in that case, so this is always safe to call."
        ),
    )
    def batch_retry_events(
        pot_id: str,
        body: BatchRetryEventsRequest,
        actor: Any = Depends(require_auth),
        container: IngestionServerContainer = Depends(get_container),
        db: Session = Depends(get_db),
    ):
        if not body.event_ids:
            raise HTTPException(status_code=400, detail="event_ids must be non-empty")
        # Cap to a sane batch — bigger bulk ops should use windowed batching.
        if len(body.event_ids) > 200:
            raise HTTPException(
                status_code=400,
                detail="event_ids exceeds 200; submit smaller batches",
            )
        # De-dupe inputs while preserving order so the response mirrors the request.
        seen: set[str] = set()
        unique_ids: list[str] = []
        for eid in body.event_ids:
            if eid in seen:
                continue
            seen.add(eid)
            unique_ids.append(eid)

        _enforce(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_SUBMIT_EVENT,
            pot_id=pot_id,
        )

        events_store = container.ingestion_event_store(db)
        # Validate each event exists and belongs to this pot up-front so we
        # don't partially apply a retry on a mixed-pot bulk.
        rows = []
        for eid in unique_ids:
            ev = events_store.get_event(eid)
            if ev is None:
                raise HTTPException(status_code=404, detail=f"Unknown event_id: {eid}")
            if ev.pot_id != pot_id:
                raise HTTPException(
                    status_code=400,
                    detail=(f"event_id {eid} does not belong to pot {pot_id}"),
                )
            rows.append(ev)

        reco = container.reconciliation_ledger(db)
        batches = container.batch_repository(db)
        jobs = container.jobs

        event_ids = [ev.event_id for ev in rows]

        # Three bulk statements for the whole set — not a per-event loop.
        # Each step is all-or-nothing, so the request can't half-apply: if
        # any step raises it propagates as a 5xx with nothing partially
        # transitioned, rather than silently queueing some events and
        # dropping others. The full set lands in one pending batch, which
        # the worker then hands to the agent in a single chunked pass.
        reco.mark_events_for_retry(event_ids)
        batch_id = batches.add_events_to_open_batch_for_pot(pot_id, event_ids)
        reco.mark_events_queued(event_ids)

        if jobs is not None:
            try:
                jobs.enqueue_batch(batch_id)
            except Exception:
                # The batch is durable; the windowed flusher / next event
                # on this pot re-enqueues. Don't fail the request for a
                # transient broker blip after the writes committed.
                _logger.exception(
                    "batch_retry_events: enqueue_batch failed for batch %s "
                    "(pot %s); batch is durable, will be re-enqueued",
                    batch_id,
                    pot_id,
                )

        return JSONResponse(
            status_code=202,
            content={
                "status": "queued",
                "pot_id": pot_id,
                "batch_id": batch_id,
                "event_ids": event_ids,
                "count": len(event_ids),
            },
        )

    @router.get(
        "/pots/{pot_id}/events",
        summary="List ingestion events for a pot (latest first)",
    )
    def list_pot_events(
        pot_id: str,
        actor: Any = Depends(require_auth),
        container: IngestionServerContainer = Depends(get_container),
        db: Session = Depends(get_db),
        cursor: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        status: list[str] | None = Query(
            None,
            description="Filter by canonical lifecycle (repeat param for multiple).",
        ),
        ingestion_kind: list[str] | None = Query(
            None,
            description="Filter by ingestion_kind (repeat for multiple).",
        ),
        source_system: list[str] | None = Query(
            None,
            description="Filter by source_system (repeat for multiple).",
        ),
        source_channel: list[str] | None = Query(
            None,
            description="Filter by source_channel (cli/mcp/http/webhook).",
        ),
        actor_user_id: list[str] | None = Query(
            None,
            description="Filter by Potpie user id that submitted the event.",
        ),
        actor_surface: list[str] | None = Query(
            None,
            description="Filter by actor surface (cli/mcp/http/webhook/system).",
        ),
        from_date: datetime | None = Query(
            None,
            description="Only return events submitted at or after this ISO 8601 datetime.",
        ),
        to_date: datetime | None = Query(
            None,
            description="Only return events submitted at or before this ISO 8601 datetime.",
        ),
        q: str | None = Query(
            None,
            description=(
                "Free-text needle (case-insensitive) matched against "
                "event_id, repo_name, event_type, action, and the raw "
                "episode title fields."
            ),
            max_length=200,
        ),
    ) -> dict[str, Any]:
        _enforce(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_READ,
            pot_id=pot_id,
        )
        # Normalize empty / whitespace-only q to None so the filter dataclass
        # treats "no needle" uniformly.
        q_norm = q.strip() if q else None
        if q_norm == "":
            q_norm = None
        filters = EventListFilters(
            statuses=_parse_event_status_filters(status),
            ingestion_kinds=tuple(ingestion_kind) if ingestion_kind else None,
            source_systems=tuple(source_system) if source_system else None,
            source_channels=tuple(source_channel) if source_channel else None,
            actor_user_ids=tuple(actor_user_id) if actor_user_id else None,
            actor_surfaces=tuple(actor_surface) if actor_surface else None,
            submitted_after=from_date,
            submitted_before=to_date,
            q=q_norm,
        )
        page = container.event_query_service(db).list_events(
            pot_id, filters, cursor=cursor, limit=limit
        )
        return {
            "items": [ingestion_event_to_payload(ev) for ev in page.items],
            "next_cursor": page.next_cursor,
        }

    @router.get(
        "/pots/{pot_id}/timeline",
        summary="Activity timeline for a pot (what changed, latest first)",
        description=(
            "Recent Activity events (PR merged, deploy, alert, discussion, "
            "decision), newest first. Optionally anchored to one or more "
            "services and/or restricted to a set of verb_class kinds. Backed "
            "by the canonical timeline reader over the activity claim graph "
            "(TOUCHED / PERFORMED / MENTIONS); the window is applied to "
            "each event's occurred_at (claim valid_at)."
        ),
    )
    async def get_pot_timeline(
        pot_id: str,
        actor: Any = Depends(require_auth),
        container: IngestionServerContainer = Depends(get_container),
        service: list[str] | None = Query(
            None,
            description="Anchor to one or more service names (repeat for multiple).",
        ),
        window: str | None = Query(
            "14d",
            description="Relative lookback: e.g. '24h', '7d', '14d', '90d'. Ignored when 'since' is set.",
        ),
        since: datetime | None = Query(
            None,
            description="Only events at/after this ISO 8601 datetime (overrides 'window').",
        ),
        until: datetime | None = Query(
            None, description="Only events at/before this ISO 8601 datetime."
        ),
        verb_class: list[str] | None = Query(
            None,
            description=(
                "Restrict to these event kinds "
                "(code_change/deployment/alert/discussion/decision; repeat for multiple)."
            ),
        ),
        limit: int = Query(30, ge=1, le=50),
        include_invalidated: bool = Query(False),
    ) -> dict[str, Any]:
        _enforce(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_READ,
            pot_id=pot_id,
        )
        if container.graph is None:
            raise HTTPException(
                status_code=503,
                detail="Canonical graph service is not configured.",
            )
        # An explicit `since` wins; otherwise derive it from the relative
        # `window`. valid_at is compared as a UTC ISO string downstream, so
        # anchor the derived bound to UTC.
        resolved_since = since
        if resolved_since is None:
            delta = _parse_window(window)
            if delta is not None:
                resolved_since = datetime.now(timezone.utc) - delta

        envelope = container.graph.resolve(
            ResolveRequest(
                pot_id=pot_id,
                include=("timeline",),
                scope={
                    "services": [s.strip() for s in (service or []) if s and s.strip()]
                },
                since=resolved_since,
                until=until,
                include_invalidated=include_invalidated,
                max_items=limit,
                metadata={"http_timeline": True},
            )
        )
        envelope_payload = envelope.to_dict()

        wanted = {v.strip().lower() for v in (verb_class or []) if v and v.strip()}
        items: list[dict[str, Any]] = []
        for entry in envelope_payload.get("items", []):
            payload = entry.get("payload") or {}
            kind = payload.get("verb_class")
            if wanted and (kind or "").lower() not in wanted:
                continue
            items.append(
                {
                    "id": entry.get("candidate_key"),
                    "activity_key": payload.get("activity_key")
                    or payload.get("subject_key"),
                    "timestamp": payload.get("valid_at"),
                    "verb_class": kind,
                    "title": payload.get("fact"),
                    "predicate": payload.get("predicate"),
                    "subject_key": payload.get("subject_key"),
                    "object_key": payload.get("object_key"),
                    "source_system": payload.get("source_system"),
                    "source_ref": payload.get("source_ref"),
                    "evidence_strength": payload.get("evidence_strength"),
                    "score": entry.get("score"),
                }
            )
        return {
            "pot_id": pot_id,
            "items": items,
            "coverage": envelope_payload.get("overall_confidence"),
            "window": {
                "since": resolved_since.isoformat() if resolved_since else None,
                "until": until.isoformat() if until else None,
            },
        }

    # ----- Per-pot ingestion config & force flush ------------------------------
    # Lets the user toggle between immediate and windowed batching and
    # force-flush the open batch on demand. The "Queued: N ⚡" CTA in the
    # list view calls /ingest/flush.

    @router.get(
        "/pots/{pot_id}/ingestion-config",
        summary="Get the pot's ingestion mode and window",
    )
    def get_ingestion_config(
        pot_id: str,
        actor: Any = Depends(require_auth),
        container: IngestionServerContainer = Depends(get_container),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        _enforce(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_READ,
            pot_id=pot_id,
        )
        cfg = container.ingestion_config(db).get(pot_id)
        return {
            "pot_id": cfg.pot_id,
            "mode": cfg.mode,
            "window_minutes": cfg.window_minutes,
            "min_batch_size": cfg.min_batch_size,
        }

    @router.put(
        "/pots/{pot_id}/ingestion-config",
        summary="Update the pot's ingestion mode and window",
    )
    def put_ingestion_config(
        pot_id: str,
        body: IngestionConfigBody,
        actor: Any = Depends(require_auth),
        container: IngestionServerContainer = Depends(get_container),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        _enforce(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_SUBMIT_EVENT,
            pot_id=pot_id,
        )
        actor_uid = getattr(actor, "id", None) or getattr(actor, "sub", None)
        try:
            cfg = container.ingestion_config(db).set(
                pot_id=pot_id,
                mode=body.mode,  # type: ignore[arg-type]
                window_minutes=body.window_minutes,
                min_batch_size=body.min_batch_size,
                actor_user_id=actor_uid if isinstance(actor_uid, str) else None,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        db.commit()
        return {
            "pot_id": cfg.pot_id,
            "mode": cfg.mode,
            "window_minutes": cfg.window_minutes,
            "min_batch_size": cfg.min_batch_size,
        }

    @router.post(
        "/pots/{pot_id}/ingest/flush",
        summary="Force-enqueue the pot's open pending batch (manual flush)",
        description=(
            "Closes-and-enqueues the pot's open pending batch immediately, "
            "ignoring the windowed timer. Works for any ingestion mode — "
            "useful as an escape hatch if the user wants their queued "
            "events processed right now. Returns 200 with ``batch_id=None`` "
            "when there is nothing pending to flush."
        ),
    )
    def force_flush_pot_endpoint(
        pot_id: str,
        actor: Any = Depends(require_auth),
        container: IngestionServerContainer = Depends(get_container),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        from potpie_context_engine.application.use_cases.flush_windowed_batches import force_flush_pot

        _enforce(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_SUBMIT_EVENT,
            pot_id=pot_id,
        )
        batch_id = force_flush_pot(
            pot_id=pot_id,
            batches=container.batch_repository(db),
            jobs=container.jobs,
        )
        return {
            "pot_id": pot_id,
            "batch_id": batch_id,
            "status": "queued" if batch_id else "no_pending_batch",
        }

    @router.get(
        "/pots/{pot_id}/ingest/pipeline",
        summary="Open-batch / window state for the pipeline UI",
        description=(
            "Snapshot the pot's ingestion pipeline: mode + window, the open "
            "(pending) batch with its event count and window deadline. Backs "
            "the events-screen 'batched / queued' section + countdown."
        ),
    )
    def get_ingest_pipeline(
        pot_id: str,
        actor: Any = Depends(require_auth),
        container: IngestionServerContainer = Depends(get_container),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        from datetime import timedelta

        _enforce(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_READ,
            pot_id=pot_id,
        )
        cfg = container.ingestion_config(db).get(pot_id)
        batches = container.batch_repository(db)
        open_batch: dict[str, Any] | None = None
        open_id = batches.get_open_batch_id_for_pot(pot_id)
        if open_id:
            b = batches.get_batch(open_id)
            refs = batches.list_events_for_batch(open_id)
            pending = [r for r in refs if r.processed_at is None]
            created_at = getattr(b, "created_at", None) if b else None
            deadline = None
            if created_at is not None and cfg.mode == "windowed" and cfg.window_minutes:
                deadline = (
                    created_at + timedelta(minutes=cfg.window_minutes)
                ).isoformat()
            open_batch = {
                "batch_id": open_id,
                "created_at": created_at.isoformat() if created_at else None,
                "event_count": len(pending),
                "window_deadline": deadline,
            }
        return {
            "pot_id": pot_id,
            "mode": cfg.mode,
            "window_minutes": cfg.window_minutes,
            "min_batch_size": cfg.min_batch_size,
            "open_batch": open_batch,
            "queued_event_count": (open_batch["event_count"] if open_batch else 0),
        }

    # ----- Live event streaming -------------------------------------------------
    # Two endpoints: per-event activity (side panel) and per-pot status deltas
    # (list view). Both return ``StreamingResponse`` of newline-delimited JSON
    # so the chat-stream client pattern can be reused on the frontend.

    @router.get(
        "/events/{event_id}/stream",
        summary="Stream live agent activity for one event (NDJSON)",
        description=(
            "Tails the durable agent execution log for the batch this event "
            "belongs to: model text / thinking token deltas, tool calls + "
            "results, graph mutations, then a terminal ``{type:'end'}``. "
            "Replays records after ``cursor`` (a seq) then live-tails. The "
            "log is durable, so a reconnect resumes exactly where it left "
            "off and survives a worker crash."
        ),
    )
    def stream_event_activity(
        event_id: str,
        actor: Any = Depends(require_auth),
        container: IngestionServerContainer = Depends(get_container),
        db: Session = Depends(get_db),
        cursor: str | None = Query(
            None,
            description="Execution-log seq to resume after (last seen stream_id).",
        ),
        idle_timeout_seconds: float = Query(
            120.0,
            ge=1.0,
            le=600.0,
            description="Close the connection if no events arrive within this window.",
        ),
    ):
        events = container.event_query_service(db)
        ev = events.get_event(event_id)
        if ev is None:
            raise HTTPException(status_code=404, detail="Unknown event_id")
        _enforce(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_READ,
            pot_id=ev.pot_id,
        )

        batch_id = container.batch_repository(db).get_latest_batch_id_for_event(
            event_id
        )
        try:
            cursor_seq = int(cursor) if cursor not in (None, "") else 0
        except (TypeError, ValueError):
            cursor_seq = 0
        exec_log = container.agent_execution_log(db)

        def _iter() -> Any:
            if batch_id is None:
                # Event admitted but not yet batched/run. Emit a transient
                # end so the client backs off and reconnects until the
                # agent picks it up (the durable log appears then).
                yield _ndjson_line(
                    {
                        "type": "end",
                        "status": "queued",
                        "message": "Event is queued; agent has not started.",
                        "stream_id": str(cursor_seq),
                    }
                )
                return
            try:
                for event in exec_log.replay_and_tail(
                    batch_id=batch_id,
                    cursor_seq=cursor_seq,
                    idle_timeout_seconds=idle_timeout_seconds,
                ):
                    yield _ndjson_line(event)
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "stream_event_activity iterator failed for %s: %s",
                    event_id,
                    exc,
                )
                yield _ndjson_line(
                    {"type": "end", "status": "error", "message": str(exc)}
                )

        from fastapi.responses import StreamingResponse

        return StreamingResponse(
            _iter(),
            media_type="application/x-ndjson",
            headers={
                # Disable proxy buffering so chunks reach the browser immediately.
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get(
        "/pots/{pot_id}/events/stream",
        summary="Stream live status deltas for events in a pot (NDJSON)",
        description=(
            "Tails ``{type:'status'|'end'}`` records as events in the pot "
            "transition. Subscribed by the list view to update row indicators "
            "without polling. The stream never naturally ends — clients "
            "disconnect when they navigate away. ``idle_timeout_seconds`` "
            "bounds an unattended connection."
        ),
    )
    def stream_pot_events(
        pot_id: str,
        actor: Any = Depends(require_auth),
        container: IngestionServerContainer = Depends(get_container),
        cursor: str | None = Query(
            None,
            description="Redis stream id to resume after (last seen stream_id).",
        ),
        idle_timeout_seconds: float = Query(
            120.0,
            ge=1.0,
            le=600.0,
            description="Close the connection if no events arrive within this window.",
        ),
    ):
        _enforce(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_READ,
            pot_id=pot_id,
        )
        publisher = container.event_stream_publisher

        def _iter() -> Any:
            try:
                for event in publisher.replay_and_tail_pot_status(
                    pot_id=pot_id,
                    cursor=cursor,
                    idle_timeout_seconds=idle_timeout_seconds,
                ):
                    yield _ndjson_line(event)
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "stream_pot_events iterator failed for %s: %s", pot_id, exc
                )
                yield _ndjson_line(
                    {"type": "end", "status": "error", "message": str(exc)}
                )

        from fastapi.responses import StreamingResponse

        return StreamingResponse(
            _iter(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @router.post(
        "/record",
        summary="Record durable context through the minimal agent context port",
    )
    def post_context_record(
        body: ContextRecordRequest,
        request: Request,
        auth_user: Any = Depends(require_auth),
        container: IngestionServerContainer = Depends(get_container),
        db: Session = Depends(get_db),
        sync: bool = Query(
            False,
            description=(
                "Compatibility flag; context_record now applies through the "
                "deterministic semantic mutation path."
            ),
        ),
        x_context_ingest_sync: str | None = Header(
            None,
            alias="X-Context-Ingest-Sync",
        ),
    ):
        actor = _resolve_actor(auth_user, request)
        _enforce(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_RECORD,
            pot_id=body.pot_id,
        )
        try:
            receipt, record_type, source_id = record_durable_context(
                container.ingestion_submission(db),
                pot_id=body.pot_id,
                record=DurableContextPayload(
                    record_type=body.record.type,
                    summary=body.record.summary,
                    details=dict(body.record.details),
                    source_refs=tuple(body.record.source_refs),
                    confidence=body.record.confidence,
                    visibility=body.record.visibility,
                ),
                scope={
                    **body.scope.model_dump(exclude_none=True),
                    "source_refs": list(body.scope.source_refs),
                },
                actor=actor,
                idempotency_key=body.idempotency_key,
                occurred_at=body.occurred_at,
                sync=_wants_sync(sync, x_context_ingest_sync),
            )
            db.commit()
        except ValueError as exc:
            db.rollback()
            err = str(exc)
            if err == "unknown_pot_id":
                raise HTTPException(status_code=404, detail=UNKNOWN_POT_DETAIL) from exc
            if err in {"context_graph_disabled", "async_requires_database"}:
                raise HTTPException(status_code=503, detail=err) from exc
            raise HTTPException(status_code=400, detail=err) from exc
        except Exception:
            db.rollback()
            raise

        return {
            "ok": receipt.error is None,
            "status": "duplicate" if receipt.duplicate else receipt.status,
            "event_id": receipt.event_id,
            "job_id": receipt.job_id,
            "record_type": record_type,
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

    @router.post(
        "/status",
        summary="Pot readiness, source health, event ledger, resolver capabilities",
    )
    def post_context_status(
        body: ContextStatusRequest,
        actor: Any = Depends(require_auth),
        container: IngestionServerContainer = Depends(get_container),
        db: Session | None = Depends(get_db_optional),
    ) -> dict[str, Any]:
        _enforce(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_READ,
            pot_id=body.pot_id,
        )
        report = report_status(
            container,
            pot_id=body.pot_id,
            scope={
                **body.scope.model_dump(exclude_none=True),
                "source_refs": list(body.scope.source_refs),
            },
            intent=body.intent,
            db=db,
        )
        if report.unknown_pot:
            raise HTTPException(status_code=404, detail=UNKNOWN_POT_DETAIL)
        return report.payload

    @router.post(
        "/query/context-graph",
        summary="Unsupported legacy ContextGraphQuery endpoint",
        description=(
            "Legacy remote graph-query clients are no longer supported. Use the "
            "local HostShell / AgentContextPort / GraphService surfaces."
        ),
    )
    async def post_context_graph_query(
        body: dict[str, Any],
        actor: Any = Depends(require_auth),
        container: IngestionServerContainer = Depends(get_container),
    ) -> dict[str, Any]:
        del body
        _enforce(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_READ,
            pot_id=None,
        )
        raise HTTPException(
            status_code=501,
            detail={
                "code": "http_context_graph_query_not_supported",
                "message": (
                    "Remote ContextGraphQuery is no longer a supported client surface."
                ),
                "recommended_next_action": (
                    "Use local context_resolve/context_search or graph read."
                ),
            },
        )

    return router


context_router = create_context_router(
    require_auth=require_api_key,
    get_container=get_container_or_503,
    get_db=get_db,
    get_db_optional=get_db_optional,
)

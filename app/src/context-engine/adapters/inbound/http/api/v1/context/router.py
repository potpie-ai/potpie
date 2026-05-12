"""Context graph HTTP API (standalone service + host-injected dependencies)."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from adapters.inbound.http.api.v1.context.event_payload import ingestion_event_to_payload
from adapters.inbound.http.deps import (
    get_container_or_503,
    get_db,
    get_db_optional,
    require_api_key,
)
from adapters.outbound.graphiti.episodic import GraphitiEpisodicAdapter
from adapters.outbound.postgres.ledger import SqlAlchemyIngestionLedger
from adapters.outbound.postgres.reconciliation_ledger import (
    SqlAlchemyReconciliationLedger,
)
from application.use_cases.hard_reset_pot import hard_reset_pot
from application.use_cases.record_durable_context import (
    DurableContextPayload,
    record_durable_context,
)
from application.use_cases.report_status import report_status
from application.use_cases.submit_raw_episode import submit_raw_episode
from bootstrap.container import ContextEngineContainer
from domain.actor import Actor, normalize_surface
from domain.graph_query import ContextGraphQuery
from domain.ingestion_event_models import (
    EventListFilters,
    IngestionEventStatus,
    IngestionSubmissionRequest,
)
from domain.ingestion_kinds import INGESTION_KIND_AGENT_RECONCILIATION
from domain.ports.policy import (
    ACTION_POT_INGEST_EPISODE,
    ACTION_POT_MAINTENANCE,
    ACTION_POT_READ,
    ACTION_POT_RECORD,
    ACTION_POT_RESET,
    ACTION_POT_RESOLVE_CONFLICT,
    ACTION_POT_SUBMIT_EVENT,
    RESOURCE_POT,
    REASON_UNKNOWN_POT,
    PolicyDecision,
)

UNKNOWN_POT_DETAIL = (
    "Unknown pot_id for this user (create with POST /api/v2/context/pots "
    "and attach at least one repository)."
)

_logger = logging.getLogger(__name__)


# Operator/admin routes (``/reset``, ``/conflicts/*``, ``/maintenance/*``) are
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


def _context_graph_jsonable(value: Any) -> Any:
    """Convert graph adapter payloads, including Neo4j temporal values, to JSON-safe data."""
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
    iso_format = getattr(value, "iso_format", None)
    if callable(iso_format):
        try:
            return str(iso_format())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _context_graph_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_context_graph_jsonable(v) for v in value]
    return str(value)


class HardResetRequest(BaseModel):
    """Hard-delete all context-graph data for a pot."""

    model_config = ConfigDict(populate_by_name=True)

    pot_id: str = Field(
        description="Pot scope id (Graphiti group_id / Neo4j partition)."
    )
    skip_ledger: bool = Field(
        default=False,
        description="If true, only clear Graphiti + structural Neo4j; do not delete Postgres ledger rows.",
    )


class IngestEpisodeRequest(BaseModel):
    """Raw Graphiti episode (same fields as episodic.add_episode)."""

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


class ClassifyModifiedEdgesRequest(BaseModel):
    """Reclassify vague ``MODIFIED`` episodic edges (Neo4j maintenance)."""

    model_config = ConfigDict(populate_by_name=True)

    pot_id: str
    dry_run: bool = Field(
        default=True,
        description=(
            "If false, updates RELATES_TO.name and lifecycle_status. "
            "Server must set CONTEXT_ENGINE_CLASSIFY_MODIFIED_EDGES=1 and "
            "CONTEXT_ENGINE_ALLOW_EDGE_CLASSIFY_WRITE=1."
        ),
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


class ConflictListRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pot_id: str


class ConflictResolveRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pot_id: str
    issue_uuid: str
    action: str = "supersede_older"


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
    provider: str = "github"
    provider_host: str = "github.com"
    repo_name: str
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
    container: ContextEngineContainer,
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


def create_context_router(
    *,
    require_auth: Callable[..., Any],
    get_container: Callable[..., ContextEngineContainer],
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
        """Derive Actor from auth principal + self-declared client headers."""
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
        surface = normalize_surface(declared) or "http"
        client_name = (
            request.headers.get("x-potpie-client-name") if request else None
        )
        client_name = client_name.strip() if client_name and client_name.strip() else None
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
            "Ingest a raw episode into Graphiti for the pot (group_id). "
            "When DATABASE_URL is set, defaults to async (202): event is persisted then applied by a worker. "
            "Use sync=true or header X-Context-Ingest-Sync for inline apply."
        ),
    )
    def post_ingest_episode(
        body: IngestEpisodeRequest,
        request: Request,
        auth_user: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
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
                    "episode_uuid": None,
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
                "episode_uuid": result.episode_uuid,
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
            "workers cannot re-apply after the graph is cleared), then Graphiti episodic data "
            "and structural Entity/FILE/NODE nodes in the default Neo4j database. "
            "There is no dry-run mode: callers must scope to exactly one ``pot_id``. "
            "Each successful call emits a ``context_engine.operator_audit`` log record."
        ),
        tags=[OPERATOR_TAG],
    )
    def post_hard_reset(
        body: HardResetRequest,
        actor: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
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
        container: ContextEngineContainer = Depends(get_container),
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
        container: ContextEngineContainer = Depends(get_container),
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

    @router.get(
        "/pots/{pot_id}/events",
        summary="List ingestion events for a pot (latest first)",
    )
    def list_pot_events(
        pot_id: str,
        actor: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
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
    ) -> dict[str, Any]:
        _enforce(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_READ,
            pot_id=pot_id,
        )
        filters = EventListFilters(
            statuses=_parse_event_status_filters(status),
            ingestion_kinds=tuple(ingestion_kind) if ingestion_kind else None,
            source_systems=tuple(source_system) if source_system else None,
            source_channels=tuple(source_channel) if source_channel else None,
            actor_user_ids=tuple(actor_user_id) if actor_user_id else None,
            actor_surfaces=tuple(actor_surface) if actor_surface else None,
            submitted_after=from_date,
            submitted_before=to_date,
        )
        page = container.event_query_service(db).list_events(
            pot_id, filters, cursor=cursor, limit=limit
        )
        return {
            "items": [ingestion_event_to_payload(ev) for ev in page.items],
            "next_cursor": page.next_cursor,
        }

    @router.post(
        "/record",
        summary="Record durable context through the minimal agent context port",
    )
    def post_context_record(
        body: ContextRecordRequest,
        request: Request,
        auth_user: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
        db: Session = Depends(get_db),
        sync: bool = Query(
            False, description="Inline reconcile (200) instead of enqueue (202)."
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
        container: ContextEngineContainer = Depends(get_container),
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
        summary="Direct ContextGraphQuery endpoint",
        description=(
            "Executes the minimal query surface directly: one graph query method "
            "with goal, strategy, scope, filters, temporal controls, and budget."
        ),
    )
    async def post_context_graph_query(
        body: ContextGraphQuery,
        actor: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ) -> dict[str, Any]:
        _enforce(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_READ,
            pot_id=body.pot_id,
        )
        if container.context_graph is None:
            raise HTTPException(
                status_code=503,
                detail="Unified context graph query port is not configured.",
            )
        result = await container.context_graph.query_async(body)
        return _context_graph_jsonable(result.model_dump())

    @router.post(
        "/conflicts/list",
        summary="[operator] List open predicate-family conflicts (QualityIssue)",
        description=(
            "Operator/admin read for graph hygiene. Returns open conflicts "
            "surfaced by predicate-family invariants. Paired with "
            "``/conflicts/resolve`` for repair workflows."
        ),
        tags=[OPERATOR_TAG],
    )
    def post_conflicts_list(
        body: ConflictListRequest,
        actor: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ) -> dict[str, Any]:
        decision = _enforce(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_READ,
            pot_id=body.pot_id,
        )
        if not container.episodic.enabled:
            return {"ok": False, "items": [], "error": "episodic_graph_unavailable"}
        resolved_pot_id = decision.metadata.get("resolved_pot_id", body.pot_id)
        try:
            items = container.episodic.list_open_conflicts(resolved_pot_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"ok": True, "items": items}

    @router.post(
        "/conflicts/resolve",
        summary="[operator] Resolve an open predicate-family conflict",
        description=(
            "**Operator/admin action — mutates graph state.** Resolves one "
            "conflict (default action ``supersede_older``). Each call emits "
            "a ``context_engine.operator_audit`` log record with the actor, "
            "pot, issue uuid, and action. No dry-run: operators list first "
            "via ``/conflicts/list`` and resolve targeted rows."
        ),
        tags=[OPERATOR_TAG],
    )
    def post_conflicts_resolve(
        body: ConflictResolveRequest,
        actor: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ) -> dict[str, Any]:
        decision = _enforce(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_RESOLVE_CONFLICT,
            pot_id=body.pot_id,
        )
        resolved_pot_id = decision.metadata.get("resolved_pot_id", body.pot_id)
        try:
            out = container.episodic.resolve_open_conflict(
                resolved_pot_id, body.issue_uuid, body.action
            )
        except Exception as exc:
            _audit_operator_action(
                action="resolve_conflict",
                pot_id=body.pot_id,
                actor=actor,
                extra={
                    "issue_uuid": body.issue_uuid,
                    "conflict_action": body.action,
                    "outcome": "error",
                    "error": str(exc),
                },
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        _audit_operator_action(
            action="resolve_conflict",
            pot_id=body.pot_id,
            actor=actor,
            extra={
                "issue_uuid": body.issue_uuid,
                "conflict_action": body.action,
                "outcome": "ok" if out.get("ok", True) else "failed",
            },
        )
        return out

    @router.post(
        "/maintenance/classify-modified-edges",
        summary="[operator] Reclassify vague MODIFIED episodic edges",
        description=(
            "**Operator/admin maintenance job.** Dry-run by default "
            "(``dry_run=true``); the response summarises proposed changes "
            "without touching Neo4j. Writes require BOTH "
            "``CONTEXT_ENGINE_CLASSIFY_MODIFIED_EDGES=1`` and "
            "``CONTEXT_ENGINE_ALLOW_EDGE_CLASSIFY_WRITE=1`` on the server. "
            "Every invocation emits a ``context_engine.operator_audit`` log "
            "record carrying the actor, pot, and ``dry_run`` flag."
        ),
        tags=[OPERATOR_TAG],
    )
    def post_classify_modified_edges(
        body: ClassifyModifiedEdgesRequest,
        actor: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ) -> dict[str, Any]:
        _enforce(
            container,
            actor=actor,
            resource=RESOURCE_POT,
            action=ACTION_POT_MAINTENANCE,
            pot_id=body.pot_id,
            dry_run=body.dry_run,
        )
        if not isinstance(container.episodic, GraphitiEpisodicAdapter):
            raise HTTPException(
                status_code=501,
                detail="Episodic backend does not support this maintenance job.",
            )
        result = container.episodic.classify_modified_edges_for_pot(
            body.pot_id, dry_run=body.dry_run
        )
        _audit_operator_action(
            action="maintenance_classify_modified_edges",
            pot_id=body.pot_id,
            actor=actor,
            dry_run=bool(body.dry_run),
            extra={"outcome": "ok" if result.get("ok", True) else "failed"},
        )
        return result

    return router


context_router = create_context_router(
    require_auth=require_api_key,
    get_container=get_container_or_503,
    get_db=get_db,
    get_db_optional=get_db_optional,
)

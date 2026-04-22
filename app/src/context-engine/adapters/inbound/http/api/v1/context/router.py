"""Context graph HTTP API (standalone service + host-injected dependencies)."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

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
from application.use_cases.backfill_pot import backfill_pot_context
from application.use_cases.event_reconciliation import ingestion_event_to_payload
from application.use_cases.hard_reset_pot import hard_reset_pot
from application.use_cases.run_raw_episode_ingestion import run_raw_episode_ingestion
from application.use_cases.replay_context_event import replay_context_event
from bootstrap.container import ContextEngineContainer
from domain.agent_context_port import (
    build_context_record_source_id,
    context_port_manifest,
    context_recipe_for_intent,
    normalize_record_type,
)
from domain.context_status import (
    DEFAULT_RESOLVER_CAPABILITIES,
    EventLedgerHealth,
    ResolverCapability,
    derive_pot_last_success_at,
    event_ledger_health_to_payload,
    resolver_capabilities_to_payload,
    status_source_to_payload,
)
from domain.graph_quality import assess_graph_quality
from domain.graph_query import ContextGraphQuery
from domain.ingestion_event_models import (
    EventListFilters,
    IngestionEventStatus,
    IngestionSubmissionRequest,
)
from domain.ingestion_kinds import (
    INGESTION_KIND_AGENT_RECONCILIATION,
    INGESTION_KIND_GITHUB_MERGED_PR,
)
from domain.intelligence_models import CoverageReport
from domain.source_references import SourceReferenceRecord
from domain.reconciliation_flags import agent_planner_enabled, reconciliation_enabled

UNKNOWN_POT_DETAIL = (
    "Unknown pot_id for this user (create with POST /api/v2/context/pots "
    "and attach at least one repository)."
)

_logger = logging.getLogger(__name__)

# Phase 6: event API clarification. ``/events/ingest`` is retained as a hidden
# compatibility alias of the canonical ``/events/reconcile`` path. Aliased
# requests emit deprecation headers and increment this counter so operators
# can track migration progress.
_EVENTS_INGEST_ALIAS_CALLS = 0
_EVENTS_INGEST_ALIAS_WARNING = (
    '299 - "POST /events/ingest is a deprecated alias of /events/reconcile; '
    'migrate clients to /events/reconcile."'
)


def events_ingest_alias_call_count() -> int:
    """Return the number of times the ``/events/ingest`` alias has been called.

    Exposed for tests and lightweight operational introspection.
    """

    return _EVENTS_INGEST_ALIAS_CALLS


def reset_events_ingest_alias_counter() -> None:
    """Reset the alias usage counter (tests only)."""

    global _EVENTS_INGEST_ALIAS_CALLS
    _EVENTS_INGEST_ALIAS_CALLS = 0


def _mark_events_ingest_alias(request: Request) -> bool:
    """Return ``True`` if the current request hit the deprecated alias path.

    Also increments the alias counter and logs a single warning per call.
    """

    try:
        path = request.url.path
    except Exception:
        return False
    if not path.endswith("/events/ingest"):
        return False
    global _EVENTS_INGEST_ALIAS_CALLS
    _EVENTS_INGEST_ALIAS_CALLS += 1
    _logger.warning(
        "context-engine: deprecated alias POST /events/ingest used; "
        "clients should migrate to POST /events/reconcile"
    )
    return True


def _apply_events_ingest_alias_headers(response: Response) -> None:
    """Attach RFC-style deprecation headers to a response for the alias path."""

    response.headers["Deprecation"] = "true"
    response.headers["Warning"] = _EVENTS_INGEST_ALIAS_WARNING
    response.headers["Link"] = '</api/v2/context/events/reconcile>; rel="successor-version"'


# Phase 7: operator surface hardening. Operator/admin routes (``/reset``,
# ``/conflicts/*``, ``/maintenance/*``) are grouped under a distinct OpenAPI
# tag so product docs and SDKs do not conflate them with everyday agent
# flows. Every destructive or graph-mutating call goes through
# ``_audit_operator_action`` so there is a single structured log line per
# action that operators and on-call can scrape.
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
    """Emit a structured audit log for a destructive or admin operator action.

    This is deliberately a log-first implementation: it gives operators a
    single scrape-friendly line per action (``extra`` fields attached via the
    logging ``extra=`` mechanism) without coupling to a metrics or event-store
    backend. Hosts can attach a richer handler if they want to persist audit
    rows.
    """

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


class SyncRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pot_ids: Optional[list[str]] = Field(
        default=None,
        description="If set, only these pot IDs (must exist in CONTEXT_ENGINE_POTS).",
    )


class IngestPrRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pot_id: str
    pr_number: int
    is_live_bridge: bool = True
    repo_name: Optional[str] = Field(
        default=None,
        description="GitHub owner/repo when the pot has multiple repository sources.",
    )


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
        description="Usually ``agent_reconciliation`` (default).",
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


class ReplayContextEventBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    event_id: str = Field(description="Existing ``context_events.id`` row.")


def _wants_sync(sync: bool, x_sync: str | None) -> bool:
    if sync:
        return True
    if x_sync and x_sync.strip().lower() in ("true", "1", "yes"):
        return True
    return False


class ContextMutationHandlers(Protocol):
    """Host-provided mutations (e.g. job queue enqueue) instead of inline use cases."""

    def handle_sync(
        self,
        payload: Optional[SyncRequest],
        container: ContextEngineContainer,
        db: Session,
        *,
        auth_user: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def handle_ingest_pr(
        self,
        body: IngestPrRequest,
        container: ContextEngineContainer,
        db: Session,
        *,
        auth_user: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


def _require_pot_access(container: ContextEngineContainer, pot_id: str) -> None:
    if container.pots.resolve_pot(pot_id) is None:
        raise HTTPException(
            status_code=404,
            detail=UNKNOWN_POT_DETAIL,
        )


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


def _inline_sync(
    payload: Optional[SyncRequest],
    container: ContextEngineContainer,
    db: Session,
) -> dict[str, Any]:
    if not container.settings.is_enabled():
        raise HTTPException(
            status_code=503,
            detail="Context graph is disabled (opt in by unsetting CONTEXT_GRAPH_ENABLED or setting true).",
        )
    mapping = payload.pot_ids if payload and payload.pot_ids else None
    results: list[dict[str, Any]] = []
    pots = container.pots
    if not hasattr(pots, "known_pot_ids"):
        raise HTTPException(
            status_code=500,
            detail="Pot resolution does not support listing IDs",
        )
    ids = mapping or pots.known_pot_ids()  # type: ignore[union-attr]
    for pid in ids:
        resolved = container.pots.resolve_pot(pid)
        if not resolved or not resolved.repos:
            results.append(
                {
                    "status": "skipped",
                    "pot_id": pid,
                    "reason": "unknown_pot_id",
                }
            )
            continue
        if container.context_graph is None:
            results.append(
                {
                    "status": "error",
                    "pot_id": pid,
                    "reason": "context_graph_unavailable",
                }
            )
            continue
        out = backfill_pot_context(
            settings=container.settings,
            pots=container.pots,
            source_for_repo=container.source_for_repo,
            ledger=container.ledger(db),
            context_graph=container.context_graph,
            pot_id=pid,
        )
        results.append(out)
    return {"status": "success", "results": results}


def _inline_ingest_pr(
    body: IngestPrRequest,
    container: ContextEngineContainer,
    db: Session,
) -> dict[str, Any]:
    if not container.settings.is_enabled():
        raise HTTPException(status_code=503, detail="Context graph is not enabled.")
    resolved = container.pots.resolve_pot(body.pot_id)
    if not resolved or not resolved.repos:
        raise HTTPException(
            status_code=404,
            detail=UNKNOWN_POT_DETAIL,
        )
    svc = container.ingestion_submission(db)
    req = IngestionSubmissionRequest(
        pot_id=body.pot_id,
        ingestion_kind=INGESTION_KIND_GITHUB_MERGED_PR,
        source_channel="http",
        source_system="github",
        event_type="pull_request",
        action="merged",
        payload={
            "pr_number": body.pr_number,
            "is_live_bridge": body.is_live_bridge,
            "repo_name": body.repo_name,
        },
        repo_name=body.repo_name,
    )
    try:
        receipt = svc.submit(req, sync=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if receipt.duplicate:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "duplicate_event",
                "event_id": receipt.event_id,
                "message": "Merged PR already recorded for this pot and PR number.",
            },
        )
    if receipt.status == "error":
        raise HTTPException(
            status_code=502,
            detail=receipt.error or "ingest_failed",
        )
    out = dict(receipt.extras or {})
    out["event_id"] = receipt.event_id
    return out


def create_context_router(
    *,
    require_auth: Callable[..., Any],
    get_container: Callable[..., ContextEngineContainer],
    get_db: Callable[..., Any],
    get_db_optional: Callable[..., Any],
    mutation_handlers: ContextMutationHandlers | None = None,
    enforce_pot_access: bool = False,
) -> APIRouter:
    """Build context API routes with injected FastAPI dependencies."""

    router = APIRouter()

    @router.post("/sync")
    def post_sync(
        payload: Optional[SyncRequest] = None,
        auth_user: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
        db: Session = Depends(get_db),
    ):
        if mutation_handlers is not None:
            return mutation_handlers.handle_sync(
                payload, container, db, auth_user=auth_user
            )
        return _inline_sync(payload, container, db)

    @router.post("/ingest-pr")
    def post_ingest_pr(
        body: IngestPrRequest,
        auth_user: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
        db: Session = Depends(get_db),
    ):
        if mutation_handlers is not None:
            return mutation_handlers.handle_ingest_pr(
                body, container, db, auth_user=auth_user
            )
        return _inline_ingest_pr(body, container, db)

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
        _: Any = Depends(require_auth),
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
        try:
            result = run_raw_episode_ingestion(
                container=container,
                db=db,
                pot_id=body.pot_id,
                name=body.name,
                episode_body=body.episode_body,
                source_description=body.source_description,
                reference_time=body.reference_time,
                idempotency_key=body.idempotency_key,
                sync=want_sync,
                source_channel="http",
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
                raise HTTPException(
                    status_code=404,
                    detail=UNKNOWN_POT_DETAIL,
                )
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
            raise HTTPException(
                status_code=503,
                detail=err,
            )

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
        if not container.settings.is_enabled():
            raise HTTPException(
                status_code=503,
                detail="Context graph is disabled (opt in by unsetting CONTEXT_GRAPH_ENABLED or setting true).",
            )
        if enforce_pot_access:
            _require_pot_access(container, body.pot_id)
        elif container.pots.resolve_pot(body.pot_id) is None:
            raise HTTPException(
                status_code=404,
                detail=UNKNOWN_POT_DETAIL,
            )
        use_ledger = db is not None and not body.skip_ledger
        ledger = SqlAlchemyIngestionLedger(db) if use_ledger else None
        reconciliation_ledger = (
            SqlAlchemyReconciliationLedger(db) if use_ledger else None
        )
        if container.context_graph is None:
            raise HTTPException(status_code=503, detail="context_graph_unavailable")
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
                status_code=502,
                detail=out.get("error") or "reset_failed",
            )
        _audit_operator_action(
            action="hard_reset_pot",
            pot_id=body.pot_id,
            actor=actor,
            dry_run=False,
            extra={
                "skip_ledger": bool(body.skip_ledger),
                "outcome": "ok",
            },
        )
        return out

    @router.post(
        "/events/reconcile",
        summary="Record event and run ingestion agent (reconciliation)",
        description=(
            "Canonical event submission path. Persists a row in ``context_events`` "
            "(deduped by scope + source_system + source_id), then runs the configured "
            "ingestion agent and applies the plan (async by default). New clients must "
            "use this route; ``/events/ingest`` is a deprecated alias retained for "
            "backwards compatibility."
        ),
    )
    @router.post(
        "/events/ingest",
        summary="Deprecated alias of /events/reconcile (compatibility only)",
        description=(
            "Compatibility alias of ``/events/reconcile``. Responses include "
            "``Deprecation: true`` plus a ``Warning`` header. New clients must use "
            "``/events/reconcile``."
        ),
        include_in_schema=False,
        deprecated=True,
    )
    def post_events_reconcile(
        body: ContextEventHttpBody,
        request: Request,
        response: Response,
        _: Any = Depends(require_auth),
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
        is_alias = _mark_events_ingest_alias(request)
        if is_alias:
            _apply_events_ingest_alias_headers(response)
        if not reconciliation_enabled():
            raise HTTPException(
                status_code=503,
                detail="Reconciliation is disabled (CONTEXT_ENGINE_RECONCILIATION_ENABLED).",
            )
        if not agent_planner_enabled():
            raise HTTPException(
                status_code=503,
                detail="Agent planner is disabled (CONTEXT_ENGINE_AGENT_PLANNER_ENABLED).",
            )
        if container.reconciliation_agent is None:
            raise HTTPException(
                status_code=503,
                detail="No reconciliation agent on the container; install context-engine[reconciliation-agent] "
                "and enable CONTEXT_ENGINE_AGENT_PLANNER_ENABLED.",
            )
        if not container.settings.is_enabled():
            raise HTTPException(
                status_code=503,
                detail="Context graph is disabled (CONTEXT_GRAPH_ENABLED).",
            )
        if enforce_pot_access:
            _require_pot_access(container, body.pot_id)
        elif container.pots.resolve_pot(body.pot_id) is None:
            raise HTTPException(
                status_code=404,
                detail=UNKNOWN_POT_DETAIL,
            )
        want_sync = _wants_sync(sync, x_context_ingest_sync)
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
            out = svc.submit(req, sync=want_sync)
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
        if not want_sync:
            queued = JSONResponse(
                status_code=202,
                content={
                    "status": "queued",
                    "event_id": out.event_id,
                    "job_id": out.job_id,
                },
            )
            if is_alias:
                _apply_events_ingest_alias_headers(queued)
            return queued
        r = out.reconciliation
        assert r is not None
        return {
            "event_id": out.event_id,
            "ok": r.ok,
            "episode_uuids": r.episode_uuids,
            "mutation_summary": {
                "episodes_written": r.mutation_summary.episodes_written,
                "entity_upserts_applied": r.mutation_summary.entity_upserts_applied,
                "edge_upserts_applied": r.mutation_summary.edge_upserts_applied,
                "edge_deletes_applied": r.mutation_summary.edge_deletes_applied,
                "invalidations_applied": r.mutation_summary.invalidations_applied,
                "stamp_counts": r.mutation_summary.stamp_counts,
            },
            "error": r.error,
            "downgrades": list(r.downgrades or []),
            "reconciliation_errors": list(r.reconciliation_errors or []),
        }

    @router.post(
        "/events/replay",
        summary="Retry reconciliation for an existing event",
    )
    def post_events_replay(
        body: ReplayContextEventBody,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        if not reconciliation_enabled():
            raise HTTPException(
                status_code=503,
                detail="Reconciliation is disabled (CONTEXT_ENGINE_RECONCILIATION_ENABLED).",
            )
        if not agent_planner_enabled():
            raise HTTPException(
                status_code=503,
                detail="Agent planner is disabled (CONTEXT_ENGINE_AGENT_PLANNER_ENABLED).",
            )
        if container.reconciliation_agent is None:
            raise HTTPException(
                status_code=503,
                detail="No reconciliation agent on the container.",
            )
        if not container.settings.is_enabled():
            raise HTTPException(
                status_code=503,
                detail="Context graph is disabled (CONTEXT_GRAPH_ENABLED).",
            )
        row = SqlAlchemyReconciliationLedger(db).get_event_by_id(body.event_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Unknown event_id")
        if enforce_pot_access:
            _require_pot_access(container, row.pot_id)
        elif container.pots.resolve_pot(row.pot_id) is None:
            raise HTTPException(
                status_code=404,
                detail=UNKNOWN_POT_DETAIL,
            )
        reco = SqlAlchemyReconciliationLedger(db)
        if container.context_graph is None:
            raise HTTPException(status_code=503, detail="Context graph unavailable.")
        try:
            r = replay_context_event(
                container.context_graph,
                container.reconciliation_agent,
                reco,
                body.event_id,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "event_id": body.event_id,
            "ok": r.ok,
            "episode_uuids": r.episode_uuids,
            "mutation_summary": {
                "episodes_written": r.mutation_summary.episodes_written,
                "entity_upserts_applied": r.mutation_summary.entity_upserts_applied,
                "edge_upserts_applied": r.mutation_summary.edge_upserts_applied,
                "edge_deletes_applied": r.mutation_summary.edge_deletes_applied,
                "invalidations_applied": r.mutation_summary.invalidations_applied,
                "stamp_counts": r.mutation_summary.stamp_counts,
            },
            "error": r.error,
            "downgrades": list(r.downgrades or []),
            "reconciliation_errors": list(r.reconciliation_errors or []),
        }

    @router.get("/events/{event_id}", summary="Fetch a persisted context event")
    def get_context_event(
        event_id: str,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        q = container.event_query_service(db)
        ev = q.get_event(event_id)
        if ev is None:
            raise HTTPException(status_code=404, detail="Unknown event_id")
        if enforce_pot_access:
            _require_pot_access(container, ev.pot_id)
        elif container.pots.resolve_pot(ev.pot_id) is None:
            raise HTTPException(
                status_code=404,
                detail=UNKNOWN_POT_DETAIL,
            )
        reco = SqlAlchemyReconciliationLedger(db)
        steps = reco.list_episode_steps(event_id)
        step_payload = [
            {
                "sequence": s.sequence,
                "step_kind": s.step_kind,
                "status": s.status,
                "attempt_count": s.attempt_count,
                "applied_at": s.applied_at.isoformat() if s.applied_at else None,
                "error": s.error,
            }
            for s in steps
        ]
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
        out = ingestion_event_to_payload(ev, episode_steps=step_payload)
        out["reconciliation_runs"] = run_payload
        return out

    @router.get(
        "/pots/{pot_id}/events",
        summary="List ingestion events for a pot (latest first)",
    )
    def list_pot_events(
        pot_id: str,
        _: Any = Depends(require_auth),
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
        from_date: datetime | None = Query(
            None,
            description="Only return events submitted at or after this ISO 8601 datetime.",
        ),
        to_date: datetime | None = Query(
            None,
            description="Only return events submitted at or before this ISO 8601 datetime.",
        ),
    ) -> dict[str, Any]:
        if enforce_pot_access:
            _require_pot_access(container, pot_id)
        elif container.pots.resolve_pot(pot_id) is None:
            raise HTTPException(
                status_code=404,
                detail=UNKNOWN_POT_DETAIL,
            )
        filters = EventListFilters(
            statuses=_parse_event_status_filters(status),
            ingestion_kinds=tuple(ingestion_kind) if ingestion_kind else None,
            source_systems=tuple(source_system) if source_system else None,
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
        _: Any = Depends(require_auth),
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
        if not reconciliation_enabled():
            raise HTTPException(
                status_code=503,
                detail="Reconciliation is disabled (CONTEXT_ENGINE_RECONCILIATION_ENABLED).",
            )
        if not agent_planner_enabled():
            raise HTTPException(
                status_code=503,
                detail="Agent planner is disabled (CONTEXT_ENGINE_AGENT_PLANNER_ENABLED).",
            )
        if container.reconciliation_agent is None:
            raise HTTPException(
                status_code=503,
                detail="No reconciliation agent on the container.",
            )
        if not container.settings.is_enabled():
            raise HTTPException(status_code=503, detail="Context graph is disabled.")
        if enforce_pot_access:
            _require_pot_access(container, body.pot_id)
        elif container.pots.resolve_pot(body.pot_id) is None:
            raise HTTPException(status_code=404, detail=UNKNOWN_POT_DETAIL)

        try:
            record_type = normalize_record_type(body.record.type)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        scope_payload = body.scope.model_dump(exclude_none=True)
        source_refs = list(
            dict.fromkeys(body.record.source_refs + body.scope.source_refs)
        )
        source_id = build_context_record_source_id(
            record_type=record_type,
            summary=body.record.summary,
            scope=scope_payload,
            source_refs=source_refs,
            idempotency_key=body.idempotency_key,
        )
        req = IngestionSubmissionRequest(
            pot_id=body.pot_id,
            ingestion_kind=INGESTION_KIND_AGENT_RECONCILIATION,
            source_channel="http",
            source_system="agent",
            event_type="context_record",
            action=record_type,
            source_id=source_id,
            repo_name=body.scope.repo_name,
            artifact_refs=tuple(source_refs),
            occurred_at=body.occurred_at,
            idempotency_key=body.idempotency_key,
            payload={
                "record": {
                    "type": record_type,
                    "summary": body.record.summary,
                    "details": dict(body.record.details),
                    "source_refs": source_refs,
                    "confidence": body.record.confidence,
                    "visibility": body.record.visibility,
                },
                "scope": scope_payload,
            },
        )
        try:
            receipt = container.ingestion_submission(db).submit(
                req,
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
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
        db: Session | None = Depends(get_db_optional),
    ) -> dict[str, Any]:
        resolved = container.pots.resolve_pot(body.pot_id)
        if resolved is None:
            raise HTTPException(status_code=404, detail=UNKNOWN_POT_DETAIL)
        if enforce_pot_access:
            _require_pot_access(container, body.pot_id)

        gaps: list[dict[str, str]] = []
        if not container.settings.is_enabled():
            gaps.append(
                {
                    "code": "context_graph_disabled",
                    "message": "Context graph is disabled for this server.",
                }
            )
        if container.resolution_service is None:
            gaps.append(
                {
                    "code": "resolver_unavailable",
                    "message": "Context resolution service is not configured.",
                }
            )
        if not resolved.repos:
            gaps.append(
                {
                    "code": "no_repositories",
                    "message": "This pot has no attached repositories.",
                }
            )
        if not container.episodic.enabled:
            gaps.append(
                {
                    "code": "episodic_graph_unavailable",
                    "message": "Graphiti episodic search is not enabled.",
                }
            )

        sources = []
        if container.pot_source_listing is not None:
            try:
                sources = container.pot_source_listing.list_pot_sources(body.pot_id)
            except Exception:
                sources = []

        ledger_health = EventLedgerHealth()
        if db is not None:
            try:
                ledger_health = container.event_query_service(db).summarize_pot_events(
                    body.pot_id
                )
            except Exception:
                ledger_health = EventLedgerHealth()

        if ledger_health.counts.get("error"):
            gaps.append(
                {
                    "code": "event_errors_present",
                    "message": (
                        f"{ledger_health.counts['error']} event(s) currently in error state; "
                        "review recent_errors and consider replay."
                    ),
                }
            )
        for s in sources:
            if not s.sync_enabled:
                continue
            if s.last_error:
                gaps.append(
                    {
                        "code": "source_sync_error",
                        "message": (
                            f"Source {s.source_kind}:{s.source_id} reported an error: "
                            f"{s.last_error[:200]}"
                        ),
                    }
                )

        if container.resolution_service is None:
            capabilities = [
                ResolverCapability(
                    policy=c.policy,
                    available=False,
                    reason="Context resolution service is not configured.",
                )
                for c in DEFAULT_RESOLVER_CAPABILITIES
            ]
        else:
            advertised_policies: set[str] = set()
            if container.source_resolver is not None:
                for entry in container.source_resolver.capabilities():
                    advertised_policies.update(entry.policies)
            capabilities = []
            for c in DEFAULT_RESOLVER_CAPABILITIES:
                if c.policy in advertised_policies and not c.available:
                    capabilities.append(
                        ResolverCapability(
                            policy=c.policy, available=True, reason=None
                        )
                    )
                else:
                    capabilities.append(c)

        recommended_recipe = context_recipe_for_intent(body.intent)
        coverage = CoverageReport(
            status="partial" if gaps else "complete",
            available=["pot", "repositories"]
            + (["sources"] if sources else [])
            + (["event_ledger"] if db is not None else []),
            missing=[gap["code"] for gap in gaps],
            missing_reasons={gap["code"]: gap["message"] for gap in gaps},
        )
        source_ref_records = [
            SourceReferenceRecord(
                ref=ref,
                source_type=ref.split(":", 1)[0] if ":" in ref else "unknown",
                external_id=ref.split(":", 1)[1] if ":" in ref else ref,
                freshness="needs_verification",
                sync_status="needs_resync",
                verification_state="unverified",
            )
            for ref in body.scope.source_refs
        ]
        quality = assess_graph_quality(
            refs=source_ref_records,
            coverage=coverage,
            fallbacks=[],
        )
        open_conflicts: list[dict[str, Any]] = []
        if container.episodic.enabled:
            try:
                open_conflicts = container.episodic.list_open_conflicts(resolved.pot_id)
            except Exception:
                open_conflicts = []
        quality.conflicts = open_conflicts
        reco: list[dict[str, Any]] = []
        if not gaps:
            reco = [
                {
                    "action": "resolve",
                    "intent": recommended_recipe["intent"],
                    "include": recommended_recipe["include"],
                    "mode": recommended_recipe["mode"],
                    "source_policy": recommended_recipe["source_policy"],
                    "reason": "Gather a bounded context wrap for the task scope.",
                }
            ]
            if open_conflicts and any(
                isinstance(x, dict) and x.get("auto_resolvable") is False
                for x in open_conflicts
            ):
                reco.append(
                    {
                        "action": "review_conflicts",
                        "reason": (
                            "Predicate-family conflicts are open; verify or resolve before "
                            "trusting contradictory facts."
                        ),
                    }
                )
        last_pot_success_at = derive_pot_last_success_at(sources, ledger_health)
        last_verifications = [
            s.last_verified_at.isoformat()
            for s in sources
            if s.last_verified_at is not None
        ]
        last_source_verification = max(last_verifications) if last_verifications else None
        return {
            "ok": not gaps,
            "pot": {
                "id": resolved.pot_id,
                "name": resolved.name,
                "ready": resolved.ready,
                "repos": [
                    {
                        "repo_name": repo.repo_name,
                        "provider": repo.provider,
                        "provider_host": repo.provider_host,
                        "default_branch": repo.default_branch,
                        "ready": repo.ready,
                    }
                    for repo in resolved.repos
                ],
                "last_success_at": (
                    last_pot_success_at.isoformat() if last_pot_success_at else None
                ),
            },
            "sources": [
                status_source_to_payload(s, resolver=container.source_resolver)
                for s in sources
            ],
            "event_ledger": event_ledger_health_to_payload(ledger_health),
            "resolver_capabilities": resolver_capabilities_to_payload(capabilities),
            "scope": body.scope.model_dump(exclude_none=True),
            "coverage": asdict(coverage),
            "freshness": {
                "status": "unknown" if last_pot_success_at is None else "known",
                "last_graph_update": (
                    last_pot_success_at.isoformat() if last_pot_success_at else None
                ),
                # last_source_event_at is distinct from last_graph_update: it
                # answers "when did the source actually produce this?" vs
                # "when did we last ingest". No sources attached yet → unknown.
                "last_source_event_at": None,
                "last_source_verification": last_source_verification,
                "stale_refs": [],
                "needs_verification_refs": body.scope.source_refs,
            },
            "source_refs": body.scope.source_refs,
            "quality": asdict(quality),
            "agent_port": context_port_manifest(),
            "recommended_recipe": recommended_recipe,
            "fallbacks": gaps,
            "open_conflicts": open_conflicts,
            "recommended_next_actions": reco,
        }

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
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ) -> dict[str, Any]:
        if enforce_pot_access:
            _require_pot_access(container, body.pot_id)
        if container.context_graph is None:
            raise HTTPException(
                status_code=503,
                detail="Unified context graph query port is not configured.",
            )
        result = await container.context_graph.query_async(body)
        return result.model_dump()

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
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ) -> dict[str, Any]:
        resolved = container.pots.resolve_pot(body.pot_id)
        if resolved is None:
            raise HTTPException(status_code=404, detail=UNKNOWN_POT_DETAIL)
        if enforce_pot_access:
            _require_pot_access(container, body.pot_id)
        if not container.episodic.enabled:
            return {"ok": False, "items": [], "error": "episodic_graph_unavailable"}
        try:
            items = container.episodic.list_open_conflicts(resolved.pot_id)
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
        resolved = container.pots.resolve_pot(body.pot_id)
        if resolved is None:
            raise HTTPException(status_code=404, detail=UNKNOWN_POT_DETAIL)
        if enforce_pot_access:
            _require_pot_access(container, body.pot_id)
        if not container.episodic.enabled:
            raise HTTPException(
                status_code=503, detail="episodic_graph_unavailable"
            )
        try:
            out = container.episodic.resolve_open_conflict(
                resolved.pot_id, body.issue_uuid, body.action
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
        if not container.settings.is_enabled():
            raise HTTPException(
                status_code=503,
                detail="Context graph is disabled (CONTEXT_GRAPH_ENABLED).",
            )
        if enforce_pot_access:
            _require_pot_access(container, body.pot_id)
        elif container.pots.resolve_pot(body.pot_id) is None:
            raise HTTPException(
                status_code=404,
                detail=UNKNOWN_POT_DETAIL,
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

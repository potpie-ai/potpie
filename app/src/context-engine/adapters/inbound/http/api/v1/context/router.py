"""Context graph HTTP API (standalone service + host-injected dependencies)."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from adapters.inbound.http.deps import (
    get_container_or_503,
    get_db,
    get_db_optional,
    require_api_key,
)
from adapters.outbound.postgres.ledger import SqlAlchemyIngestionLedger
from adapters.outbound.postgres.reconciliation_ledger import (
    SqlAlchemyReconciliationLedger,
)
from application.use_cases.backfill_pot import backfill_pot_context
from application.use_cases.event_reconciliation import ingestion_event_to_payload
from application.use_cases.hard_reset_pot import hard_reset_pot
from application.use_cases.run_raw_episode_ingestion import run_raw_episode_ingestion
from application.use_cases.query_context import (
    get_change_history,
    get_decisions,
    get_file_owners,
    get_pr_diff,
    get_pr_review_context,
    get_project_graph,
    search_pot_context,
)
from application.use_cases.replay_context_event import replay_context_event
from application.use_cases.resolve_context import resolve_context
from bootstrap.container import ContextEngineContainer
from domain.agent_context_port import (
    build_context_record_source_id,
    bundle_to_agent_envelope,
    context_port_manifest,
    context_recipe_for_intent,
    normalize_record_type,
)
from domain.graph_quality import assess_graph_quality
from domain.ingestion_event_models import (
    EventListFilters,
    IngestionEventStatus,
    IngestionSubmissionRequest,
)
from domain.ingestion_kinds import (
    INGESTION_KIND_AGENT_RECONCILIATION,
    INGESTION_KIND_GITHUB_MERGED_PR,
)
from domain.intelligence_models import (
    ArtifactRef,
    ContextBudget,
    ContextResolutionRequest,
    ContextScope,
    CoverageReport,
)
from domain.source_references import SourceReferenceRecord
from domain.reconciliation_flags import agent_planner_enabled, reconciliation_enabled

UNKNOWN_POT_DETAIL = (
    "Unknown pot_id for this user (create with POST /api/v2/context/pots "
    "and attach at least one repository)."
)


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


class ChangeHistoryQuery(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pot_id: str
    function_name: Optional[str] = None
    file_path: Optional[str] = None
    limit: int = 10
    repo_name: Optional[str] = None
    as_of: Optional[datetime] = Field(
        default=None,
        description="Return only changes that were active at this point in time (ISO 8601).",
    )


class FileOwnersQuery(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pot_id: str
    file_path: str
    limit: int = 5
    repo_name: Optional[str] = None


class DecisionsQuery(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pot_id: str
    file_path: Optional[str] = None
    function_name: Optional[str] = None
    limit: int = 20
    repo_name: Optional[str] = None


class PrReviewContextQuery(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pot_id: str
    pr_number: int = Field(ge=1, description="GitHub pull request number")
    repo_name: Optional[str] = None


class PrDiffQuery(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pot_id: str
    pr_number: int = Field(ge=1, description="GitHub pull request number")
    file_path: Optional[str] = None
    limit: int = 30
    repo_name: Optional[str] = None


class SearchQuery(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pot_id: str
    query: str
    limit: int = 8
    node_labels: Optional[list[str]] = None
    repo_name: Optional[str] = None
    source_description: Optional[str] = Field(
        default=None,
        description="Optional episodic source label filter (matches CLI --source).",
    )
    include_invalidated: bool = False
    as_of: Optional[datetime] = Field(
        default=None,
        description="Restrict to Graphiti edges valid at this instant (ISO 8601).",
    )


class ProjectGraphQuery(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pot_id: str
    repo_name: Optional[str] = None
    services: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    environment: Optional[str] = None
    user: Optional[str] = None
    include: list[str] = Field(default_factory=list)
    pr_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional pull request number to focus the graph on",
    )
    limit: int = 12


class ResolveArtifactBody(BaseModel):
    kind: str = Field(description="Artifact kind, e.g. pr, issue")
    identifier: str = Field(description="Artifact identifier, e.g. PR number")


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


class ResolveBudgetBody(BaseModel):
    max_items: int = Field(default=12, ge=1, le=50)
    max_tokens: Optional[int] = Field(default=None, ge=256, le=200000)
    timeout_ms: int = Field(default=4000, ge=500, le=30000)
    freshness: str = Field(default="prefer_fresh")


class ResolveContextRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pot_id: str
    query: str
    consumer_hint: Optional[str] = None
    artifact: Optional[ResolveArtifactBody] = None
    scope: Optional[ResolveScopeBody] = None
    intent: Optional[str] = None
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    mode: str = Field(default="fast")
    source_policy: str = Field(default="references_only")
    budget: ResolveBudgetBody = Field(default_factory=ResolveBudgetBody)
    as_of: Optional[datetime] = None
    timeout_ms: int = Field(default=4000, ge=500, le=30000)


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
        out = backfill_pot_context(
            settings=container.settings,
            pots=container.pots,
            source_for_repo=container.source_for_repo,
            ledger=container.ledger(db),
            episodic=container.episodic,
            structural=container.structural,
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
            err = result.error or "ingest_failed"
            if err == "unknown_pot_id":
                raise HTTPException(
                    status_code=404,
                    detail=UNKNOWN_POT_DETAIL,
                )
            if err == "async_requires_database":
                raise HTTPException(
                    status_code=503,
                    detail="Async raw ingest requires DATABASE_URL (or pass sync=true for inline Graphiti write).",
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

        if result.status == "legacy_direct":
            return {"episode_uuid": result.episode_uuid}
        if result.status == "applied":
            return {
                "episode_uuid": result.episode_uuid,
                "event_id": result.event_id,
                "job_id": result.job_id,
            }
        return JSONResponse(
            status_code=202,
            content={
                "status": "queued",
                "event_id": result.event_id,
                "job_id": result.job_id,
            },
        )

    @router.post(
        "/reset",
        summary="Hard-reset pot context graph",
        description=(
            "Deletes Postgres reconciliation/ingestion rows for the pot first (so async "
            "workers cannot re-apply after the graph is cleared), then Graphiti episodic data "
            "and structural Entity/FILE/NODE nodes in the default Neo4j database."
        ),
    )
    def post_hard_reset(
        body: HardResetRequest,
        _: Any = Depends(require_auth),
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
        out = hard_reset_pot(
            container.episodic,
            container.structural,
            body.pot_id,
            ledger=ledger,
            reconciliation_ledger=reconciliation_ledger,
        )
        if not out.get("ok"):
            raise HTTPException(
                status_code=502,
                detail=out.get("error") or "reset_failed",
            )
        return out

    @router.post(
        "/events/reconcile",
        summary="Record event and run ingestion agent (reconciliation)",
        description=(
            "Persists a canonical row in ``context_events`` (deduped by scope + source_system + source_id), "
            "then runs the configured ingestion agent and applies the plan (async by default)."
        ),
    )
    @router.post(
        "/events/ingest",
        summary="Alias of /events/reconcile (ingestion agent)",
        include_in_schema=False,
    )
    def post_events_reconcile(
        body: ContextEventHttpBody,
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
            return JSONResponse(
                status_code=202,
                content={
                    "status": "queued",
                    "event_id": out.event_id,
                    "job_id": out.job_id,
                },
            )
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
        try:
            r = replay_context_event(
                container.episodic,
                container.structural,
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
        return ingestion_event_to_payload(ev, episode_steps=step_payload)

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
        summary="Lightweight context readiness and freshness status",
    )
    def post_context_status(
        body: ContextStatusRequest,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
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

        recommended_recipe = context_recipe_for_intent(body.intent)
        coverage = CoverageReport(
            status="partial" if gaps else "complete",
            available=["pot", "repositories"],
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
            },
            "scope": body.scope.model_dump(exclude_none=True),
            "coverage": asdict(coverage),
            "freshness": {
                "status": "unknown",
                "last_graph_update": None,
                "last_source_verification": None,
                "stale_refs": [],
                "needs_verification_refs": body.scope.source_refs,
            },
            "source_refs": body.scope.source_refs,
            "quality": asdict(quality),
            "agent_port": context_port_manifest(),
            "recommended_recipe": recommended_recipe,
            "fallbacks": gaps,
            "recommended_next_actions": [
                {
                    "action": "resolve",
                    "intent": recommended_recipe["intent"],
                    "include": recommended_recipe["include"],
                    "mode": recommended_recipe["mode"],
                    "source_policy": recommended_recipe["source_policy"],
                    "reason": "Gather a bounded context wrap for the task scope.",
                }
            ]
            if not gaps
            else [],
        }

    @router.post("/query/change-history")
    def post_change_history(
        body: ChangeHistoryQuery,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ):
        if enforce_pot_access:
            _require_pot_access(container, body.pot_id)
        return get_change_history(
            container.structural,
            body.pot_id,
            function_name=body.function_name,
            file_path=body.file_path,
            limit=body.limit,
            repo_name=body.repo_name,
            as_of=body.as_of,
        )

    @router.post("/query/file-owners")
    def post_file_owners(
        body: FileOwnersQuery,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ):
        if enforce_pot_access:
            _require_pot_access(container, body.pot_id)
        return get_file_owners(
            container.structural,
            body.pot_id,
            body.file_path,
            body.limit,
            repo_name=body.repo_name,
        )

    @router.post("/query/decisions")
    def post_decisions(
        body: DecisionsQuery,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ):
        if enforce_pot_access:
            _require_pot_access(container, body.pot_id)
        return get_decisions(
            container.structural,
            body.pot_id,
            file_path=body.file_path,
            function_name=body.function_name,
            limit=body.limit,
            repo_name=body.repo_name,
        )

    @router.post("/query/pr-review-context")
    def post_pr_review_context(
        body: PrReviewContextQuery,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ):
        if enforce_pot_access:
            _require_pot_access(container, body.pot_id)
        return get_pr_review_context(
            container.structural, body.pot_id, body.pr_number, repo_name=body.repo_name
        )

    @router.post("/query/pr-diff")
    def post_pr_diff(
        body: PrDiffQuery,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ):
        if enforce_pot_access:
            _require_pot_access(container, body.pot_id)
        return get_pr_diff(
            container.structural,
            body.pot_id,
            body.pr_number,
            file_path=body.file_path,
            limit=body.limit,
            repo_name=body.repo_name,
        )

    @router.post(
        "/query/search",
        summary="Semantic search (Graphiti episodic)",
    )
    def post_search(
        body: SearchQuery,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ):
        if enforce_pot_access:
            _require_pot_access(container, body.pot_id)
        return search_pot_context(
            container.episodic,
            body.pot_id,
            body.query,
            limit=body.limit,
            node_labels=body.node_labels,
            repo_name=body.repo_name,
            source_description=body.source_description,
            include_invalidated=body.include_invalidated,
            as_of=body.as_of,
        )

    @router.post("/query/project-graph")
    def post_project_graph(
        body: ProjectGraphQuery,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ):
        if enforce_pot_access:
            _require_pot_access(container, body.pot_id)
        return get_project_graph(
            container.structural,
            body.pot_id,
            pr_number=body.pr_number,
            limit=body.limit,
            scope={
                "repo_name": body.repo_name,
                "services": body.services,
                "features": body.features,
                "environment": body.environment,
                "user": body.user,
            },
            include=body.include,
        )

    @router.post("/query/resolve-context")
    async def post_resolve_context(
        body: ResolveContextRequest,
        _: Any = Depends(require_auth),
        container: ContextEngineContainer = Depends(get_container),
    ):
        if not container.settings.is_enabled():
            raise HTTPException(
                status_code=503,
                detail="Context graph is not enabled. Set CONTEXT_GRAPH_ENABLED=true.",
            )
        if enforce_pot_access:
            _require_pot_access(container, body.pot_id)
        if container.resolution_service is None:
            raise HTTPException(
                status_code=503,
                detail="Context intelligence resolution is not available.",
            )
        art = (
            ArtifactRef(kind=body.artifact.kind, identifier=body.artifact.identifier)
            if body.artifact
            else None
        )
        scope = None
        if body.scope:
            scope = ContextScope(
                repo_name=body.scope.repo_name,
                branch=body.scope.branch,
                file_path=body.scope.file_path,
                function_name=body.scope.function_name,
                symbol=body.scope.symbol,
                pr_number=body.scope.pr_number,
                services=body.scope.services,
                features=body.scope.features,
                environment=body.scope.environment,
                ticket_ids=body.scope.ticket_ids,
                user=body.scope.user,
                source_refs=body.scope.source_refs,
            )
        req = ContextResolutionRequest(
            pot_id=body.pot_id,
            query=body.query,
            consumer_hint=body.consumer_hint,
            artifact_ref=art,
            scope=scope,
            intent=body.intent,
            include=body.include,
            exclude=body.exclude,
            mode=body.mode,
            source_policy=body.source_policy,
            budget=ContextBudget(
                max_items=body.budget.max_items,
                max_tokens=body.budget.max_tokens,
                timeout_ms=body.budget.timeout_ms,
                freshness=body.budget.freshness,
            ),
            as_of=body.as_of,
            timeout_ms=body.timeout_ms,
        )
        bundle = await resolve_context(container.resolution_service, req)
        return bundle_to_agent_envelope(bundle)

    return router


context_router = create_context_router(
    require_auth=require_api_key,
    get_container=get_container_or_503,
    get_db=get_db,
    get_db_optional=get_db_optional,
)

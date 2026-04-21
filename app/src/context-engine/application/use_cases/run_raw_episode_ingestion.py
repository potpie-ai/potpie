"""Single entry for raw episodic ingest: event store + queue or legacy direct Graphiti."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from sqlalchemy.orm import Session

from application.services.ingestion_submission_service import DefaultIngestionSubmissionService
from application.use_cases.ingest_episode import ingest_episode
from bootstrap.container import ContextEngineContainer
from domain.ingestion_event_models import EventReceipt, IngestionSubmissionRequest
from domain.ingestion_kinds import INGESTION_KIND_RAW_EPISODE
from domain.ports.graph_mutation_applier import GraphMutationApplierPort
from domain.reconciliation import ReconciliationResult


@dataclass(slots=True)
class RunRawEpisodeIngestionResult:
    """Outcome for CLI, HTTP, and MCP raw ingest."""

    ok: bool
    status: Literal[
        "applied",
        "queued",
        "legacy_direct",
        "duplicate",
        "error",
        "reconciliation_rejected",
    ]
    episode_uuid: str | None = None
    event_id: str | None = None
    job_id: str | None = None
    error: str | None = None
    duplicate_reason: str | None = None
    reconciliation_errors: list[dict[str, str]] | None = None
    downgrades: list[dict[str, str]] | None = None


def _downgrades_from_reconciliation(receipt: EventReceipt) -> list[dict[str, str]]:
    rec = receipt.reconciliation
    if isinstance(rec, ReconciliationResult) and rec.downgrades:
        return list(rec.downgrades)
    return []


def _receipt_to_run_result(receipt: EventReceipt) -> RunRawEpisodeIngestionResult:
    r = receipt
    if r.duplicate:
        return RunRawEpisodeIngestionResult(
            ok=False,
            status="duplicate",
            event_id=r.event_id,
            duplicate_reason=r.error,
        )
    if r.status == "error" or (r.error and not r.duplicate):
        rec = r.reconciliation
        if (
            isinstance(rec, ReconciliationResult)
            and (rec.reconciliation_errors or [])
        ):
            return RunRawEpisodeIngestionResult(
                ok=False,
                status="reconciliation_rejected",
                event_id=r.event_id,
                job_id=r.job_id,
                error=r.error,
                reconciliation_errors=list(rec.reconciliation_errors),
                downgrades=_downgrades_from_reconciliation(r),
            )
        return RunRawEpisodeIngestionResult(
            ok=False,
            status="error",
            event_id=r.event_id,
            job_id=r.job_id,
            error=r.error,
        )
    if r.status == "queued":
        return RunRawEpisodeIngestionResult(
            ok=True,
            status="queued",
            episode_uuid=r.episode_uuid,
            event_id=r.event_id,
            job_id=r.job_id,
            error=None,
            downgrades=_downgrades_from_reconciliation(r),
        )
    return RunRawEpisodeIngestionResult(
        ok=True,
        status="applied",
        episode_uuid=r.episode_uuid,
        event_id=r.event_id,
        job_id=r.job_id,
        error=None,
        downgrades=_downgrades_from_reconciliation(r),
    )


def run_raw_episode_ingestion(
    *,
    container: ContextEngineContainer,
    db: Session | None,
    pot_id: str,
    name: str,
    episode_body: str,
    source_description: str,
    reference_time: datetime,
    idempotency_key: str | None,
    sync: bool,
    mutation_applier: GraphMutationApplierPort | None = None,
    source_channel: str = "http",
) -> RunRawEpisodeIngestionResult:
    """
    Unified raw episode ingest.

    - With ``DATABASE_URL`` / ``db``: persist ``context_events`` + step, then
      ``sync=True`` applies inline; ``sync=False`` enqueues apply (or inline when
      job queue is no-op).
    - Without ``db`` and ``sync=True``: legacy direct ``ingest_episode`` (Graphiti only).
    - Without ``db`` and ``sync=False``: error (async requires Postgres).
    """
    if not container.settings.is_enabled():
        return RunRawEpisodeIngestionResult(
            ok=False,
            status="error",
            error="context_graph_disabled",
        )

    resolved = container.pots.resolve_pot(pot_id)
    if resolved is None:
        return RunRawEpisodeIngestionResult(
            ok=False,
            status="error",
            error="unknown_pot_id",
        )

    if db is None:
        if not sync:
            return RunRawEpisodeIngestionResult(
                ok=False,
                status="error",
                error="async_requires_database",
            )
        out = ingest_episode(
            container.episodic,
            pot_id,
            name,
            episode_body,
            source_description,
            reference_time,
        )
        uid = out.get("episode_uuid")
        if uid is None:
            return RunRawEpisodeIngestionResult(
                ok=False,
                status="error",
                error="graphiti_returned_no_uuid",
            )
        return RunRawEpisodeIngestionResult(
            ok=True,
            status="legacy_direct",
            episode_uuid=uid,
        )

    svc = DefaultIngestionSubmissionService(container, db, mutation_applier=mutation_applier)
    req = IngestionSubmissionRequest(
        pot_id=pot_id,
        ingestion_kind=INGESTION_KIND_RAW_EPISODE,
        source_channel=source_channel,
        source_system="context_engine_raw",
        event_type="episode",
        action="ingest",
        payload={
            "name": name,
            "episode_body": episode_body,
            "source_description": source_description,
            "reference_time": reference_time.isoformat(),
        },
        idempotency_key=idempotency_key,
        dedup_key=idempotency_key,
    )
    try:
        receipt = svc.submit(req, sync=sync)
    except ValueError as e:
        msg = str(e)
        if msg == "unknown_pot_id":
            return RunRawEpisodeIngestionResult(ok=False, status="error", error="unknown_pot_id")
        return RunRawEpisodeIngestionResult(ok=False, status="error", error=msg)
    return _receipt_to_run_result(receipt)

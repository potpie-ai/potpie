"""Submit a raw episode for ingestion.

Two paths:

- With a SQLAlchemy session, the episode is shaped into a
  :class:`IngestionSubmissionRequest` (kind ``raw_episode``) and admitted
  through the standard async batch pipeline.
- Without a session, the episode is written directly through
  :meth:`ContextGraphPort.write_raw_episode` — used by standalone CLI
  invocations and by tests that don't run a full Postgres stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from sqlalchemy.orm import Session

from bootstrap.container import ContextEngineContainer
from domain.actor import Actor
from domain.ingestion_event_models import EventReceipt, IngestionSubmissionRequest
from domain.ingestion_kinds import INGESTION_KIND_RAW_EPISODE
from domain.reconciliation import ReconciliationResult


@dataclass(slots=True)
class RawEpisodeSubmissionResult:
    """Outcome for CLI, HTTP, and MCP raw ingest."""

    ok: bool
    status: Literal[
        "applied",
        "queued",
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


def _receipt_to_run_result(receipt: EventReceipt) -> RawEpisodeSubmissionResult:
    r = receipt
    if r.duplicate:
        return RawEpisodeSubmissionResult(
            ok=False,
            status="duplicate",
            event_id=r.event_id,
            duplicate_reason=r.error,
        )
    if r.status == "error" or (r.error and not r.duplicate):
        rec = r.reconciliation
        if isinstance(rec, ReconciliationResult) and (rec.reconciliation_errors or []):
            return RawEpisodeSubmissionResult(
                ok=False,
                status="reconciliation_rejected",
                event_id=r.event_id,
                job_id=r.job_id,
                error=r.error,
                reconciliation_errors=list(rec.reconciliation_errors),
                downgrades=_downgrades_from_reconciliation(r),
            )
        return RawEpisodeSubmissionResult(
            ok=False,
            status="error",
            event_id=r.event_id,
            job_id=r.job_id,
            error=r.error,
        )
    if r.status == "queued":
        return RawEpisodeSubmissionResult(
            ok=True,
            status="queued",
            episode_uuid=r.episode_uuid,
            event_id=r.event_id,
            job_id=r.job_id,
            error=None,
            downgrades=_downgrades_from_reconciliation(r),
        )
    return RawEpisodeSubmissionResult(
        ok=True,
        status="applied",
        episode_uuid=r.episode_uuid,
        event_id=r.event_id,
        job_id=r.job_id,
        error=None,
        downgrades=_downgrades_from_reconciliation(r),
    )


def submit_raw_episode(
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
    source_channel: str = "http",
    actor: Actor | None = None,
) -> RawEpisodeSubmissionResult:
    """Apply or admit a raw Graphiti episode for ``pot_id``.

    Callers are responsible for authorization — HTTP routes call
    ``PolicyPort.authorize`` once before invoking this verb. The use case
    no longer re-checks ``settings.is_enabled()`` or ``pots.resolve_pot``.
    """
    if db is None:
        if not sync:
            return RawEpisodeSubmissionResult(
                ok=False, status="error", error="async_requires_database"
            )
        if container.context_graph is None:
            return RawEpisodeSubmissionResult(
                ok=False, status="error", error="context_graph_unavailable"
            )
        out = container.context_graph.write_raw_episode(
            pot_id,
            name,
            episode_body,
            source_description,
            reference_time,
            actor=actor,
        )
        uid = out.get("episode_uuid")
        if uid is None:
            return RawEpisodeSubmissionResult(
                ok=False, status="error", error="graphiti_returned_no_uuid"
            )
        return RawEpisodeSubmissionResult(ok=True, status="applied", episode_uuid=uid)

    svc = container.ingestion_submission(db)
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
        actor=actor,
    )
    try:
        receipt = svc.submit(req, sync=sync)
    except ValueError as e:
        msg = str(e)
        if msg == "unknown_pot_id":
            return RawEpisodeSubmissionResult(
                ok=False, status="error", error="unknown_pot_id"
            )
        return RawEpisodeSubmissionResult(ok=False, status="error", error=msg)
    return _receipt_to_run_result(receipt)

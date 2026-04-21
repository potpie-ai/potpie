"""Legacy no-agent raw episodic ingest helper."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from domain.context_events import ContextEvent, EventScope
from domain.ingestion_kinds import INGESTION_KIND_RAW_EPISODE, STEP_KIND_RAW_EPISODE
from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.graph_mutation_applier import GraphMutationApplierPort
from domain.ports.jobs import JobEnqueuePort, NoOpJobEnqueue
from domain.ports.reconciliation_ledger import ReconciliationLedgerPort
from domain.ports.structural_graph import StructuralGraphPort

from application.use_cases.apply_episode_step import apply_episode_step_for_event
from application.use_cases.record_context_event import record_context_event


@dataclass(slots=True)
class RawEpisodeIngestOutcome:
    inserted: bool
    event_id: str
    episode_uuid: str | None
    job_id: str | None
    error: str | None
    reconciliation_errors: list[dict[str, str]] | None = None


def record_raw_episode_ingestion(
    episodic: EpisodicGraphPort,
    structural: StructuralGraphPort,
    reco_ledger: ReconciliationLedgerPort,
    scope: EventScope,
    *,
    pot_id: str,
    name: str,
    episode_body: str,
    source_description: str,
    reference_time: datetime,
    idempotency_key: str | None,
    sync: bool,
    jobs: JobEnqueuePort | None,
    mutation_applier: GraphMutationApplierPort | None = None,
    source_channel: str | None = None,
) -> RawEpisodeIngestOutcome:
    """Persist a ``raw_episode`` event, durable step row, then sync apply or async queue.

    New persisted raw ingest routes through ``DefaultIngestionSubmissionService`` and the
    Ingestion Agent. This helper remains for legacy/direct fallback tests and adapters.
    """
    jq = jobs or NoOpJobEnqueue()
    ev = ContextEvent(
        event_id=str(uuid4()),
        source_system="context_engine_raw",
        event_type="episode",
        action="ingest",
        pot_id=pot_id,
        source_channel=source_channel,
        provider=scope.provider,
        provider_host=scope.provider_host,
        repo_name=scope.repo_name,
        source_id=f"raw_episode_{uuid4()}",
        payload={
            "name": name,
            "episode_body": episode_body,
            "source_description": source_description,
            "reference_time": reference_time.isoformat(),
        },
        occurred_at=reference_time,
        ingestion_kind=INGESTION_KIND_RAW_EPISODE,
        idempotency_key=idempotency_key,
    )
    event_id, inserted = record_context_event(reco_ledger, scope, ev)
    if not inserted:
        return RawEpisodeIngestOutcome(
            inserted=False,
            event_id=event_id,
            episode_uuid=None,
            job_id=None,
            error="duplicate_idempotency" if idempotency_key else "duplicate_event",
        )

    step_json: dict[str, Any] = {
        "version": 1,
        "name": name,
        "episode_body": episode_body,
        "source_description": source_description,
        "reference_time": reference_time.isoformat(),
    }
    reco_ledger.replace_episode_steps_for_event(
        pot_id,
        event_id,
        None,
        [(1, STEP_KIND_RAW_EPISODE, step_json)],
    )

    jid = str(uuid4())
    reco_ledger.set_event_job_metadata(event_id, job_id=jid, correlation_id=jid)

    if sync:
        r = apply_episode_step_for_event(
            episodic,
            structural,
            reco_ledger,
            event_id,
            1,
            mutation_applier=mutation_applier,
        )
        uid = r.episode_uuids[0] if r.episode_uuids else None
        return RawEpisodeIngestOutcome(
            inserted=True,
            event_id=event_id,
            episode_uuid=uid,
            job_id=jid,
            error=r.error,
            reconciliation_errors=list(r.reconciliation_errors)
            if r.reconciliation_errors
            else None,
        )

    reco_ledger.mark_event_queued(event_id)
    jq.enqueue_episode_apply(pot_id, event_id, 1)
    if isinstance(jq, NoOpJobEnqueue):
        r = apply_episode_step_for_event(
            episodic,
            structural,
            reco_ledger,
            event_id,
            1,
            mutation_applier=mutation_applier,
        )
        uid = r.episode_uuids[0] if r.episode_uuids else None
        return RawEpisodeIngestOutcome(
            inserted=True,
            event_id=event_id,
            episode_uuid=uid,
            job_id=jid,
            error=r.error,
            reconciliation_errors=list(r.reconciliation_errors)
            if r.reconciliation_errors
            else None,
        )

    return RawEpisodeIngestOutcome(
        inserted=True,
        event_id=event_id,
        episode_uuid=None,
        job_id=jid,
        error=None,
    )

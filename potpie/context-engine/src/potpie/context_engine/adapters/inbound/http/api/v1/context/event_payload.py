"""JSON serialization for ingestion events (read model / HTTP responses)."""

from __future__ import annotations

from typing import Any

from potpie.context_engine.domain.ingestion_event_models import IngestionEvent


def ingestion_event_to_payload(ev: IngestionEvent) -> dict[str, Any]:
    """JSON view of a stored ingestion event (used by EventQueryService + HTTP)."""
    legacy = ev.raw_status or ev.status
    return {
        "id": ev.event_id,
        "pot_id": ev.pot_id,
        "provider": ev.provider,
        "provider_host": ev.provider_host,
        "repo_name": ev.repo_name,
        "source_system": ev.source_system,
        "event_type": ev.event_type,
        "action": ev.action,
        "source_id": ev.source_id,
        "source_event_id": ev.source_event_id,
        "payload": ev.payload,
        "occurred_at": ev.occurred_at.isoformat() if ev.occurred_at else None,
        "received_at": ev.submitted_at.isoformat(),
        "status": legacy,
        "lifecycle_status": ev.status,
        "ingestion_kind": ev.ingestion_kind,
        "source_channel": ev.source_channel,
        "dedup_key": ev.dedup_key,
        "stage": ev.stage,
        "step_total": ev.step_total,
        "step_done": ev.step_done,
        "step_error": ev.step_error,
        "error": ev.error,
        "job_id": ev.job_id,
        "correlation_id": ev.correlation_id,
        "idempotency_key": ev.idempotency_key,
        "metadata": ev.metadata,
        "actor": ev.actor.to_payload() if ev.actor else None,
    }

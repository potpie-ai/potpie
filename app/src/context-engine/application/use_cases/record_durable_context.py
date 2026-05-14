"""Record a durable agent context entry — fix, decision, runbook note, etc.

Backed by the same async pipeline as every other event submission. The
verb shapes the inbound payload into an ``IngestionSubmissionRequest``
and hands it to the configured :class:`IngestionSubmissionService`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from domain.actor import Actor
from domain.agent_context_port import (
    build_context_record_source_id,
    normalize_record_type,
)
from domain.ingestion_event_models import EventReceipt, IngestionSubmissionRequest
from domain.ingestion_kinds import INGESTION_KIND_AGENT_RECONCILIATION
from domain.ports.ingestion_submission import IngestionSubmissionService


@dataclass(slots=True, frozen=True)
class DurableContextPayload:
    record_type: str
    summary: str
    details: Mapping[str, Any] = ()  # type: ignore[assignment]
    source_refs: tuple[str, ...] = ()
    confidence: float = 0.7
    visibility: str = "project"


def record_durable_context(
    submission: IngestionSubmissionService,
    *,
    pot_id: str,
    record: DurableContextPayload,
    scope: Mapping[str, Any],
    actor: Actor,
    idempotency_key: str | None = None,
    occurred_at: datetime | None = None,
    sync: bool = False,
) -> tuple[EventReceipt, str, str]:
    """Submit a durable record. Returns ``(receipt, record_type, source_id)``.

    Raises ``ValueError`` for unknown record types and any submission
    failure that the ingestion service surfaces (``unknown_pot_id``,
    ``context_graph_disabled``, …) — callers translate to transport.
    """
    record_type = normalize_record_type(record.record_type)
    scope_payload = {k: v for k, v in dict(scope).items() if v not in (None, "", [])}
    source_refs_in = list(record.source_refs) + list(
        scope_payload.get("source_refs") or []
    )
    source_refs = list(dict.fromkeys(source_refs_in))
    source_id = build_context_record_source_id(
        record_type=record_type,
        summary=record.summary,
        scope=scope_payload,
        source_refs=source_refs,
        idempotency_key=idempotency_key,
    )
    req = IngestionSubmissionRequest(
        pot_id=pot_id,
        ingestion_kind=INGESTION_KIND_AGENT_RECONCILIATION,
        source_channel=actor.surface,
        source_system="agent",
        event_type="context_record",
        action=record_type,
        source_id=source_id,
        repo_name=scope_payload.get("repo_name"),
        artifact_refs=tuple(source_refs),
        occurred_at=occurred_at,
        idempotency_key=idempotency_key,
        actor=actor,
        payload={
            "record": {
                "type": record_type,
                "summary": record.summary,
                "details": dict(record.details),
                "source_refs": source_refs,
                "confidence": record.confidence,
                "visibility": record.visibility,
            },
            "scope": scope_payload,
        },
    )
    receipt = submission.submit(req, sync=sync)
    return receipt, record_type, source_id

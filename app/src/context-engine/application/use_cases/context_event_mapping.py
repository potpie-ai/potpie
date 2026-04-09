"""Map persisted reconciliation rows to domain ``ContextEvent``."""

from __future__ import annotations

from domain.context_events import ContextEvent
from domain.ports.reconciliation_ledger import ContextEventRow


def context_event_row_to_domain(row: ContextEventRow) -> ContextEvent:
    """Rehydrate a domain ``ContextEvent`` from a persisted row."""
    return ContextEvent(
        event_id=row.id,
        source_system=row.source_system,
        event_type=row.event_type,
        action=row.action,
        pot_id=row.pot_id,
        provider=row.provider,
        provider_host=row.provider_host,
        repo_name=row.repo_name,
        source_id=row.source_id,
        source_event_id=row.source_event_id,
        artifact_refs=list(row.payload.get("artifact_refs") or []),
        occurred_at=row.occurred_at,
        received_at=row.received_at,
        payload=dict(row.payload),
        ingestion_kind=row.ingestion_kind,
        idempotency_key=row.idempotency_key,
    )

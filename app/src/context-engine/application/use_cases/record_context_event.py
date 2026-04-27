"""Persist a normalized ``ContextEvent`` for deduplication and lifecycle tracking."""

from __future__ import annotations

from domain.context_events import ContextEvent, EventScope
from domain.ports.reconciliation_ledger import ReconciliationLedgerPort


def record_context_event(
    reco_ledger: ReconciliationLedgerPort,
    scope: EventScope,
    event: ContextEvent,
) -> tuple[str, bool]:
    """Append event; returns ``(event_id, inserted)`` where inserted is False on duplicate."""
    return reco_ledger.append_event(scope, event)

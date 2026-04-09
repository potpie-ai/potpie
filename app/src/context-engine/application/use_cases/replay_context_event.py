"""Replay reconciliation for an existing persisted event."""

from __future__ import annotations

from domain.errors import ReconciliationApplyError
from domain.context_events import ContextEvent
from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.graph_mutation_applier import GraphMutationApplierPort
from domain.ports.reconciliation_agent import ReconciliationAgentPort
from domain.ports.reconciliation_ledger import ContextEventRow, ReconciliationLedgerPort
from domain.ports.structural_graph import StructuralGraphPort
from domain.reconciliation import ReconciliationResult

from application.use_cases.build_reconciliation_request import build_reconciliation_request
from application.use_cases.reconcile_event import reconcile_event


def _row_to_event(row: ContextEventRow) -> ContextEvent:
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
        payload=row.payload,
        occurred_at=row.occurred_at,
        received_at=row.received_at,
    )


def replay_context_event(
    episodic: EpisodicGraphPort,
    structural: StructuralGraphPort,
    agent: ReconciliationAgentPort,
    reco_ledger: ReconciliationLedgerPort,
    event_id: str,
    *,
    mutation_applier: GraphMutationApplierPort | None = None,
    reset_for_retry: bool = True,
) -> ReconciliationResult:
    """Load event by id, optionally mark for retry, then ``reconcile_event``."""
    row = reco_ledger.get_event_by_id(event_id)
    if row is None:
        raise ReconciliationApplyError(f"unknown context event: {event_id}")
    if reset_for_retry:
        reco_ledger.mark_event_for_retry(event_id)
    event = _row_to_event(row)
    request = build_reconciliation_request(event)
    return reconcile_event(
        episodic,
        structural,
        agent,
        request,
        reco_ledger=reco_ledger,
        mutation_applier=mutation_applier,
    )

"""Hard-reset all context-graph data for a single pot."""

from __future__ import annotations

from typing import Any, Optional

from domain.ports.context_graph import ContextGraphPort
from domain.ports.ingestion_ledger import IngestionLedgerPort
from domain.ports.reconciliation_ledger import ReconciliationLedgerPort


def hard_reset_pot(
    context_graph: ContextGraphPort,
    pot_id: str,
    *,
    ledger: Optional[IngestionLedgerPort] = None,
    reconciliation_ledger: Optional[ReconciliationLedgerPort] = None,
) -> dict[str, Any]:
    """Clear Postgres pipeline rows first, then reset the unified graph layer.

    Reconciliation / ingestion ledger rows are removed **before** Neo4j so async workers
    still holding ``event_id`` see ``unknown_event`` and cannot re-apply episodes after
    the graph was cleared.
    """
    out: dict[str, Any] = {"pot_id": pot_id, "ok": False}

    if reconciliation_ledger is not None:
        out["reconciliation_rows_deleted"] = reconciliation_ledger.delete_all_for_pot(pot_id)

    if ledger is not None:
        out["ledger_rows_deleted"] = ledger.delete_all_for_pot(pot_id)

    graph_out = context_graph.reset_pot(pot_id)
    out.update(graph_out)
    return out

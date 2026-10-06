"""Hard-reset all context-graph data for a single pot."""

from __future__ import annotations

from typing import Any, Optional

from potpie_context_engine.core.ports.resource_store import ResourceStorePort
from potpie_context_engine.domain.ports.context_graph import ContextGraphPort
from potpie_context_engine.domain.ports.ingestion_ledger import IngestionLedgerPort
from potpie_context_engine.domain.ports.reconciliation_ledger import (
    ReconciliationLedgerPort,
)


def hard_reset_pot(
    context_graph: ContextGraphPort,
    pot_id: str,
    *,
    ledger: Optional[IngestionLedgerPort] = None,
    reconciliation_ledger: Optional[ReconciliationLedgerPort] = None,
    resources: Optional[ResourceStorePort] = None,
) -> dict[str, Any]:
    """Clear Postgres pipeline rows first, then reset the graph and resources.

    Reconciliation / ingestion ledger rows are removed **before** Neo4j so async
    workers still holding ``event_id`` see ``unknown_event`` and cannot re-apply
    episodes after the graph was cleared. Resource bytes are purged **after**
    the graph succeeds so a failed graph reset cannot leave claims pointing at
    deleted chunk files (R8).
    """
    out: dict[str, Any] = {"pot_id": pot_id, "ok": False}

    # Refused before any ledger row goes: the graph reset would refuse a pot
    # whose journal is capturing, and stopping there would leave the ledgers
    # cleared and the graph intact.
    capturing = getattr(context_graph, "journal_capture_active", None)
    if callable(capturing) and capturing(pot_id) is True:
        out["error"] = "journal_capture_active"
        out["detail"] = (
            "hard reset is refused while graph journal capture is active for "
            "this pot; it would discard the pot's commit history"
        )
        return out

    if reconciliation_ledger is not None:
        out["reconciliation_rows_deleted"] = reconciliation_ledger.delete_all_for_pot(
            pot_id
        )

    if ledger is not None:
        out["ledger_rows_deleted"] = ledger.delete_all_for_pot(pot_id)

    graph_out = context_graph.reset_pot(pot_id)
    out.update(graph_out)
    if out.get("ok") and resources is not None:
        out["resources_purged"] = bool(resources.purge_pot(pot_id))
    return out

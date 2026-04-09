"""Hard-reset all context-graph data for a single pot (Neo4j + optional Postgres ledger)."""

from __future__ import annotations

from typing import Any, Optional

from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.ingestion_ledger import IngestionLedgerPort
from domain.ports.reconciliation_ledger import ReconciliationLedgerPort
from domain.ports.structural_graph import StructuralGraphPort


def hard_reset_pot(
    episodic: EpisodicGraphPort,
    structural: StructuralGraphPort,
    pot_id: str,
    *,
    ledger: Optional[IngestionLedgerPort] = None,
    reconciliation_ledger: Optional[ReconciliationLedgerPort] = None,
) -> dict[str, Any]:
    """Clear Postgres pipeline rows first, then Graphiti episodic data, then structural nodes.

    Reconciliation / ingestion ledger rows are removed **before** Neo4j so async workers
    still holding ``event_id`` see ``unknown_event`` and cannot re-apply episodes after
    the graph was cleared.
    """
    out: dict[str, Any] = {"pot_id": pot_id, "ok": False}

    if reconciliation_ledger is not None:
        out["reconciliation_rows_deleted"] = reconciliation_ledger.delete_all_for_pot(pot_id)

    if ledger is not None:
        out["ledger_rows_deleted"] = ledger.delete_all_for_pot(pot_id)

    ep = episodic.reset_pot(pot_id)
    out["episodic"] = ep
    if not ep.get("ok"):
        out["error"] = ep.get("error", "episodic_reset_failed")
        return out

    st = structural.reset_pot(pot_id)
    out["structural"] = st
    if not st.get("ok"):
        out["error"] = st.get("error", "structural_reset_failed")
        return out

    out["ok"] = True
    return out

"""``EventReconcilerPort`` — normalized events → graph claims.

This is the **parked decision seam**. How a batch of normalized ledger events
becomes graph mutations — a deterministic extractor, an LLM reconciliation
agent, or a hybrid — is an explicitly *undecided* low-level choice. The
skeleton keeps it behind this port with a dummy implementation so the rest of
the architecture (ledger pull → reconcile → GraphService.record/mutation) does
not change when the decision is made.

    TODO(decide): LLM-vs-deterministic reconciliation strategy.
    The existing pydantic reconciliation agent and the deterministic
    extractors are both candidate implementations of this port.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from domain.ports.ledger.client import LedgerEvent


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    """Outcome of reconciling one batch of events into a pot's graph."""

    pot_id: str
    events_in: int
    claims_written: int = 0
    skipped: int = 0
    detail: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class EventReconcilerPort(Protocol):
    """Turn a batch of normalized events into graph mutations for a pot."""

    def reconcile(
        self, *, pot_id: str, events: Sequence[LedgerEvent]
    ) -> ReconcileResult:
        """Reconcile ``events`` into the pot's graph and report counts."""
        ...


__all__ = ["EventReconcilerPort", "ReconcileResult"]

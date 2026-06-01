"""Event Ledger ports — the external source-event seam.

The ledger is not graph storage. A graph pulls normalized events through
``EventLedgerClientPort``, tracks position with ``LedgerCursorStorePort``, and
turns events into claims through ``EventReconcilerPort`` (the parked
LLM-vs-deterministic decision).
"""

from __future__ import annotations

from domain.ports.ledger.client import (
    EventLedgerClientPort,
    LedgerCursor,
    LedgerEvent,
    LedgerHealth,
    LedgerPage,
    LedgerSource,
)
from domain.ports.ledger.cursor import LedgerCursorStorePort
from domain.ports.ledger.reconciler import EventReconcilerPort, ReconcileResult

__all__ = [
    "EventLedgerClientPort",
    "EventReconcilerPort",
    "LedgerCursor",
    "LedgerCursorStorePort",
    "LedgerEvent",
    "LedgerHealth",
    "LedgerPage",
    "LedgerSource",
    "ReconcileResult",
]

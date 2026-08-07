"""``LedgerCursorStorePort`` — durable per-source pull position.

Cursors live with the *graph* (the consumer), not the ledger, so the same
ledger can feed multiple graphs at independent positions and a graph can replay
from a known point. Stored alongside pot state.
"""

from __future__ import annotations

from typing import Protocol

from potpie_context_engine.domain.ports.ledger.client import LedgerCursor


class LedgerCursorStorePort(Protocol):
    """Persist and retrieve per-(pot, source) ledger cursors."""

    def get(self, *, pot_id: str, source_id: str) -> LedgerCursor | None:
        """Return the stored cursor for a source, or ``None`` if never pulled."""
        ...

    def set(self, *, pot_id: str, cursor: LedgerCursor) -> None:
        """Persist the advance cursor after a successful pull/reconcile."""
        ...


__all__ = ["LedgerCursorStorePort"]

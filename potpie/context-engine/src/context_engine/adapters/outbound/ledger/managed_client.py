"""Managed Event Ledger client (dummy/TODO).

The managed Potpie Event Ledger owns provider credentials, webhooks, normalized
event history, and cursors. The local graph only pulls through this port — no
provider credentials enter the local process. Not implemented yet; reports
unavailable so ``potpie ledger status`` is honest.

    TODO(stage-N): implement the managed ledger HTTP client (auth via
    ``cloud login``), pull pagination, and source listing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from context_engine.domain.errors import CapabilityNotImplemented
from context_engine.domain.ports.ledger.client import (
    LedgerCursor,
    LedgerHealth,
    LedgerPage,
    LedgerSource,
)


@dataclass(slots=True)
class ManagedEventLedgerClient:
    binding: str = "managed"

    def fetch(
        self,
        *,
        pot_id: str,
        source_id: str,
        cursor: LedgerCursor | None,
        limit: int = 100,
    ) -> LedgerPage:
        return LedgerPage()

    def query(
        self,
        *,
        pot_id: str,
        source_id: str | None = None,
        kind: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> LedgerPage:
        raise CapabilityNotImplemented(
            "ledger.managed.query",
            detail="managed ledger history query not implemented",
            recommended_next_action="bind a ledger with 'potpie ledger use managed' once managed login lands (HU3)",
        )

    def sources(self, *, pot_id: str) -> list[LedgerSource]:
        return []

    def health(self) -> LedgerHealth:
        return LedgerHealth(
            available=False,
            binding=self.binding,
            detail="managed ledger client not implemented (TODO)",
        )


__all__ = ["ManagedEventLedgerClient"]

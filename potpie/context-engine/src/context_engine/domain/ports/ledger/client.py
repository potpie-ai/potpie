"""``EventLedgerClientPort`` — the pull/cursor seam to the Event Ledger.

The Event Ledger is a *separate* managed-or-self-hosted source-event service.
It owns provider credentials, webhooks, normalized event history, and cursors.
A graph (local or managed) does not store source events — it *pulls* normalized
events from the ledger through this port and reconciles them into claims.

This port is deliberately the only thing the local daemon needs to consume a
ledger; provider credentials never enter the local process. See
``cursor.py`` (cursor storage) and ``reconciler.py`` (events → claims).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class LedgerCursor:
    """Opaque per-source position in the ledger event stream."""

    source_id: str
    token: str | None = None


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    """One normalized source event as the graph consumes it.

    Already provider-normalized by the ledger — the graph never parses raw
    GitHub/Linear/Slack payloads.
    """

    event_id: str
    source_id: str
    provider: str  # github | linear | slack | notion | ...
    kind: str  # normalized event kind (e.g. pr_merge, issue_create)
    payload: Mapping[str, Any] = field(default_factory=dict)
    occurred_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class LedgerSource:
    """A source connector configured on the ledger for a pot."""

    source_id: str
    provider: str
    display_name: str
    connected: bool = False


@dataclass(frozen=True, slots=True)
class LedgerHealth:
    """Ledger reachability/binding for ``ledger status`` and readiness."""

    available: bool
    binding: str = "none"  # none | managed | self_hosted
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class LedgerPage:
    """One page of pulled events plus the advance cursor."""

    events: tuple[LedgerEvent, ...] = ()
    next_cursor: LedgerCursor | None = None
    has_more: bool = False


class EventLedgerClientPort(Protocol):
    """Pull normalized source events from the Event Ledger."""

    def fetch(
        self,
        *,
        pot_id: str,
        source_id: str,
        cursor: LedgerCursor | None,
        limit: int = 100,
    ) -> LedgerPage:
        """Fetch the next page of events for ``source_id`` after ``cursor``."""
        ...

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
        """Inspect ledger event history WITHOUT touching the graph consumer cursor.

        Distinct from ``fetch``: ``query`` is read-only history inspection
        (``potpie ledger query``) — it never advances the per-(pot, source)
        consumer cursor. ``fetch`` is the cursor-based pull that drives ``pull``.
        """
        ...

    def sources(self, *, pot_id: str) -> list[LedgerSource]:
        """List source connectors configured on the ledger for this pot."""
        ...

    def health(self) -> LedgerHealth:
        """Ledger reachability + binding kind."""
        ...


__all__ = [
    "EventLedgerClientPort",
    "LedgerCursor",
    "LedgerEvent",
    "LedgerHealth",
    "LedgerPage",
    "LedgerSource",
]

"""Self-hosted Event Ledger client + an in-process fixture client.

``SelfHostedEventLedgerClient`` targets a user-run ledger URL using the same
pull/cursor contract as the managed client (dummy/TODO).

``FixtureEventLedgerClient`` is an in-process client seeded with synthetic
normalized events so tests can exercise ledger reads without any network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from potpie.context_engine.domain.errors import CapabilityNotImplemented
from potpie.context_engine.domain.ports.ledger.client import (
    LedgerCursor,
    LedgerEvent,
    LedgerHealth,
    LedgerPage,
    LedgerSource,
)


@dataclass(slots=True)
class SelfHostedEventLedgerClient:
    url: str
    binding: str = "self_hosted"

    def fetch(
        self,
        *,
        pot_id: str,
        source_id: str,
        cursor: LedgerCursor | None,
        limit: int = 100,
    ) -> LedgerPage:
        # TODO(stage-N): HTTP pull against ``self.url`` with cursor pagination.
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
            "ledger.self_hosted.query",
            detail=f"self-hosted ledger history query not implemented (url={self.url})",
            recommended_next_action="implement the self-hosted ledger query HTTP contract",
        )

    def sources(self, *, pot_id: str) -> list[LedgerSource]:
        return []

    def health(self) -> LedgerHealth:
        return LedgerHealth(
            available=False,
            binding=self.binding,
            detail=f"self-hosted ledger client not implemented (url={self.url}) (TODO)",
        )


@dataclass(slots=True)
class FixtureEventLedgerClient:
    """In-process ledger seeded with events — for POC + tests only."""

    events_by_source: dict[str, list[LedgerEvent]] = field(default_factory=dict)
    binding: str = "self_hosted"

    def seed(self, source_id: str, events: list[LedgerEvent]) -> None:
        self.events_by_source.setdefault(source_id, []).extend(events)

    def fetch(
        self,
        *,
        pot_id: str,
        source_id: str,
        cursor: LedgerCursor | None,
        limit: int = 100,
    ) -> LedgerPage:
        events = self.events_by_source.get(source_id, [])
        start = int(cursor.token) if cursor and cursor.token else 0
        page = events[start : start + limit]
        new_index = start + len(page)
        return LedgerPage(
            events=tuple(page),
            next_cursor=LedgerCursor(source_id=source_id, token=str(new_index)),
            has_more=new_index < len(events),
        )

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
        # Read-only history scan across all seeded sources; no cursor involved.
        matches: list[LedgerEvent] = []
        for sid, events in self.events_by_source.items():
            if source_id and sid != source_id:
                continue
            for event in events:
                if kind and event.kind != kind:
                    continue
                if since and event.occurred_at and event.occurred_at < since:
                    continue
                if until and event.occurred_at and event.occurred_at > until:
                    continue
                matches.append(event)
        return LedgerPage(
            events=tuple(matches[:limit]),
            next_cursor=None,
            has_more=len(matches) > limit,
        )

    def sources(self, *, pot_id: str) -> list[LedgerSource]:
        return [
            LedgerSource(
                source_id=sid,
                provider=(events[0].provider if events else "unknown"),
                display_name=sid,
                connected=True,
            )
            for sid, events in self.events_by_source.items()
        ]

    def health(self) -> LedgerHealth:
        return LedgerHealth(
            available=True, binding=self.binding, detail="fixture ledger"
        )


__all__ = ["FixtureEventLedgerClient", "SelfHostedEventLedgerClient"]

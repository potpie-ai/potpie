"""Primary operational store for ingestion event lifecycle (not a side ledger)."""

from __future__ import annotations

from typing import Protocol

from domain.ingestion_event_models import (
    CreateIngestionEventParams,
    EventListFilters,
    EventListPage,
    EventTransition,
    IngestionEvent,
)


class IngestionEventStore(Protocol):
    """Event lifecycle and progress; separate from plan/step stores conceptually."""

    def create_event(self, params: CreateIngestionEventParams) -> IngestionEvent:
        ...

    def get_event(self, event_id: str) -> IngestionEvent | None:
        ...

    def find_duplicate(
        self, pot_id: str, dedup_key: str | None, ingestion_kind: str
    ) -> IngestionEvent | None:
        """Return existing event when dedupe matches; ``dedup_key`` may be null (no match)."""
        ...

    def transition_event(self, event_id: str, transition: EventTransition) -> IngestionEvent | None:
        """Apply lifecycle fields; return updated row or None if missing."""
        ...

    def list_events(
        self,
        pot_id: str,
        filters: EventListFilters | None,
        *,
        cursor: str | None,
        limit: int,
    ) -> EventListPage:
        """Latest-first listing with opaque cursor pagination."""
        ...

    def record_progress(
        self,
        event_id: str,
        *,
        step_total: int | None = None,
        step_done: int | None = None,
        step_error: int | None = None,
    ) -> None:
        ...

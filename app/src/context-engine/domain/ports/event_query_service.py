"""Read model for event detail and listings (API + dashboard)."""

from __future__ import annotations

from typing import Protocol

from domain.ingestion_event_models import EventListFilters, EventListPage, IngestionEvent


class EventQueryService(Protocol):
    """
    Query ingestion state from the same store as the control plane (not broker state).

    Typically implemented atop :class:`IngestionEventStore` or a read replica.
    """

    def get_event(self, event_id: str) -> IngestionEvent | None:
        ...

    def list_events(
        self,
        pot_id: str,
        filters: EventListFilters | None,
        *,
        cursor: str | None,
        limit: int,
    ) -> EventListPage:
        ...

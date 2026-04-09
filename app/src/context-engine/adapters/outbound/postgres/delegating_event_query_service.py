"""Read model delegating to :class:`IngestionEventStore` (same operational truth)."""

from __future__ import annotations

from domain.ingestion_event_models import EventListFilters, EventListPage, IngestionEvent
from domain.ports.event_query_service import EventQueryService
from domain.ports.ingestion_event_store import IngestionEventStore


class DelegatingEventQueryService(EventQueryService):
    def __init__(self, store: IngestionEventStore) -> None:
        self._store = store

    def get_event(self, event_id: str) -> IngestionEvent | None:
        return self._store.get_event(event_id)

    def list_events(
        self,
        pot_id: str,
        filters: EventListFilters | None,
        *,
        cursor: str | None,
        limit: int,
    ) -> EventListPage:
        return self._store.list_events(pot_id, filters, cursor=cursor, limit=limit)

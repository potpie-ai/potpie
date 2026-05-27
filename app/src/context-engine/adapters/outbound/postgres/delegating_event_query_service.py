"""Read model delegating to :class:`IngestionEventStore` (same operational truth)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from adapters.outbound.postgres.models import ContextEventModel
from domain.context_status import EventLedgerHealth
from domain.ingestion_db_status import canonical_statuses_to_db_filters, db_status_to_canonical
from domain.ingestion_event_models import EventListFilters, EventListPage, IngestionEvent
from domain.ports.event_query_service import EventQueryService
from domain.ports.ingestion_event_store import IngestionEventStore


class DelegatingEventQueryService(EventQueryService):
    def __init__(self, store: IngestionEventStore, session: Session | None = None) -> None:
        self._store = store
        self._db = session

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

    def summarize_pot_events(
        self,
        pot_id: str,
        *,
        recent_error_limit: int = 5,
    ) -> EventLedgerHealth:
        if self._db is None:
            return EventLedgerHealth()

        counts: dict[str, int] = {"queued": 0, "processing": 0, "done": 0, "error": 0}
        rows = self._db.execute(
            select(ContextEventModel.status, func.count())
            .where(ContextEventModel.pot_id == pot_id)
            .group_by(ContextEventModel.status)
        ).all()
        for raw_status, n in rows:
            canonical = db_status_to_canonical(raw_status)
            counts[canonical] = counts.get(canonical, 0) + int(n or 0)

        done_filter = canonical_statuses_to_db_filters(("done",))
        last_success = self._db.scalar(
            select(func.max(ContextEventModel.completed_at))
            .where(
                ContextEventModel.pot_id == pot_id,
                ContextEventModel.status.in_(tuple(done_filter)),
            )
        )
        if last_success is None:
            last_success = self._db.scalar(
                select(func.max(ContextEventModel.received_at))
                .where(
                    ContextEventModel.pot_id == pot_id,
                    ContextEventModel.status.in_(tuple(done_filter)),
                )
            )

        error_filter = canonical_statuses_to_db_filters(("error",))
        error_rows = self._db.execute(
            select(
                ContextEventModel.id,
                ContextEventModel.event_type,
                ContextEventModel.action,
                ContextEventModel.source_system,
                ContextEventModel.event_error,
                ContextEventModel.completed_at,
                ContextEventModel.received_at,
                ContextEventModel.repo_name,
            )
            .where(
                ContextEventModel.pot_id == pot_id,
                ContextEventModel.status.in_(tuple(error_filter)),
            )
            .order_by(
                desc(func.coalesce(ContextEventModel.completed_at, ContextEventModel.received_at))
            )
            .limit(max(recent_error_limit, 0))
        ).all()

        recent_errors = []
        last_error_at: datetime | None = None
        for r in error_rows:
            ts = r.completed_at or r.received_at
            if last_error_at is None and isinstance(ts, datetime):
                last_error_at = _ensure_aware(ts)
            recent_errors.append(
                {
                    "event_id": r.id,
                    "event_type": r.event_type,
                    "action": r.action,
                    "source_system": r.source_system,
                    "repo_name": r.repo_name,
                    "error": r.event_error,
                    "at": _ensure_aware(ts).isoformat() if isinstance(ts, datetime) else None,
                }
            )

        return EventLedgerHealth(
            counts=counts,
            last_success_at=_ensure_aware(last_success) if isinstance(last_success, datetime) else None,
            last_error_at=last_error_at,
            recent_errors=recent_errors,
        )


def _ensure_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

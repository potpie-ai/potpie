"""Postgres implementation of :class:`IngestionEventStore` over ``context_events``."""

from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session

from adapters.outbound.postgres.models import ContextEventModel
from domain.ingestion_db_status import (
    canonical_status_to_db,
    canonical_statuses_to_db_filters,
    db_status_to_canonical,
)
from domain.ingestion_event_models import (
    CreateIngestionEventParams,
    EventListFilters,
    EventListPage,
    EventTransition,
    IngestionEvent,
)
from domain.ports.ingestion_event_store import IngestionEventStore

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _encode_list_cursor(received_at: datetime, event_id: str) -> str:
    ra = _ensure_aware(received_at)
    raw = f"v1|{ra.isoformat()}|{event_id}".encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_list_cursor(cursor: str) -> tuple[datetime, str]:
    pad = "=" * (-len(cursor) % 4)
    decoded = base64.urlsafe_b64decode(cursor + pad).decode()
    parts = decoded.split("|", 2)
    if len(parts) != 3 or parts[0] != "v1":
        raise ValueError("invalid list cursor")
    ts = datetime.fromisoformat(parts[1])
    return _ensure_aware(ts), parts[2]


class SqlAlchemyIngestionEventStore(IngestionEventStore):
    """Maps canonical ingestion events to ``context_events`` (legacy status strings in DB)."""

    def __init__(self, session: Session) -> None:
        self._db = session

    def create_event(self, params: CreateIngestionEventParams) -> IngestionEvent:
        submitted = params.submitted_at or _utcnow()
        row = ContextEventModel(
            id=params.event_id,
            pot_id=params.pot_id,
            provider=params.provider,
            provider_host=params.provider_host,
            repo_name=params.repo_name,
            source_system=params.source_system,
            event_type=params.event_type,
            action=params.action,
            source_id=params.source_id,
            payload=params.payload,
            occurred_at=None,
            received_at=submitted,
            status=canonical_status_to_db(params.status),
            ingestion_kind=params.ingestion_kind,
            source_channel=params.source_channel,
            dedup_key=params.dedup_key,
            stage=params.stage,
            step_total=0,
            step_done=0,
            step_error=0,
            event_metadata=dict(params.metadata),
            idempotency_key=None,
        )
        self._db.add(row)
        self._db.commit()
        self._db.refresh(row)
        return self._row_to_domain(row)

    def get_event(self, event_id: str) -> IngestionEvent | None:
        row = self._db.scalar(select(ContextEventModel).where(ContextEventModel.id == event_id))
        if row is None:
            return None
        return self._row_to_domain(row)

    def find_duplicate(
        self, pot_id: str, dedup_key: str | None, ingestion_kind: str
    ) -> IngestionEvent | None:
        if dedup_key is None or dedup_key == "":
            return None
        row = self._db.scalar(
            select(ContextEventModel).where(
                ContextEventModel.pot_id == pot_id,
                ContextEventModel.ingestion_kind == ingestion_kind,
                ContextEventModel.dedup_key == dedup_key,
            )
        )
        if row is None:
            return None
        return self._row_to_domain(row)

    def transition_event(self, event_id: str, transition: EventTransition) -> IngestionEvent | None:
        row = self._db.scalar(select(ContextEventModel).where(ContextEventModel.id == event_id))
        if row is None:
            return None
        vals: dict[str, Any] = {}
        if transition.to_status is not None:
            vals["status"] = canonical_status_to_db(transition.to_status)
        if transition.to_stage is not None:
            vals["stage"] = transition.to_stage
        if transition.error is not None:
            vals["event_error"] = transition.error[:8000]
        if transition.started_at is not None:
            vals["started_at"] = transition.started_at
        if transition.completed_at is not None:
            vals["completed_at"] = transition.completed_at
        if vals:
            self._db.execute(update(ContextEventModel).where(ContextEventModel.id == event_id).values(**vals))
            self._db.commit()
        refreshed = self._db.scalar(select(ContextEventModel).where(ContextEventModel.id == event_id))
        return self._row_to_domain(refreshed) if refreshed else None

    def list_events(
        self,
        pot_id: str,
        filters: EventListFilters | None,
        *,
        cursor: str | None,
        limit: int,
    ) -> EventListPage:
        q = select(ContextEventModel).where(ContextEventModel.pot_id == pot_id)
        if filters:
            if filters.ingestion_kinds:
                q = q.where(ContextEventModel.ingestion_kind.in_(tuple(filters.ingestion_kinds)))
            if filters.statuses:
                db_set = canonical_statuses_to_db_filters(tuple(filters.statuses))
                q = q.where(ContextEventModel.status.in_(tuple(db_set)))
        if cursor:
            try:
                ts, eid = _decode_list_cursor(cursor)
            except (ValueError, UnicodeDecodeError) as e:
                logger.warning("invalid event list cursor: %s", e)
                return EventListPage(items=(), next_cursor=None)
            q = q.where(
                or_(
                    ContextEventModel.received_at < ts,
                    and_(ContextEventModel.received_at == ts, ContextEventModel.id < eid),
                )
            )
        q = q.order_by(ContextEventModel.received_at.desc(), ContextEventModel.id.desc()).limit(limit + 1)
        rows = list(self._db.scalars(q).all())
        has_more = len(rows) > limit
        page_rows = rows[:limit]
        items = tuple(self._row_to_domain(r) for r in page_rows)
        next_cursor: str | None = None
        if has_more and page_rows:
            last = page_rows[-1]
            ra = last.received_at
            if isinstance(ra, datetime):
                next_cursor = _encode_list_cursor(ra, last.id)
        return EventListPage(items=items, next_cursor=next_cursor)

    def record_progress(
        self,
        event_id: str,
        *,
        step_total: int | None = None,
        step_done: int | None = None,
        step_error: int | None = None,
    ) -> None:
        vals: dict[str, Any] = {}
        if step_total is not None:
            vals["step_total"] = step_total
        if step_done is not None:
            vals["step_done"] = step_done
        if step_error is not None:
            vals["step_error"] = step_error
        if not vals:
            return
        self._db.execute(update(ContextEventModel).where(ContextEventModel.id == event_id).values(**vals))
        self._db.commit()

    def _row_to_domain(self, row: ContextEventModel) -> IngestionEvent:
        meta = row.event_metadata
        if not isinstance(meta, dict):
            meta = {}
        received = row.received_at
        if not isinstance(received, datetime):
            received = _utcnow()
        started = row.started_at
        if started is not None and not isinstance(started, datetime):
            started = None
        completed = row.completed_at
        if completed is not None and not isinstance(completed, datetime):
            completed = None
        return IngestionEvent(
            event_id=row.id,
            pot_id=row.pot_id,
            ingestion_kind=row.ingestion_kind,
            source_channel=getattr(row, "source_channel", None) or "unknown",
            source_system=row.source_system,
            event_type=row.event_type,
            action=row.action,
            source_id=row.source_id,
            dedup_key=getattr(row, "dedup_key", None),
            status=db_status_to_canonical(row.status),
            stage=getattr(row, "stage", None),
            submitted_at=_ensure_aware(received),
            started_at=started,
            completed_at=completed,
            error=getattr(row, "event_error", None),
            payload=row.payload if isinstance(row.payload, dict) else {},
            metadata=meta,
            step_total=int(getattr(row, "step_total", 0) or 0),
            step_done=int(getattr(row, "step_done", 0) or 0),
            step_error=int(getattr(row, "step_error", 0) or 0),
            provider=row.provider,
            provider_host=row.provider_host,
            repo_name=row.repo_name,
            source_event_id=getattr(row, "source_event_id", None),
            job_id=getattr(row, "job_id", None),
            correlation_id=getattr(row, "correlation_id", None),
            idempotency_key=getattr(row, "idempotency_key", None),
            occurred_at=row.occurred_at if isinstance(getattr(row, "occurred_at", None), datetime) else None,
            raw_status=row.status,
        )

"""Postgres implementation of ``BatchRepositoryPort``."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from adapters.outbound.postgres.models import (
    ContextReconciliationBatchEventModel,
    ContextReconciliationBatchModel,
)
from domain.ports.batch_repository import BatchRepositoryPort
from domain.reconciliation_batch import (
    BATCH_STATUS_CLAIMED,
    BATCH_STATUS_DONE,
    BATCH_STATUS_FAILED,
    BATCH_STATUS_PENDING,
    BATCH_STATUS_RUNNING,
    BatchEventRef,
    ReconciliationBatch,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class SqlAlchemyBatchRepository(BatchRepositoryPort):
    """Postgres-backed batch repo with per-pot coalescing and skip-locked claim-by-id."""

    def __init__(self, session: Session) -> None:
        self._db = session

    def upsert_open_batch_for_pot(self, pot_id: str, event_id: str) -> str:
        # Single-event admission/retry: a degenerate bulk of one. One code
        # path keeps the coalescing + advisory-lock semantics identical.
        return self.add_events_to_open_batch_for_pot(pot_id, [event_id])

    def add_events_to_open_batch_for_pot(
        self, pot_id: str, event_ids: list[str]
    ) -> str:
        if not event_ids:
            raise ValueError("event_ids must be non-empty")
        # De-dupe while preserving order so a caller passing the same id
        # twice doesn't widen the VALUES list needlessly.
        unique_ids = list(dict.fromkeys(event_ids))

        batch_id = self._resolve_or_create_pending_batch_id(pot_id)

        # One bulk membership insert for the whole set. The composite PK
        # (batch_id, event_id) makes this idempotent — re-adding an event
        # already in the batch is a no-op, not a duplicate or an error.
        self._db.execute(
            pg_insert(ContextReconciliationBatchEventModel)
            .values(
                [{"batch_id": batch_id, "event_id": eid} for eid in unique_ids]
            )
            .on_conflict_do_nothing()
        )
        self._db.commit()
        return batch_id

    def _resolve_or_create_pending_batch_id(self, pot_id: str) -> str:
        """Return the pot's open pending batch id, creating one if needed.

        Serializes per-pot with a transaction-scoped advisory lock so two
        concurrent producers can't both insert a "first" batch for the same
        pot. ``pg_advisory_xact_lock`` is released automatically at
        COMMIT/ROLLBACK. Does not commit — the caller owns the transaction.
        """
        self._db.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:pot_id))"),
            {"pot_id": pot_id},
        )

        # Coalesce into the pot's pending batch only. Events landing while a
        # batch is claimed/running get a *new* pending batch — the in-flight
        # run has already snapshotted its event list and would otherwise
        # silently drop the new events.
        existing = self._db.scalar(
            select(ContextReconciliationBatchModel)
            .where(
                ContextReconciliationBatchModel.pot_id == pot_id,
                ContextReconciliationBatchModel.status == BATCH_STATUS_PENDING,
            )
            .order_by(ContextReconciliationBatchModel.created_at.asc())
            .limit(1)
        )
        if existing is not None:
            return existing.id

        batch_id = str(uuid4())
        self._db.add(
            ContextReconciliationBatchModel(
                id=batch_id,
                pot_id=pot_id,
                status=BATCH_STATUS_PENDING,
                attempt_count=0,
            )
        )
        try:
            self._db.flush()
        except IntegrityError:
            # Lost the race despite the advisory lock (e.g. another node
            # without the same hash). Re-select.
            self._db.rollback()
            existing = self._db.scalar(
                select(ContextReconciliationBatchModel).where(
                    ContextReconciliationBatchModel.pot_id == pot_id,
                    ContextReconciliationBatchModel.status == BATCH_STATUS_PENDING,
                )
            )
            if existing is None:
                raise
            return existing.id
        return batch_id

    def claim_batch_by_id(self, batch_id: str) -> ReconciliationBatch | None:
        """Atomically transition ``pending`` → ``claimed`` for one batch id."""
        row = self._db.execute(
            select(ContextReconciliationBatchModel)
            .where(
                ContextReconciliationBatchModel.id == batch_id,
                ContextReconciliationBatchModel.status == BATCH_STATUS_PENDING,
            )
            .with_for_update(skip_locked=True)
        ).scalar_one_or_none()
        if row is None:
            return None
        row.status = BATCH_STATUS_CLAIMED
        row.claimed_at = _utcnow()
        row.attempt_count = (row.attempt_count or 0) + 1
        claimed = self._to_batch(row)
        self._db.commit()
        return claimed

    def get_batch(self, batch_id: str) -> ReconciliationBatch | None:
        row = self._db.scalar(
            select(ContextReconciliationBatchModel).where(
                ContextReconciliationBatchModel.id == batch_id
            )
        )
        return self._to_batch(row) if row else None

    def list_stale_in_flight_batches(
        self, older_than_seconds: float
    ) -> list[ReconciliationBatch]:
        # ``claimed_at`` is stamped by ``claim_batch_by_id`` and left intact
        # by ``mark_batch_running``, so it marks the start of the current
        # in-flight attempt. Compare it against the *database* clock
        # (``now()``) so the lease cutoff doesn't depend on worker wall-clock
        # skew. The ``timedelta`` is bound as an ``interval`` parameter, so
        # the cutoff stays parameterized without PG-specific function syntax.
        rows = self._db.execute(
            select(ContextReconciliationBatchModel)
            .where(
                ContextReconciliationBatchModel.status.in_(
                    (BATCH_STATUS_CLAIMED, BATCH_STATUS_RUNNING)
                ),
                ContextReconciliationBatchModel.claimed_at.is_not(None),
                ContextReconciliationBatchModel.claimed_at
                < func.now() - timedelta(seconds=older_than_seconds),
            )
            .order_by(ContextReconciliationBatchModel.claimed_at.asc())
        ).scalars().all()
        return [self._to_batch(r) for r in rows]

    def list_events_for_batch(self, batch_id: str) -> list[BatchEventRef]:
        rows = self._db.execute(
            select(ContextReconciliationBatchEventModel)
            .where(ContextReconciliationBatchEventModel.batch_id == batch_id)
            .order_by(ContextReconciliationBatchEventModel.added_at.asc())
        ).scalars().all()
        return [
            BatchEventRef(
                event_id=r.event_id,
                added_at=_ensure_aware(r.added_at) if r.added_at else _utcnow(),
                processed_at=_ensure_aware(r.processed_at) if r.processed_at else None,
            )
            for r in rows
        ]

    def mark_batch_running(self, batch_id: str) -> None:
        self._db.execute(
            update(ContextReconciliationBatchModel)
            .where(ContextReconciliationBatchModel.id == batch_id)
            .values(status=BATCH_STATUS_RUNNING)
        )
        self._db.commit()

    def mark_batch_done(
        self,
        batch_id: str,
        *,
        completed_event_ids: list[str],
    ) -> None:
        now = _utcnow()
        self._db.execute(
            update(ContextReconciliationBatchModel)
            .where(ContextReconciliationBatchModel.id == batch_id)
            .values(status=BATCH_STATUS_DONE, completed_at=now, last_error=None)
        )
        if completed_event_ids:
            self._db.execute(
                update(ContextReconciliationBatchEventModel)
                .where(
                    ContextReconciliationBatchEventModel.batch_id == batch_id,
                    ContextReconciliationBatchEventModel.event_id.in_(completed_event_ids),
                )
                .values(processed_at=now)
            )
        self._db.commit()

    def get_open_batch_id_for_pot(self, pot_id: str) -> str | None:
        row_id = self._db.scalar(
            select(ContextReconciliationBatchModel.id)
            .where(
                ContextReconciliationBatchModel.pot_id == pot_id,
                ContextReconciliationBatchModel.status == BATCH_STATUS_PENDING,
            )
            .order_by(ContextReconciliationBatchModel.created_at.asc())
            .limit(1)
        )
        return row_id  # type: ignore[return-value]

    def get_latest_batch_id_for_event(self, event_id: str) -> str | None:
        """Most-recent batch this event belongs to (for the activity stream).

        An event is re-added to a fresh open batch on retry, so we want the
        newest membership — that's the run the user is watching.
        """
        row_id = self._db.scalar(
            select(ContextReconciliationBatchEventModel.batch_id)
            .join(
                ContextReconciliationBatchModel,
                ContextReconciliationBatchModel.id
                == ContextReconciliationBatchEventModel.batch_id,
            )
            .where(ContextReconciliationBatchEventModel.event_id == event_id)
            .order_by(ContextReconciliationBatchModel.created_at.desc())
            .limit(1)
        )
        return row_id  # type: ignore[return-value]

    def mark_batch_failed(self, batch_id: str, error: str) -> None:
        self._db.execute(
            update(ContextReconciliationBatchModel)
            .where(ContextReconciliationBatchModel.id == batch_id)
            .values(
                status=BATCH_STATUS_FAILED,
                completed_at=_utcnow(),
                last_error=error[:8000] if error else None,
            )
        )
        self._db.commit()

    @staticmethod
    def _to_batch(row: ContextReconciliationBatchModel) -> ReconciliationBatch:
        return ReconciliationBatch(
            id=row.id,
            pot_id=row.pot_id,
            status=row.status,
            attempt_count=row.attempt_count or 0,
            created_at=_ensure_aware(row.created_at) if row.created_at else _utcnow(),
            claimed_at=_ensure_aware(row.claimed_at) if row.claimed_at else None,
            completed_at=_ensure_aware(row.completed_at) if row.completed_at else None,
            last_error=row.last_error,
        )

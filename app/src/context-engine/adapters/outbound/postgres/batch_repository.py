"""Postgres implementation of ``BatchRepositoryPort``."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select, text, update
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
        # Serialize per-pot upsert with a transaction-scoped advisory lock so two
        # concurrent producers can't both insert a "first" batch for the same pot.
        # `pg_advisory_xact_lock` is released automatically at COMMIT/ROLLBACK.
        self._db.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:pot_id))"),
            {"pot_id": pot_id},
        )

        # Coalesce into the pot's pending batch only. Events landing while a
        # batch is claimed/running get a *new* pending batch — the in-flight
        # run has already snapshotted its event list and would otherwise
        # silently drop the new event.
        existing = self._db.scalar(
            select(ContextReconciliationBatchModel)
            .where(
                ContextReconciliationBatchModel.pot_id == pot_id,
                ContextReconciliationBatchModel.status == BATCH_STATUS_PENDING,
            )
            .order_by(ContextReconciliationBatchModel.created_at.asc())
            .limit(1)
        )

        if existing is None:
            batch_id = str(uuid4())
            row = ContextReconciliationBatchModel(
                id=batch_id,
                pot_id=pot_id,
                status=BATCH_STATUS_PENDING,
                attempt_count=0,
            )
            self._db.add(row)
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
                batch_id = existing.id
            else:
                self._add_event_membership(batch_id, event_id)
                self._db.commit()
                return batch_id

        batch_id = existing.id
        self._add_event_membership(batch_id, event_id)
        self._db.commit()
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

    def _add_event_membership(self, batch_id: str, event_id: str) -> None:
        existing = self._db.scalar(
            select(ContextReconciliationBatchEventModel).where(
                ContextReconciliationBatchEventModel.batch_id == batch_id,
                ContextReconciliationBatchEventModel.event_id == event_id,
            )
        )
        if existing is not None:
            return
        self._db.add(
            ContextReconciliationBatchEventModel(
                batch_id=batch_id,
                event_id=event_id,
            )
        )
        try:
            self._db.flush()
        except IntegrityError:
            self._db.rollback()  # concurrent insert — that's fine

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

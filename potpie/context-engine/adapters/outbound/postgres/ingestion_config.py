"""SqlAlchemy adapter for :class:`IngestionConfigPort`."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.orm import Session

from adapters.outbound.postgres.models import ContextPotIngestionConfigModel
from domain.ports.ingestion_config import (
    IngestionConfig,
    IngestionConfigPort,
    IngestionMode,
    parse_ingestion_mode,
)


def _to_config(row: ContextPotIngestionConfigModel) -> IngestionConfig:
    return IngestionConfig(
        pot_id=row.pot_id,
        mode=parse_ingestion_mode(row.mode),
        window_minutes=row.window_minutes,
        min_batch_size=row.min_batch_size,
    )


class SqlAlchemyIngestionConfig(IngestionConfigPort):
    """Reads ``context_pot_ingestion_config`` and synthesises defaults."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, pot_id: str) -> IngestionConfig:
        row = self._session.get(ContextPotIngestionConfigModel, pot_id)
        if row is None:
            # Pot has no row yet — admission falls through to the platform
            # default. This matches the migration's intent (every existing
            # pot got a windowed row), but new pots created after deploy
            # without an explicit config still behave correctly.
            return IngestionConfig.default(pot_id)
        return _to_config(row)

    def set(
        self,
        *,
        pot_id: str,
        mode: IngestionMode,
        window_minutes: int,
        min_batch_size: int | None,
        actor_user_id: str | None = None,
    ) -> IngestionConfig:
        if mode not in ("immediate", "windowed"):
            raise ValueError(f"unknown ingestion mode: {mode}")
        if window_minutes < 1 or window_minutes > 24 * 60:
            raise ValueError(
                f"window_minutes must be in [1, 1440]; got {window_minutes}"
            )
        if min_batch_size is not None and min_batch_size < 1:
            raise ValueError(f"min_batch_size must be >= 1; got {min_batch_size}")

        row = self._session.get(ContextPotIngestionConfigModel, pot_id)
        now = datetime.now(timezone.utc)
        if row is None:
            row = ContextPotIngestionConfigModel(
                pot_id=pot_id,
                mode=mode,
                window_minutes=window_minutes,
                min_batch_size=min_batch_size,
                updated_at=now,
                updated_by_user_id=actor_user_id,
            )
            self._session.add(row)
        else:
            row.mode = mode
            row.window_minutes = window_minutes
            row.min_batch_size = min_batch_size
            row.updated_at = now
            row.updated_by_user_id = actor_user_id
        self._session.flush()
        return _to_config(row)

    def list_windowed_pots_ready_to_flush(
        self, *, as_of_unix_seconds: float
    ) -> list[IngestionConfig]:
        """SQL: join configs with open pending batches and filter on window/size.

        Uses one query so the dispatcher doesn't pay N round-trips per pot.
        ``min_batch_size`` triggers an early flush regardless of age.
        """
        # We join the config table with context_reconciliation_batches
        # (pending) and aggregate event counts. The query intentionally lives
        # in raw SQL — the relationship across context-engine packages can't
        # use SQLAlchemy mappers without a circular import.
        as_of_dt = datetime.fromtimestamp(as_of_unix_seconds, tz=timezone.utc)
        rows = self._session.execute(
            text(
                """
                SELECT
                    cfg.pot_id,
                    cfg.mode,
                    cfg.window_minutes,
                    cfg.min_batch_size
                FROM context_pot_ingestion_config cfg
                JOIN context_reconciliation_batches b
                  ON b.pot_id = cfg.pot_id
                 AND b.status = 'pending'
                LEFT JOIN context_reconciliation_batch_events be
                  ON be.batch_id = b.id
                WHERE cfg.mode = 'windowed'
                GROUP BY cfg.pot_id, cfg.mode, cfg.window_minutes,
                         cfg.min_batch_size, b.created_at
                HAVING
                    EXTRACT(EPOCH FROM (:as_of - b.created_at))
                        >= cfg.window_minutes * 60
                    OR (
                        cfg.min_batch_size IS NOT NULL
                        AND COUNT(be.event_id) >= cfg.min_batch_size
                    )
                """
            ),
            {"as_of": as_of_dt},
        ).all()
        return [
            IngestionConfig(
                pot_id=r.pot_id,
                mode=parse_ingestion_mode(r.mode),
                window_minutes=r.window_minutes,
                min_batch_size=r.min_batch_size,
            )
            for r in rows
        ]


__all__ = ["SqlAlchemyIngestionConfig"]

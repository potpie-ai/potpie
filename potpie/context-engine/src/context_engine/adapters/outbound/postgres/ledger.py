"""Postgres implementation of IngestionLedgerPort (sync-state + bulk delete)."""

from __future__ import annotations

import logging
from datetime import datetime
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from context_engine.adapters.outbound.postgres.models import ContextEventModel, ContextSyncState
from context_engine.domain.ingestion_kinds import INGESTION_KIND_CONNECTOR_SYNC
from context_engine.domain.ports.ingestion_ledger import (
    IngestionLedgerPort,
    LedgerScope,
    SyncStateRow,
)

logger = logging.getLogger(__name__)


class SqlAlchemyIngestionLedger(IngestionLedgerPort):
    def __init__(self, session: Session) -> None:
        self._db = session

    def rollback(self) -> None:
        self._db.rollback()

    def get_or_create_sync_state(
        self, scope: LedgerScope, source_type: str
    ) -> SyncStateRow:
        row = self._db.scalar(
            select(ContextSyncState).where(
                ContextSyncState.pot_id == scope.pot_id,
                ContextSyncState.provider == scope.provider,
                ContextSyncState.provider_host == scope.provider_host,
                ContextSyncState.repo_name == scope.repo_name,
                ContextSyncState.source_type == source_type,
            )
        )
        if row:
            return self._to_sync_row(row)
        row = ContextSyncState(
            id=str(uuid4()),
            pot_id=scope.pot_id,
            provider=scope.provider,
            provider_host=scope.provider_host,
            repo_name=scope.repo_name,
            source_type=source_type,
            status="idle",
        )
        self._db.add(row)
        self._db.commit()
        self._db.refresh(row)
        return self._to_sync_row(row)

    def update_sync_state_running(self, scope: LedgerScope, source_type: str) -> None:
        row = self._db.scalar(
            select(ContextSyncState).where(
                ContextSyncState.pot_id == scope.pot_id,
                ContextSyncState.provider == scope.provider,
                ContextSyncState.provider_host == scope.provider_host,
                ContextSyncState.repo_name == scope.repo_name,
                ContextSyncState.source_type == source_type,
            )
        )
        if row:
            row.status = "running"
            row.error = None
            self._db.commit()

    def update_sync_state_success(
        self,
        scope: LedgerScope,
        source_type: str,
        last_synced_at: datetime | None,
    ) -> None:
        row = self._db.scalar(
            select(ContextSyncState).where(
                ContextSyncState.pot_id == scope.pot_id,
                ContextSyncState.provider == scope.provider,
                ContextSyncState.provider_host == scope.provider_host,
                ContextSyncState.repo_name == scope.repo_name,
                ContextSyncState.source_type == source_type,
            )
        )
        if row:
            row.status = "success"
            row.last_synced_at = last_synced_at
            row.error = None
            self._db.commit()

    def update_sync_state_error(
        self, scope: LedgerScope, source_type: str, error: str
    ) -> None:
        row = self._db.scalar(
            select(ContextSyncState).where(
                ContextSyncState.pot_id == scope.pot_id,
                ContextSyncState.provider == scope.provider,
                ContextSyncState.provider_host == scope.provider_host,
                ContextSyncState.repo_name == scope.repo_name,
                ContextSyncState.source_type == source_type,
            )
        )
        if row:
            row.status = "error"
            row.error = error
            self._db.commit()

    def delete_all_for_pot(self, pot_id: str) -> int:
        """Delete merged-PR ledger rows and sync state for ``pot_id``."""
        n_ev = (
            self._db.execute(
                delete(ContextEventModel).where(
                    ContextEventModel.pot_id == pot_id,
                    ContextEventModel.ingestion_kind == INGESTION_KIND_CONNECTOR_SYNC,
                )
            ).rowcount
            or 0
        )
        n_sync = (
            self._db.execute(
                delete(ContextSyncState).where(ContextSyncState.pot_id == pot_id)
            ).rowcount
            or 0
        )
        self._db.commit()
        return int(n_ev) + int(n_sync)

    @staticmethod
    def _to_sync_row(row: ContextSyncState) -> SyncStateRow:
        return SyncStateRow(
            pot_id=row.pot_id,
            provider=row.provider,
            provider_host=row.provider_host,
            repo_name=row.repo_name,
            source_type=row.source_type,
            last_synced_at=row.last_synced_at,
            status=row.status,
            error=row.error,
        )

"""Postgres implementation of IngestionLedgerPort."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from adapters.outbound.postgres.models import (
    ContextIngestionLog,
    ContextSyncState,
    RawEvent,
)
from domain.ingestion import BridgeResult
from domain.ports.ingestion_ledger import IngestionLedgerPort, IngestionLogRow, SyncStateRow

logger = logging.getLogger(__name__)


class SqlAlchemyIngestionLedger(IngestionLedgerPort):
    def __init__(self, session: Session) -> None:
        self._db = session

    def rollback(self) -> None:
        self._db.rollback()

    def get_ingestion_log(
        self,
        project_id: str,
        source_type: str,
        source_id: str,
    ) -> IngestionLogRow | None:
        row = self._db.scalar(
            select(ContextIngestionLog).where(
                ContextIngestionLog.project_id == project_id,
                ContextIngestionLog.source_type == source_type,
                ContextIngestionLog.source_id == source_id,
            )
        )
        if row is None:
            return None
        return IngestionLogRow(
            project_id=row.project_id,
            source_type=row.source_type,
            source_id=row.source_id,
            graphiti_episode_uuid=row.graphiti_episode_uuid,
            entity_key=row.entity_key,
        )

    def try_append_ingestion_and_raw_event(
        self,
        project_id: str,
        source_type: str,
        source_id: str,
        graphiti_episode_uuid: str | None,
        payload: dict[str, Any],
    ) -> bool:
        now = datetime.utcnow()
        raw = RawEvent(
            id=str(uuid4()),
            project_id=project_id,
            source_type=source_type,
            source_id=source_id,
            payload=payload,
            received_at=now,
            processed_at=now,
        )
        log = ContextIngestionLog(
            id=str(uuid4()),
            project_id=project_id,
            source_type=source_type,
            source_id=source_id,
            graphiti_episode_uuid=graphiti_episode_uuid,
            bridge_written=False,
        )
        self._db.add(raw)
        self._db.add(log)
        try:
            self._db.commit()
            return True
        except IntegrityError:
            self._db.rollback()
            logger.info(
                "ingest_duplicate project=%s source_type=%s source_id=%s",
                project_id,
                source_type,
                source_id,
            )
            return False

    def update_bridge_status(
        self,
        project_id: str,
        source_type: str,
        source_id: str,
        entity_key: str,
        bridge_result: BridgeResult | None,
        error: str | None,
    ) -> None:
        values: dict[str, Any] = {"entity_key": entity_key}
        if error:
            values["bridge_status"] = "failed"
            values["bridge_error"] = error[:2000]
            values["bridge_written"] = False
        elif bridge_result:
            values["bridge_status"] = "success"
            values["bridge_error"] = None
            values["bridge_written"] = True
            values["bridge_touched_by"] = bridge_result.touched_by
            values["bridge_modified_in"] = bridge_result.modified_in
            values["bridge_has_decision"] = bridge_result.has_decision
        else:
            values["bridge_status"] = "skipped"

        self._db.execute(
            update(ContextIngestionLog)
            .where(
                ContextIngestionLog.project_id == project_id,
                ContextIngestionLog.source_type == source_type,
                ContextIngestionLog.source_id == source_id,
            )
            .values(**values)
        )
        self._db.commit()

    def get_or_create_sync_state(self, project_id: str, source_type: str) -> SyncStateRow:
        row = self._db.scalar(
            select(ContextSyncState).where(
                ContextSyncState.project_id == project_id,
                ContextSyncState.source_type == source_type,
            )
        )
        if row:
            return self._to_sync_row(row)
        row = ContextSyncState(
            id=str(uuid4()),
            project_id=project_id,
            source_type=source_type,
            status="idle",
        )
        self._db.add(row)
        self._db.commit()
        self._db.refresh(row)
        return self._to_sync_row(row)

    def update_sync_state_running(self, project_id: str, source_type: str) -> None:
        row = self._db.scalar(
            select(ContextSyncState).where(
                ContextSyncState.project_id == project_id,
                ContextSyncState.source_type == source_type,
            )
        )
        if row:
            row.status = "running"
            row.error = None
            self._db.commit()

    def update_sync_state_success(
        self,
        project_id: str,
        source_type: str,
        last_synced_at: datetime | None,
    ) -> None:
        row = self._db.scalar(
            select(ContextSyncState).where(
                ContextSyncState.project_id == project_id,
                ContextSyncState.source_type == source_type,
            )
        )
        if row:
            row.status = "success"
            row.last_synced_at = last_synced_at
            row.error = None
            self._db.commit()

    def update_sync_state_error(self, project_id: str, source_type: str, error: str) -> None:
        row = self._db.scalar(
            select(ContextSyncState).where(
                ContextSyncState.project_id == project_id,
                ContextSyncState.source_type == source_type,
            )
        )
        if row:
            row.status = "error"
            row.error = error
            self._db.commit()

    @staticmethod
    def _to_sync_row(row: ContextSyncState) -> SyncStateRow:
        return SyncStateRow(
            project_id=row.project_id,
            source_type=row.source_type,
            last_synced_at=row.last_synced_at,
            status=row.status,
            error=row.error,
        )

"""Postgres ledger: sync state, ingestion log, raw events (port)."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Protocol

from domain.ingestion import BridgeResult


@dataclass(slots=True)
class IngestionLogRow:
    project_id: str
    source_type: str
    source_id: str
    graphiti_episode_uuid: str | None
    entity_key: str | None


@dataclass(slots=True)
class SyncStateRow:
    project_id: str
    source_type: str
    last_synced_at: datetime | None
    status: str
    error: str | None


class IngestionLedgerPort(Protocol):
    def get_ingestion_log(
        self,
        project_id: str,
        source_type: str,
        source_id: str,
    ) -> Optional[IngestionLogRow]:
        ...

    def try_append_ingestion_and_raw_event(
        self,
        project_id: str,
        source_type: str,
        source_id: str,
        graphiti_episode_uuid: str | None,
        payload: dict[str, Any],
    ) -> bool:
        """Insert raw_events + context_ingestion_log. Return False if duplicate key."""

    def update_bridge_status(
        self,
        project_id: str,
        source_type: str,
        source_id: str,
        entity_key: str,
        bridge_result: BridgeResult | None,
        error: str | None,
    ) -> None:
        ...

    def get_or_create_sync_state(self, project_id: str, source_type: str) -> SyncStateRow:
        ...

    def update_sync_state_running(self, project_id: str, source_type: str) -> None:
        ...

    def update_sync_state_success(
        self,
        project_id: str,
        source_type: str,
        last_synced_at: datetime | None,
    ) -> None:
        ...

    def update_sync_state_error(self, project_id: str, source_type: str, error: str) -> None:
        ...

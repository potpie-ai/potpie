"""Postgres ledger: sync state, ingestion log, raw events (port)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Protocol

from domain.ingestion import BridgeResult
from domain.ports.pot_resolution import ResolvedPotRepo


@dataclass(frozen=True, slots=True)
class LedgerScope:
    """Partition key for ledger rows: pot + provider-scoped repo identity."""

    pot_id: str
    provider: str
    provider_host: str
    repo_name: str
    """Normalized ``owner/repo`` within the provider."""


@dataclass(slots=True)
class IngestionLogRow:
    pot_id: str
    provider: str
    provider_host: str
    repo_name: str
    source_type: str
    source_id: str
    graphiti_episode_uuid: str | None
    entity_key: str | None


@dataclass(slots=True)
class SyncStateRow:
    pot_id: str
    provider: str
    provider_host: str
    repo_name: str
    source_type: str
    last_synced_at: datetime | None
    status: str
    error: str | None


class IngestionLedgerPort(Protocol):
    def get_ingestion_log(
        self,
        scope: LedgerScope,
        source_type: str,
        source_id: str,
    ) -> Optional[IngestionLogRow]:
        ...

    def try_append_ingestion_and_raw_event(
        self,
        scope: LedgerScope,
        source_type: str,
        source_id: str,
        graphiti_episode_uuid: str | None,
        payload: dict[str, Any],
    ) -> bool:
        """Insert raw_events + context_ingestion_log. Return False if duplicate key."""

    def update_bridge_status(
        self,
        scope: LedgerScope,
        source_type: str,
        source_id: str,
        entity_key: str,
        bridge_result: BridgeResult | None,
        error: str | None,
    ) -> None:
        ...

    def get_or_create_sync_state(self, scope: LedgerScope, source_type: str) -> SyncStateRow:
        ...

    def update_sync_state_running(self, scope: LedgerScope, source_type: str) -> None:
        ...

    def update_sync_state_success(
        self,
        scope: LedgerScope,
        source_type: str,
        last_synced_at: datetime | None,
    ) -> None:
        ...

    def update_sync_state_error(self, scope: LedgerScope, source_type: str, error: str) -> None:
        ...


def ledger_scope_from_pot_repo(repo: ResolvedPotRepo) -> LedgerScope:
    """Build ledger keys from a resolved repo row inside a pot."""
    return LedgerScope(
        pot_id=repo.pot_id,
        provider=repo.provider,
        provider_host=repo.provider_host,
        repo_name=repo.repo_name,
    )

"""Sync-state ledger port: per-pot connector sync metadata + bulk reset."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from potpie_context_core.domain.ports.pot_resolution import ResolvedPotRepo


@dataclass(frozen=True, slots=True)
class LedgerScope:
    """Partition key for ledger rows: pot + provider-scoped repo identity."""

    pot_id: str
    provider: str
    provider_host: str
    repo_name: str
    """Normalized ``owner/repo`` within the provider."""


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
    def get_or_create_sync_state(
        self, scope: LedgerScope, source_type: str
    ) -> SyncStateRow: ...

    def update_sync_state_running(
        self, scope: LedgerScope, source_type: str
    ) -> None: ...

    def update_sync_state_success(
        self,
        scope: LedgerScope,
        source_type: str,
        last_synced_at: datetime | None,
    ) -> None: ...

    def update_sync_state_error(
        self, scope: LedgerScope, source_type: str, error: str
    ) -> None: ...

    def delete_all_for_pot(self, pot_id: str) -> int:
        """Delete connector-sync ledger rows and sync state for ``pot_id``. Returns rows removed."""
        ...


def ledger_scope_from_pot_repo(repo: ResolvedPotRepo) -> LedgerScope:
    """Build ledger keys from a resolved repo row inside a pot."""
    return LedgerScope(
        pot_id=repo.pot_id,
        provider=repo.provider,
        provider_host=repo.provider_host,
        repo_name=repo.repo_name,
    )

"""Relational state-store + migration seams (setup step 4).

The "Relational state store" component in the setup seam→owner map
(architecture.md "Seam → owner map", row 4) is three independently-ownable
methods: ``pot_service.init`` (owned by :class:`PotManagementService`),
``state_store.provision`` (this :class:`StateStorePort`), and
``migrator.migrate`` (this :class:`MigrationPort`).

The flat-file local profile needs no relational store, so its adapters report
``skipped``. The SQLite/Postgres owner fills ``provision``/``migrate`` behind
these interfaces without touching the orchestrator, which depends only on the
``-> StepResult`` signatures.
"""

from __future__ import annotations

from typing import Protocol

from potpie_context_core.domain.lifecycle import StepResult


class StateStorePort(Protocol):
    """Stand up the relational control-plane store, idempotently."""

    def provision(self) -> StepResult:
        """Create the state DB / schema. ``skipped`` when the profile is flat-file."""
        ...


class MigrationPort(Protocol):
    """Run pending schema migrations against the relational state store."""

    def migrate(self) -> StepResult:
        """Apply outstanding migrations. ``skipped`` when there are none."""
        ...


__all__ = ["MigrationPort", "StateStorePort"]

"""Flat-file ``StateStorePort`` + ``MigrationPort`` — the local-profile no-ops.

The local profile keeps control-plane state as flat JSON (see
``local_pot_store.py``), so there is no relational store to provision and no
schema to migrate: both report ``skipped`` cleanly. This is the seam the
SQLite/Postgres owner replaces — wire a ``SqliteStateStore`` / ``Migrator`` into
``build_host_shell`` behind the same interfaces and the orchestrator is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from potpie.context_engine.domain.lifecycle import SKIPPED, StepResult


@dataclass(slots=True)
class FlatFileStateStore:
    """No relational store on the flat-file profile → ``state_store.provision`` is skipped."""

    def provision(self) -> StepResult:
        return StepResult(
            step="state_store.provision",
            state=SKIPPED,
            detail="flat-file profile: no relational state store to provision",
        )


@dataclass(slots=True)
class FlatFileMigrator:
    """No schema on the flat-file profile → ``migrator.migrate`` is skipped."""

    def migrate(self) -> StepResult:
        return StepResult(
            step="migrator.migrate",
            state=SKIPPED,
            detail="flat-file profile: no migrations to run",
        )


__all__ = ["FlatFileMigrator", "FlatFileStateStore"]

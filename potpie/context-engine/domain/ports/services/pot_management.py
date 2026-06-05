"""``PotManagementService`` — the control-plane service.

The control plane for pots and sources: the active-pot pointer, pot lifecycle
(create/use/rename/reset/archive), the source registry, and graph readiness
rollup. It is the half of ``context_status`` that is *not* graph data.

A **pot** is the workspace/tenant boundary — every query, source, record,
claim, and graph operation is scoped to exactly one pot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Protocol

from domain.lifecycle import StepResult


@dataclass(frozen=True, slots=True)
class PotInfo:
    """One pot in the control plane."""

    pot_id: str
    name: str
    active: bool = False
    archived: bool = False
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """A source registered to a pot (repo, integration binding, …)."""

    source_id: str
    kind: str  # repo | github | linear | ...
    name: str
    last_sync_at: datetime | None = None
    sync_mode: str | None = None
    status: str = "unknown"  # ok | stale | error | unknown


@dataclass(frozen=True, slots=True)
class PotAggregateStatus:
    """Control-plane half of ``context_status``."""

    active_pot: PotInfo | None
    pot_count: int = 0
    sources: tuple[SourceInfo, ...] = ()
    backend_ready: bool = False
    detail: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class PotManagementService(Protocol):
    """Control plane for pots, the active pot, and the source registry."""

    # --- lifecycle ----------------------------------------------------------
    def init(self, *, mode: str, backend: str) -> StepResult:
        """Provision the control-plane state store for this mode/backend and run
        migrations to head (the setup seam). The flat-file POC self-creates; the
        real local state DB (SQLite + migrations) runs schema setup here. Pot
        creation itself is a separate step (``create_pot``)."""
        ...

    # --- pots ---------------------------------------------------------------
    def list_pots(self) -> list[PotInfo]: ...

    def active_pot(self) -> PotInfo | None: ...

    def create_pot(
        self, *, name: str, repo: str | None = None, use: bool = False
    ) -> PotInfo: ...

    def use_pot(self, *, ref: str) -> PotInfo:
        """Set the active pot by id-or-name."""
        ...

    def rename_pot(self, *, ref: str, new_name: str) -> PotInfo: ...

    def reset_pot(self, *, ref: str, confirm: bool = False) -> PotInfo:
        """Clear a pot's graph state (control-plane side of the reset)."""
        ...

    def archive_pot(self, *, ref: str) -> PotInfo: ...

    # --- sources ------------------------------------------------------------
    def add_source(
        self, *, pot_id: str, kind: str, location: str, name: str | None = None
    ) -> SourceInfo: ...

    def list_sources(self, *, pot_id: str) -> list[SourceInfo]: ...

    def source_status(self, *, pot_id: str, source_id: str) -> SourceInfo: ...

    def remove_source(self, *, pot_id: str, source_id: str) -> None: ...

    # --- rollup -------------------------------------------------------------
    def aggregate_status(self, *, pot_id: str | None = None) -> PotAggregateStatus:
        """Control-plane status for ``context_status`` (active pot, sources,
        readiness). ``None`` uses the active pot."""
        ...


__all__ = [
    "PotAggregateStatus",
    "PotInfo",
    "PotManagementService",
    "SourceInfo",
]

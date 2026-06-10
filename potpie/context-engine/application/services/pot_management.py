"""``LocalPotManagementService`` — control plane over a local pot store.

Wraps :class:`LocalPotStore` (flat-file persistence) and reports backend
readiness from the wired ``GraphBackend``. The real control plane is the local
state DB; this proves the service boundary and the CLI wiring.
"""

from __future__ import annotations

from dataclasses import dataclass

from adapters.outbound.pots.local_pot_store import LocalPotStore
from domain.errors import PotNotFound
from domain.lifecycle import DONE, StepResult
from domain.ports.graph.backend import GraphBackend
from domain.ports.services.pot_management import (
    PotAggregateStatus,
    PotInfo,
    SourceInfo,
)


@dataclass(slots=True)
class LocalPotManagementService:
    store: LocalPotStore
    backend: GraphBackend

    # --- lifecycle ----------------------------------------------------------
    def init(self, *, mode: str, backend: str) -> StepResult:
        # Flat-file store self-creates on first write; ensure the home dir
        # exists now so the control plane is ready before the first pot. The real
        # state DB runs SQLite + migrations here.
        self.store.home.mkdir(parents=True, exist_ok=True)
        return StepResult(
            step="pot.init",
            state=DONE,
            detail=f"control-plane store ready at {self.store.home} (mode={mode})",
            metadata={"mode": mode, "backend": backend},
        )

    # --- pots ---------------------------------------------------------------
    def list_pots(self) -> list[PotInfo]:
        return [_pot(row) for row in self.store.list_pots()]

    def active_pot(self) -> PotInfo | None:
        row = self.store.active()
        return _pot(row) if row else None

    def create_pot(
        self, *, name: str, repo: str | None = None, use: bool = False
    ) -> PotInfo:
        return _pot(self.store.create(name=name, repo=repo, use=use))

    def use_pot(self, *, ref: str) -> PotInfo:
        row = self.store.use(ref=ref)
        if row is None:
            raise PotNotFound(f"No pot matching '{ref}'.")
        return _pot(row)

    def rename_pot(self, *, ref: str, new_name: str) -> PotInfo:
        row = self.store.rename(ref=ref, new_name=new_name)
        if row is None:
            raise PotNotFound(f"No pot matching '{ref}'.")
        return _pot(row)

    def reset_pot(self, *, ref: str, confirm: bool = False) -> PotInfo:
        # Resolve the pot, then clear its graph state through the backend.
        target = next(
            (p for p in self.store.list_pots() if ref in (p["pot_id"], p["name"])),
            None,
        )
        if target is None:
            raise PotNotFound(f"No pot matching '{ref}'.")
        self.backend.mutation.reset_pot(target["pot_id"])
        return _pot(target)

    def archive_pot(self, *, ref: str) -> PotInfo:
        row = self.store.archive(ref=ref)
        if row is None:
            raise PotNotFound(f"No pot matching '{ref}'.")
        return _pot(row)

    # --- sources ------------------------------------------------------------
    def add_source(
        self, *, pot_id: str, kind: str, location: str, name: str | None = None
    ) -> SourceInfo:
        return _source(
            self.store.add_source(
                pot_id=pot_id, kind=kind, location=location, name=name
            )
        )

    def list_sources(self, *, pot_id: str) -> list[SourceInfo]:
        return [_source(r) for r in self.store.list_sources(pot_id=pot_id)]

    def source_status(self, *, pot_id: str, source_id: str) -> SourceInfo:
        for row in self.store.list_sources(pot_id=pot_id):
            if row.get("source_id") == source_id:
                return _source(row)
        raise PotNotFound(f"No source '{source_id}' in pot '{pot_id}'.")

    def remove_source(self, *, pot_id: str, source_id: str) -> None:
        self.store.remove_source(pot_id=pot_id, source_id=source_id)

    # --- rollup -------------------------------------------------------------
    def aggregate_status(self, *, pot_id: str | None = None) -> PotAggregateStatus:
        active = self.active_pot()
        target_id = pot_id or (active.pot_id if active else None)
        sources = tuple(self.list_sources(pot_id=target_id)) if target_id else ()
        ready = bool(target_id) and self.backend.mutation.readiness(target_id).ready
        return PotAggregateStatus(
            active_pot=active,
            pot_count=len(self.store.list_pots()),
            sources=sources,
            backend_ready=ready,
            detail=None if target_id else "no active pot — run 'potpie setup'",
        )


def _pot(row: dict) -> PotInfo:
    return PotInfo(
        pot_id=row["pot_id"],
        name=row.get("name", row["pot_id"]),
        active=bool(row.get("active")),
        archived=bool(row.get("archived")),
    )


def _source(row: dict) -> SourceInfo:
    return SourceInfo(
        source_id=row["source_id"],
        kind=row.get("kind", "unknown"),
        name=row.get("name", row.get("location", "")),
        location=row.get("location"),
        status="ok",
    )


__all__ = ["LocalPotManagementService"]

"""``LocalPotManagementService`` — control plane over a local pot store.

Wraps :class:`LocalPotStore` (flat-file persistence) and reports backend
readiness from the wired ``GraphBackend``. The real control plane is the local
state DB; this proves the service boundary and the CLI wiring.
"""

from __future__ import annotations

from dataclasses import dataclass

from potpie_context_engine.adapters.outbound.pots.local_pot_store import LocalPotStore
from potpie_context_engine.domain.errors import PotNotFound
from potpie_context_engine.domain.provisioning import DONE, StepResult
from potpie_context_engine.domain.ports.graph.backend import GraphBackend
from potpie_context_engine.domain.ports.services.pot_management import (
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

    def register_repo_source(
        self,
        *,
        pot_id: str,
        location: str,
        name: str | None = None,
        make_default: bool = True,
    ) -> tuple[SourceInfo, str, bool, bool]:
        row, repo_identity, created, default_bound = self.store.register_repo_source(
            pot_id=pot_id,
            location=location,
            name=name,
            make_default=make_default,
        )
        return _source(row), repo_identity, created, default_bound

    def list_sources(self, *, pot_id: str) -> list[SourceInfo]:
        return [_source(r) for r in self.store.list_sources(pot_id=pot_id)]

    def source_status(self, *, pot_id: str, source_id: str) -> SourceInfo:
        for row in self.store.list_sources(pot_id=pot_id):
            if row.get("source_id") == source_id:
                return _source(row)
        raise PotNotFound(f"No source '{source_id}' in pot '{pot_id}'.")

    def remove_source(self, *, pot_id: str, source_id: str) -> None:
        self.store.remove_source(pot_id=pot_id, source_id=source_id)

    # --- repo-local routing defaults ----------------------------------------
    def repo_default(self, *, repo: str) -> str | None:
        return self.store.repo_default(repo=repo)

    def set_repo_default(self, *, repo: str, pot_id: str) -> None:
        if not any(p.pot_id == pot_id for p in self.list_pots()):
            raise PotNotFound(f"No pot matching '{pot_id}'.")
        self.store.set_repo_default(repo=repo, pot_id=pot_id)

    def clear_repo_default(self, *, repo: str) -> bool:
        return self.store.clear_repo_default(repo=repo)

    def list_repo_defaults(self) -> dict[str, str]:
        return self.store.list_repo_defaults()

    # --- rollup -------------------------------------------------------------
    def aggregate_status(self, *, pot_id: str | None = None) -> PotAggregateStatus:
        active = self.active_pot()
        target_id = pot_id or (active.pot_id if active else None)
        sources = tuple(self.list_sources(pot_id=target_id)) if target_id else ()
        ready = (
            self.backend.mutation.readiness(target_id).ready
            if target_id is not None
            else False
        )
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

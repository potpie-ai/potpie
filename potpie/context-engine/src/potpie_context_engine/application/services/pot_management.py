"""``LocalPotManagementService`` — control plane over a local pot store.

Wraps :class:`LocalPotStore` (flat-file persistence) and reports backend
readiness from the wired ``GraphBackend``. The real control plane is the local
state DB; this proves the service boundary and the CLI wiring.

Pot teardown (``reset_pot`` / ``archive_pot``) also purges the resource store:
graph reset alone would leave chunk files under ``<home>/resources/<pot_dir>/``
pointing at nothing (R8). ``source remove`` is registration-only — it never
touches resources; documents are cleaned with ``resource rm`` or pot teardown.
"""

from __future__ import annotations

from dataclasses import dataclass

from potpie_context_engine.adapters.outbound.pots.local_pot_store import LocalPotStore
from potpie_context_core.errors import PotNotFound
from potpie_context_core.lifecycle import DONE, StepResult
from potpie_context_core.ports.graph.backend import GraphBackend
from potpie_context_core.ports.resource_store import ResourceStorePort
from potpie_context_engine.domain.ports.services.pot_management import (
    PotAggregateStatus,
    PotInfo,
    PotRepoSource,
    PotTeardownResult,
    SourceInfo,
)


@dataclass(slots=True)
class LocalPotManagementService:
    store: LocalPotStore
    backend: GraphBackend
    resources: ResourceStorePort | None = None

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

    def reset_pot(self, *, ref: str, confirm: bool = False) -> PotTeardownResult:
        # Resolve the pot, clear its graph partition, then drop its resource
        # tree. Graph first: a failed purge leaves orphan files (harmless,
        # overwritten on the next import); the reverse would leave live claims
        # pointing at files that no longer exist.
        del confirm  # enforced at the CLI boundary
        target = next(
            (p for p in self.store.list_pots() if ref in (p["pot_id"], p["name"])),
            None,
        )
        if target is None:
            raise PotNotFound(f"No pot matching '{ref}'.")
        return PotTeardownResult(
            pot=_pot(target),
            resources_purged=self._teardown_pot_data(target["pot_id"]),
        )

    def archive_pot(self, *, ref: str) -> PotTeardownResult:
        # Archive is pot deletion from the control plane: soft-flag the pot,
        # and tear down the graph partition plus resource tree so an archived
        # pot cannot leave dangling chunk files behind (R8).
        target = next(
            (p for p in self.store.list_pots() if ref in (p["pot_id"], p["name"])),
            None,
        )
        if target is None:
            raise PotNotFound(f"No pot matching '{ref}'.")
        purged = self._teardown_pot_data(target["pot_id"])
        row = self.store.archive(ref=ref)
        if row is None:
            raise PotNotFound(f"No pot matching '{ref}'.")
        return PotTeardownResult(pot=_pot(row), resources_purged=purged)

    def _teardown_pot_data(self, pot_id: str) -> bool | None:
        """Clear both data planes; report what the resource store actually did.

        ``None`` when no resource store is wired — the caller must be able to
        tell "nothing to purge" from "purged nothing", because it reports the
        answer to a user.
        """
        self.backend.mutation.reset_pot(pot_id)
        if self.resources is None:
            return None
        return bool(self.resources.purge_pot(pot_id))

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

    def list_repo_sources(self) -> list[PotRepoSource]:
        return [_repo_source(r) for r in self.store.list_repo_sources()]

    def source_status(self, *, pot_id: str, source_id: str) -> SourceInfo:
        for row in self.store.list_sources(pot_id=pot_id):
            if row.get("source_id") == source_id:
                return _source(row)
        raise PotNotFound(f"No source '{source_id}' in pot '{pot_id}'.")

    def remove_source(self, *, pot_id: str, source_id: str) -> None:
        # Registration only. A source's ``location`` is not a foreign key into
        # the resource store (documents carry a free-form ``source_ref`` URI),
        # so remove cannot know which documents — if any — came from it. Purge
        # documents with ``resource rm``; wipe a pot with ``pot reset``.
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


def _repo_source(row: dict) -> PotRepoSource:
    return PotRepoSource(
        pot_id=row["pot_id"],
        pot_name=row.get("pot_name", row["pot_id"]),
        name=row.get("name", row.get("location", "")),
        location=row.get("location"),
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

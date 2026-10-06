"""``LocalPotManagementService`` — control plane over a local pot store.

Wraps :class:`LocalPotStore` (flat-file persistence) and reports backend
readiness from the wired ``GraphBackend``. The real control plane is the local
state DB; this proves the service boundary and the CLI wiring.

``archived`` is a terminal lifecycle state, not a display hint. The CLI clears a
pot's graph state before archiving it, so the flag is enforced wherever a pot
can be chosen: :meth:`LocalPotManagementService._require_live` guards
selection, rename, re-archive, source registration and repo-default binding,
and the store's ref resolution and repo→pot index leave archived pots out.

Pot names are unique among live pots and never equal a pot id, because refs
resolve against both.
"""

from __future__ import annotations

from dataclasses import dataclass

from potpie.pots.contracts import (
    PotAggregateStatus,
    PotInfo,
    PotRepoSource,
    SourceInfo,
)
from potpie.pots.local_store import LocalPotStore
from potpie.pots.resolution import ARCHIVED_POT_NEXT_ACTION, archived_pot_message
from potpie_context_engine.core.errors import (
    PotArchived,
    PotNameConflict,
    PotNotFound,
)
from potpie_context_engine.core.lifecycle import DONE, StepResult
from potpie_context_engine.core.ports.graph.backend import GraphBackend


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
        # No ``_require_name_free``: reuse by name is what makes ``setup``
        # re-runnable, and that reuse is itself what keeps names unique here.
        _require_usable_name(name)
        self._require_name_is_not_a_pot_id(name)
        return _pot(self.store.create(name=name, repo=repo, use=use))

    def use_pot(self, *, ref: str) -> PotInfo:
        target = self._require_live(ref, verb="be selected")
        row = self.store.use(ref=target.pot_id)
        if row is None:
            raise PotNotFound(f"No pot matching '{ref}'.")
        return _pot(row)

    def rename_pot(self, *, ref: str, new_name: str) -> PotInfo:
        target = self._require_live(ref, verb="be renamed")
        _require_usable_name(new_name)
        self._require_name_free(new_name, allow_pot_id=target.pot_id)
        self._require_name_is_not_a_pot_id(new_name, allow_pot_id=target.pot_id)
        row = self.store.rename(ref=target.pot_id, new_name=new_name)
        if row is None:
            raise PotNotFound(f"No pot matching '{ref}'.")
        return _pot(row)

    def archive_pot(self, *, ref: str) -> PotInfo:
        target = self._require_live(ref, verb="be archived")
        row = self.store.archive(ref=target.pot_id)
        if row is None:
            raise PotNotFound(f"No pot matching '{ref}'.")
        return _pot(row)

    # --- lifecycle guards ---------------------------------------------------
    def _require_live(self, ref: str, *, verb: str) -> PotInfo:
        """The pot a ref names, refusing an archived one by name.

        ``PotNotFound`` would be the wrong answer for an archived pot: it sends
        the operator to ``pot list`` to look for a pot that listing hides on
        purpose. The store's own resolution skips archived pots, so this is the
        lookup that sees one and can tell the two refusals apart.
        """
        row = self.store.find(ref=ref, include_archived=True)
        if row is None:
            raise PotNotFound(f"No pot matching '{ref}'.")
        pot = _pot(row)
        if pot.archived:
            raise PotArchived(
                archived_pot_message(pot, verb=verb),
                recommended_next_action=ARCHIVED_POT_NEXT_ACTION,
            )
        return pot

    def _require_name_free(self, name: str, *, allow_pot_id: str) -> None:
        """Refuse a rename onto a name another live pot already answers to."""
        owner = self.store.names_in_use().get(name)
        if owner is not None and owner != allow_pot_id:
            raise PotNameConflict(
                f"Another pot already uses the name '{name}' ({owner}).",
                recommended_next_action=(
                    "pick a different name, or rename the other pot first"
                ),
            )

    def _require_name_is_not_a_pot_id(
        self, name: str, *, allow_pot_id: str | None = None
    ) -> None:
        """Refuse a name that equals some pot's id.

        Refs resolve against ids first, so a pot named after another pot's id
        would leave that other pot unreachable by name.
        """
        if name in self.store.pot_ids() and name != allow_pot_id:
            raise PotNameConflict(
                f"'{name}' is another pot's id; a name that shadows an id makes "
                "every reference to that pot ambiguous.",
                recommended_next_action="pick a name that is not a pot id",
            )

    # --- sources ------------------------------------------------------------
    def add_source(
        self, *, pot_id: str, kind: str, location: str, name: str | None = None
    ) -> SourceInfo:
        # A source registered into an archived pot is a row nothing will ever
        # route to, reported as a successful binding.
        self._require_live(pot_id, verb="take new sources")
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
        self.store.remove_source(pot_id=pot_id, source_id=source_id)

    # --- repo-local routing defaults ----------------------------------------
    def repo_default(self, *, repo: str) -> str | None:
        """The repo's default pot, unless that pot has since been archived.

        A stored pointer at an archived pot is stale: honouring it would route
        every repo-scoped read and write into a pot whose graph state was
        cleared, and the reads would come back empty rather than failing.
        """
        pot_id = self.store.repo_default(repo=repo)
        if not pot_id:
            return None
        row = self.store.find(ref=pot_id, include_archived=True)
        if row is None or row.get("archived"):
            return None
        return pot_id

    def set_repo_default(self, *, repo: str, pot_id: str) -> None:
        if not any(p.pot_id == pot_id for p in self.list_pots()):
            raise PotNotFound(f"No pot matching '{pot_id}'.")
        self._require_live(pot_id, verb="be a repo default")
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
        # Absent on every row that is not a ``create`` answer; those make no
        # claim about creation either way.
        created=row.get("created"),
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


def _require_usable_name(name: str) -> None:
    """Refuse an empty or whitespace-only name: nobody can type it as a ref."""
    if not (name or "").strip():
        blank = ValueError(
            "A pot name cannot be empty or only whitespace; it is the ref "
            "'potpie pot use' and '--pot' resolve against."
        )
        # The CLI boundary renders ValueError as validation_error and reads the
        # repair off the instance.
        blank.recommended_next_action = "pass a name, e.g. 'my-project'"  # type: ignore[attr-defined]
        raise blank


__all__ = ["LocalPotManagementService"]

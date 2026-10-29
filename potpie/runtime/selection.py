"""Per-actor context selection over a pot catalog.

One resolution algorithm serves every deployment. The local runtime supplies a
catalog with one fixed actor (whoever runs the CLI or holds the daemon's
credential); a managed host supplies one backed by its grants, so an actor
only ever resolves pots it may reach, and a pot it has no grant on is
indistinguishable from one that does not exist.

The selector is resolved after the caller is authenticated (the Resource
Manager does that first), so an unauthenticated caller learns nothing about
which pots exist.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from potpie.cli.repo_location import repo_identity_key
from potpie.pots.contracts import PotInfo, PotRepoSource
from potpie.pots.resolution import (
    ARCHIVED_POT_NEXT_ACTION,
    archived_pot_message,
    is_archived,
    match_pot_ref,
)
from potpie.runtime.resource_manager import (
    AuthenticatedActor,
    ContextSelector,
    SelectionError,
)
from potpie_context_engine import ContextIdentity, Failure, Success


class ActorPotCatalog(Protocol):
    """The pots one actor can see, and that actor's routing pointers.

    Every method answers for ``actor`` only. A pot the actor holds no grant
    on must be absent from every answer, never present and then refused.
    """

    async def pots_for(self, actor: AuthenticatedActor) -> Sequence[PotInfo]:
        """Every pot the actor may reach, archived ones included."""
        ...

    async def active_for(self, actor: AuthenticatedActor) -> PotInfo | None:
        """The actor's active pot, if it has one."""
        ...

    async def repo_default_for(
        self, actor: AuthenticatedActor, repo: str
    ) -> str | None:
        """The pot id the actor routes this repo identity to, if set."""
        ...

    async def repo_sources_for(
        self, actor: AuthenticatedActor
    ) -> Sequence[PotRepoSource]:
        """Every repo source of the actor's live pots, joined to its pot."""
        ...


class CatalogSelectorResolver:
    """Resolve explicit, active and repository selectors for one actor.

    * ``explicit``: an exact id or name of a live pot; an archived pot only by
      its exact id (ids are never reused; a name may be).
    * ``active``: the actor's active pot.
    * ``repository``: the actor's default pot for the repo identity, else the
      one pot whose registered repo source matches it, else the active pot.
      Several matching pots resolve to the active one when it is among them
      and are otherwise ambiguous.
    """

    def __init__(self, catalog: ActorPotCatalog) -> None:
        self._catalog = catalog

    @property
    def catalog(self) -> ActorPotCatalog:
        return self._catalog

    async def resolve(
        self, selector: ContextSelector, *, actor: AuthenticatedActor
    ) -> Success[ContextIdentity] | Failure[SelectionError]:
        try:
            pots = tuple(await self._catalog.pots_for(actor))
            active = await self._catalog.active_for(actor)
        except Exception:  # noqa: BLE001 - any catalog failure is "cannot say"
            return Failure(
                SelectionError(
                    code="context_selection_unavailable",
                    message="Potpie could not read the pot catalog.",
                    recommended_next_action="check daemon readiness with 'potpie doctor'",
                    retry_posture="safe",
                )
            )

        if selector.kind == "explicit":
            return _explicit(pots, selector.value or "")
        if selector.kind == "active":
            if active is not None:
                return Success(ContextIdentity(active.pot_id))
            return Failure(no_active_pot_error())
        return await self._repository(actor, pots, active, selector.value or "")

    async def _repository(
        self,
        actor: AuthenticatedActor,
        pots: tuple[PotInfo, ...],
        active: PotInfo | None,
        repo: str,
    ) -> Success[ContextIdentity] | Failure[SelectionError]:
        try:
            default_pot_id = await self._catalog.repo_default_for(actor, repo)
        except Exception:  # noqa: BLE001 - a missing default is not fatal
            default_pot_id = None
        live_ids = {pot.pot_id for pot in pots if not is_archived(pot)}
        if default_pot_id and default_pot_id in live_ids:
            return Success(ContextIdentity(str(default_pot_id)))

        # One repo->pot index read, not one source listing per pot. Matching
        # stays here; one entry per pot, in pot order.
        try:
            index = tuple(await self._catalog.repo_sources_for(actor))
        except Exception:  # noqa: BLE001 - no index means no repo match
            index = ()
        matches: dict[str, str] = {}
        for row in index:
            if row.pot_id in matches or row.pot_id not in live_ids:
                continue
            if any(
                repo_identity_key(ref) == repo
                for ref in (row.name, row.location)
                if ref
            ):
                matches[row.pot_id] = row.pot_name

        if len(matches) == 1:
            return Success(ContextIdentity(next(iter(matches))))
        if len(matches) > 1:
            if active is not None and active.pot_id in matches:
                return Success(ContextIdentity(active.pot_id))
            names = ", ".join(f"{name} ({pot_id})" for pot_id, name in matches.items())
            return Failure(
                SelectionError(
                    code="ambiguous_pot",
                    message=f"Current repo is registered in multiple pots: {names}.",
                    recommended_next_action=(
                        "pick one with '--pot <id-or-name>' or set it active with "
                        "'potpie pot use <id-or-name>'"
                    ),
                )
            )
        if active is not None:
            return Success(ContextIdentity(active.pot_id))
        return Failure(no_active_pot_error())


def _explicit(
    pots: tuple[PotInfo, ...], ref: str
) -> Success[ContextIdentity] | Failure[SelectionError]:
    live, archived = match_pot_ref(pots, ref)
    if live is not None:
        return Success(ContextIdentity(live.pot_id))
    # An exact id always names its pot, archived or not: ids are never reused,
    # and clearing a retired pot's graph state needs to reach it. What may run
    # against it is the authorizer's decision. A name never resolves to an
    # archived pot, because a live pot may reuse that name.
    if archived is not None and archived.pot_id == ref:
        return Success(ContextIdentity(archived.pot_id))
    if archived is not None:
        return Failure(
            SelectionError(
                code="pot_archived",
                message=archived_pot_message(archived),
                recommended_next_action=ARCHIVED_POT_NEXT_ACTION,
            )
        )
    return Failure(
        SelectionError(
            code="pot_not_found",
            message=f"No pot matching '{ref}'.",
            recommended_next_action="run 'potpie pot list'",
        )
    )


def no_active_pot_error() -> SelectionError:
    return SelectionError(
        code="no_active_pot",
        message=(
            "No active pot, and the current repo is not registered as a source "
            "in any pot."
        ),
        recommended_next_action=(
            "run 'potpie setup', or create a pot with 'potpie pot create <name> "
            "--use' and register this repo with 'potpie source add repo .'"
        ),
    )


__all__ = [
    "ActorPotCatalog",
    "CatalogSelectorResolver",
    "no_active_pot_error",
]

"""Pot-ref and repo→pot resolution rules shared by the CLI and the engine.

Two places pick a pot for a command: the CLI's ``resolve_pot_scope`` and the
typed engine's ``LocalContextSelectorResolver``. They must agree on two rules,
so the rules live here once:

- An archived pot never answers a ref that a live pot answers, and a ref that
  names only an archived pot is refused as archived rather than as missing.
- The repo→pot index (every live pot's repo sources, joined to their pot) is
  read in one control-plane call. Walking pot by pot is only the fallback for
  a pot service that does not serve the index.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from potpie.pots.contracts import PotRepoSource

#: The repair for every archived-pot refusal. The pot is listed only on
#: request and its graph state is gone, so the way forward is a new pot.
ARCHIVED_POT_NEXT_ACTION = (
    "see it with 'potpie pot list --archived', or start a new pot with "
    "'potpie pot create <name> --use'"
)


def is_archived(pot: Any) -> bool:
    """True only for a pot that says it is archived (``PotInfo.archived``)."""
    return getattr(pot, "archived", False) is True


def match_pot_ref(pots: Iterable[Any], ref: str) -> tuple[Any | None, Any | None]:
    """``(live, archived)`` for a pot ref; at most one of the two is set.

    Within each group an id match beats a name match. A live match always
    wins, so a retired pot cannot shadow a live pot that reuses its name.
    """
    rows = list(pots)

    def first(group: list[Any]) -> Any | None:
        for pot in group:
            if getattr(pot, "pot_id", None) == ref:
                return pot
        for pot in group:
            if getattr(pot, "name", None) == ref:
                return pot
        return None

    live = first([pot for pot in rows if not is_archived(pot)])
    if live is not None:
        return live, None
    return None, first([pot for pot in rows if is_archived(pot)])


def archived_pot_message(pot: Any, *, verb: str = "be used as a target") -> str:
    pot_id = getattr(pot, "pot_id", "")
    name = getattr(pot, "name", None) or pot_id
    return (
        f"Pot '{name}' ({pot_id}) is archived, so it cannot {verb}. "
        "Archiving cleared its graph state."
    )


def repo_source_index(pots: Any) -> list[PotRepoSource]:
    """Every live pot's repo sources, joined to their pot.

    One ``list_repo_sources`` call when the service serves the index. A service
    that predates it is walked pot by pot, skipping archived pots, which yields
    the same rows in the same order.
    """
    serve = getattr(pots, "list_repo_sources", None)
    if callable(serve):
        return list(serve())
    rows: list[PotRepoSource] = []
    for pot in pots.list_pots():
        if is_archived(pot):
            continue
        try:
            sources = pots.list_sources(pot_id=pot.pot_id)
        except Exception:  # noqa: BLE001, S112 - one unreadable pot must not mask others.
            continue
        rows.extend(
            PotRepoSource(
                pot_id=pot.pot_id,
                pot_name=getattr(pot, "name", None) or pot.pot_id,
                name=str(getattr(source, "name", "") or ""),
                location=getattr(source, "location", None),
            )
            for source in sources
            if getattr(source, "kind", None) == "repo"
        )
    return rows


__all__ = [
    "ARCHIVED_POT_NEXT_ACTION",
    "archived_pot_message",
    "is_archived",
    "match_pot_ref",
    "repo_source_index",
]

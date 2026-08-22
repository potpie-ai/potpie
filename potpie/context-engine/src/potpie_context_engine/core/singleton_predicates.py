"""Singleton-predicate registry — the deterministic edge-contradiction path.

Rebuild plan P2 (F3): some predicates admit only one live object for a
given subject. Once a service has an owner, the new owner *supersedes*
the old one rather than corroborating it; the old ``OWNED_BY`` is no
longer true even if its source text never said "supersedes". This is
the deterministic-supersede pattern — no LLM call needed.

The complement path is the LLM-driven ``dedupe_edges.resolve_edge``
port (per the rebuild plan's Phase 2). Deferred; the proper POC's F3
finding shows the deterministic path handles the structured cases
(OWNED_BY, DEPLOYED_TO, OF_SERVICE, CURRENT_VERSION) which are the
bulk of contradiction events.

When the canonical writer applies a new ``:RELATES_TO`` claim for a
singleton predicate from a deterministic source, the writer asks this
registry whether to stamp ``invalid_at`` on any prior live claim with
the *same subject and same singleton predicate but different object*.

API contract:

- :func:`is_singleton_predicate(name)` — boolean check.
- :func:`register_singleton(name)` — add a predicate to the registry.
- :func:`all_singletons()` — read-only view.

The registry is module-level state; ontology refinements in P3 will
add a declarative ``singleton: bool`` flag on ``EdgeTypeSpec`` and the
registry will be rebuilt from there. Until then, the explicit list
below is the single source of truth.
"""

from __future__ import annotations

from typing import Iterable


# Initial singleton list — singleton in the sense that *one live object
# per subject* is the intended cardinality. Multi-source corroboration
# on the *same* object is still fine; only object-change triggers the
# supersession.
_DEFAULT_SINGLETONS: frozenset[str] = frozenset(
    {
        # Ownership — one current owner per service/component.
        "OWNED_BY",
        # Topology cardinality decisions.
        "DEPLOYED_TO",
        "OF_SERVICE",
        # Versioning / current-state attributes.
        "CURRENT_VERSION",
        # Service↔Datastore primary binding (one primary store per service).
        "PRIMARY_STORE",
    }
)


_singletons: set[str] = set(_DEFAULT_SINGLETONS)


def is_singleton_predicate(name: str | None) -> bool:
    """Return True when the predicate's ``(subject, predicate)`` is singleton."""
    if not name:
        return False
    return name in _singletons


def register_singleton(name: str) -> None:
    """Mark a predicate name as singleton; idempotent."""
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"singleton predicate name must be a non-empty string: {name!r}"
        )
    _singletons.add(name)


def unregister_singleton(name: str) -> None:
    """Remove a predicate from the singleton registry (test helper)."""
    _singletons.discard(name)


def all_singletons() -> frozenset[str]:
    """Read-only snapshot of registered singleton predicates."""
    return frozenset(_singletons)


def replace_singletons(names: Iterable[str]) -> None:
    """Reset the registry to exactly ``names``. Used by ontology bootstrap (P3)."""
    new = {str(n) for n in names if isinstance(n, str) and n}
    _singletons.clear()
    _singletons.update(new)


__all__ = [
    "all_singletons",
    "is_singleton_predicate",
    "register_singleton",
    "replace_singletons",
    "unregister_singleton",
]

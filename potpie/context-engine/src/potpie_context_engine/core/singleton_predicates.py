"""Registry of predicates with one live object per subject.

The ontology's ``singleton`` flags define the initial registry. A deterministic
claim with a different object supersedes the previous claim; repeated evidence
for the same object remains corroboration.
"""

from __future__ import annotations

from typing import Iterable


_singletons: set[str] | None = None


def _registry() -> set[str]:
    global _singletons
    if _singletons is None:
        # Lazy import avoids the ontology bootstrap cycle.
        from potpie_context_engine.core.ontology import SINGLETON_EDGE_TYPES

        _singletons = set(SINGLETON_EDGE_TYPES)
    return _singletons


def is_singleton_predicate(name: str | None) -> bool:
    """Return True when the predicate's ``(subject, predicate)`` is singleton."""
    if not name:
        return False
    return name in _registry()


def register_singleton(name: str) -> None:
    """Mark a predicate name as singleton; idempotent."""
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"singleton predicate name must be a non-empty string: {name!r}"
        )
    _registry().add(name)


def unregister_singleton(name: str) -> None:
    """Remove a predicate from the singleton registry (test helper)."""
    _registry().discard(name)


def all_singletons() -> frozenset[str]:
    """Read-only snapshot of registered singleton predicates."""
    return frozenset(_registry())


def replace_singletons(names: Iterable[str]) -> None:
    """Reset the registry to exactly ``names`` during ontology bootstrap."""
    global _singletons
    _singletons = {n for n in names if isinstance(n, str) and n}


__all__ = [
    "all_singletons",
    "is_singleton_predicate",
    "register_singleton",
    "replace_singletons",
    "unregister_singleton",
]

"""Value types for the ContextReader registry (Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ReaderCost:
    """Coarse-grained cost label so the router can respect ``mode``/``budget``."""

    label: str  # "cheap" | "medium" | "expensive"
    estimated_ms: int = 0


@dataclass(frozen=True, slots=True)
class ReaderCapability:
    """What a :class:`ContextReader` advertises to the registry.

    Routing is declarative: the router consults capability fields rather
    than hard-coding family names. Adding a reader does not require any
    edit to the router.
    """

    family: str
    description: str
    intents: frozenset[str] = frozenset()
    requires_scope: frozenset[str] = frozenset()
    cost: ReaderCost = ReaderCost(label="cheap")
    backend: str = "structural"
    # ``compat=True`` flags readers preserved for legacy callers (e.g.
    # ``pr_diff``); the router stamps the response so consumers can see
    # a deprecation hint without inspecting reader internals.
    compat: bool = False


@dataclass
class ReaderResult:
    """Result envelope a :class:`ContextReader` returns to the registry."""

    family: str
    result: Any = None
    count: int | None = None
    error: str | None = None
    fallback_reason: str | None = None
    compat: bool = False


@dataclass(frozen=True, slots=True)
class ReaderManifestEntry:
    """One entry in the registry manifest surfaced through ``context_status``."""

    family: str
    description: str
    intents: tuple[str, ...]
    requires_scope: tuple[str, ...]
    cost: str
    backend: str


@dataclass(frozen=True, slots=True)
class RouterFallback:
    """Reason the router skipped or replaced a requested family."""

    family: str
    reason: str
    detail: str = ""


__all__ = [
    "ReaderCapability",
    "ReaderCost",
    "ReaderManifestEntry",
    "ReaderResult",
    "RouterFallback",
]

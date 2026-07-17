"""``GraphAnalyticsPort`` — aggregate/quality projection of a ``GraphBackend``.

Backs ``potpie graph status`` and ``graph repair``: entity/claim counts,
freshness windows, quality signals, and rebuild/repair of derived indexes.
A rebuildable projection — none of this is a source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class RepairReport:
    """Outcome of a ``repair`` run (e.g. semantic-index rebuild)."""

    pot_id: str
    targets: tuple[str, ...]
    repaired: Mapping[str, int] = field(default_factory=dict)
    detail: str | None = None


class GraphAnalyticsPort(Protocol):
    """Aggregates, freshness, quality, and repair over a pot's graph."""

    def counts(self, pot_id: str) -> Mapping[str, int]:
        """Entity/claim/predicate counts for the pot."""
        ...

    def freshness(self, pot_id: str) -> Mapping[str, Any]:
        """Freshness windows (oldest/newest claim, per-source last write)."""
        ...

    def quality(self, pot_id: str) -> Mapping[str, Any]:
        """Quality signals (open conflicts, orphan rate, status)."""
        ...

    def repair(self, pot_id: str, *, targets: Sequence[str] = ()) -> RepairReport:
        """Rebuild derived indexes/projections. Empty ``targets`` repairs all."""
        ...


__all__ = ["GraphAnalyticsPort", "RepairReport"]

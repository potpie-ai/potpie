"""``GraphSnapshotPort`` — portable export/import of a pot's graph.

Backs ``potpie graph export/import`` and the ``cloud push/pull`` snapshot move
between local and managed profiles. A snapshot is a self-describing dump of the
canonical claim store for one pot; importing into a different backend profile
rebuilds the projections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    """Metadata describing an exported/imported snapshot."""

    pot_id: str
    location: str  # file path or URI the snapshot was written to / read from
    format_version: str = "1"
    entity_count: int = 0
    claim_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


class GraphSnapshotPort(Protocol):
    """Portable pot snapshot export/import."""

    def export(self, *, pot_id: str, destination: str) -> SnapshotManifest:
        """Write a portable snapshot of the pot's graph to ``destination``."""
        ...

    def import_(self, *, pot_id: str, source: str) -> SnapshotManifest:
        """Load a snapshot from ``source`` into ``pot_id``, rebuilding
        projections for this backend profile."""
        ...


__all__ = ["GraphSnapshotPort", "SnapshotManifest"]

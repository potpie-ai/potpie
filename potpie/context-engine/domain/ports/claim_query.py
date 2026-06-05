"""Claim-query port for the P9 use-case readers.

Each P9 reader (CodingPreferences, InfraTopology, Timeline, PriorBugs)
needs a small, well-defined query surface against the Position B
:RELATES_TO claim store. Rather than coupling every reader to Neo4j /
Neo4j directly, we expose a port:

    ClaimQueryPort.find_claims(...) -> list[ClaimRow]

Implementations:

- Production: a Neo4j adapter that runs Cypher over :RELATES_TO edges.
- Tests: an in-memory dict-backed fake.

The port deliberately speaks in canonical claim properties (the same
ones the canonical writer emits): ``predicate``, ``subject_key``,
``object_key``, ``valid_at``, ``invalid_at``, ``evidence_strength``,
``source_system``, ``fact``, ``properties``. Readers translate from
this row shape into :class:`domain.ranking.Candidate` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class ClaimRow:
    """One canonical :RELATES_TO edge as the readers see it."""

    pot_id: str
    predicate: str
    subject_key: str
    object_key: str
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    evidence_strength: str = "stated"
    source_system: str | None = None
    source_ref: str | None = None
    fact: str | None = None
    properties: Mapping[str, Any] = field(default_factory=dict)
    fact_embedding: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class ClaimQueryFilter:
    """Filter spec the reader hands to the port.

    All fields are optional; the implementation interprets an empty
    field as "no filter on this axis". The port is responsible for
    short-circuiting hopeless filters (e.g. empty ``predicate_in`` plus
    ``include_invalidated=False`` always returns an empty list).
    """

    pot_id: str
    predicate_in: tuple[str, ...] = ()
    subject_key_in: tuple[str, ...] = ()
    object_key_in: tuple[str, ...] = ()
    subject_label: str | None = None  # filter by Entity label
    object_label: str | None = None
    valid_at_after: datetime | None = None
    valid_at_before: datetime | None = None
    include_invalidated: bool = False
    as_of: datetime | None = None
    source_system_in: tuple[str, ...] = ()
    limit: int | None = None
    # When set, the port runs a native vector query and returns
    # candidates ordered by similarity. Cosine distance scores are
    # stamped onto ``ClaimRow.properties["semantic_similarity"]``.
    fact_query: str | None = None


class ClaimQueryPort(Protocol):
    """Read-only query surface against canonical claim edges."""

    def find_claims(self, filter_: ClaimQueryFilter) -> list[ClaimRow]: ...

    def entity_labels(
        self, *, pot_id: str, entity_keys: Iterable[str]
    ) -> Mapping[str, tuple[str, ...]]:
        """Bulk-lookup labels for the given entity keys.

        Readers use this to filter results by canonical label
        (``Service`` / ``Person`` / ``Decision`` / …) without making
        per-row round-trips.
        """
        ...


__all__ = ["ClaimQueryFilter", "ClaimQueryPort", "ClaimRow"]

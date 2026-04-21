"""Typed structural graph mutations (reconciliation domain)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ProvenanceRef:
    """Link mutations back to a source event or produced episode."""

    pot_id: str
    source_event_id: str
    episode_uuid: str | None = None


@dataclass(slots=True)
class EntityUpsert:
    """Upsert a structural entity by deterministic key."""

    entity_key: str
    labels: tuple[str, ...]
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EdgeUpsert:
    """Upsert an edge by deterministic relationship identity.

    ``properties`` may include ``lifecycle_status`` for episodic-style edges
    (see ``domain.ontology.LifecycleStatus``).
    """

    edge_type: str
    from_entity_key: str
    to_entity_key: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QualityIssueCreate:
    """Structural reconciliation hook for ``QualityIssue`` nodes (see Phase 7 ontology)."""

    issue_uuid: str
    code: str
    kind: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EdgeDelete:
    """Delete an edge by deterministic identity."""

    edge_type: str
    from_entity_key: str
    to_entity_key: str


@dataclass(slots=True)
class InvalidationOp:
    """Mark a fact or entity invalidated with provenance.

    Either target_entity_key or target_edge must be set.
    When superseded_by_key is provided a SUPERSEDES edge is created from the
    new entity to the invalidated one; the invalidated node/edge gets valid_to
    stamped rather than being deleted, preserving the audit trail.
    """

    target_entity_key: str | None
    target_edge: tuple[str, str, str] | None
    reason: str
    superseded_by_key: str | None = None
    valid_to: str | None = None


@dataclass(slots=True)
class EpisodicSupersessionRecord:
    """Audit row when auto-supersede stamps ``invalid_at`` on a Graphiti entity edge."""

    group_id: str
    superseded_edge_uuid: str
    superseding_edge_uuid: str
    predicate_family: str
    reason: str = "predicate_family_contradiction"

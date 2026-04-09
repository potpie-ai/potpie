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
    """Upsert an edge by deterministic relationship identity."""

    edge_type: str
    from_entity_key: str
    to_entity_key: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EdgeDelete:
    """Delete an edge by deterministic identity."""

    edge_type: str
    from_entity_key: str
    to_entity_key: str


@dataclass(slots=True)
class InvalidationOp:
    """Mark a fact or entity invalidated with provenance."""

    target_entity_key: str | None
    target_edge: tuple[str, str, str] | None
    reason: str
    """Either entity_key or (edge_type, from_key, to_key) should be set."""

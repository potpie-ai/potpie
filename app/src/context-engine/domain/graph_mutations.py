"""Typed structural graph mutations (reconciliation domain)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class ProvenanceRef:
    """Link mutations back to a source event and the agent that produced them.

    This is the first-class provenance contract per
    ``docs/context-graph/implementation-next-steps.md``. Every important
    fact (entity, edge, invalidation) carries these fields so consumers
    can answer *where did this come from, when was it observed, when was
    it last written, how confident is it, and who produced it*.

    All fields after ``pot_id`` and ``source_event_id`` are optional so
    callers can construct partial provenance during migration; the
    mutation applier is responsible for stamping whatever is present.
    """

    pot_id: str
    source_event_id: str
    episode_uuid: str | None = None
    source_system: str | None = None
    source_kind: str | None = None
    source_ref: str | None = None
    event_occurred_at: datetime | None = None
    event_received_at: datetime | None = None
    graph_updated_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    confidence: float | None = None
    created_by_agent: str | None = None
    reconciliation_run_id: str | None = None

    def to_properties(self) -> dict[str, Any]:
        """Render provenance as ``prov_*`` Neo4j property keys.

        Only fields that are set are emitted so we don't overwrite
        existing properties with ``None`` during partial updates.
        """
        out: dict[str, Any] = {
            "prov_pot_id": self.pot_id,
            "prov_source_event_id": self.source_event_id,
        }
        if self.episode_uuid:
            out["prov_episode_uuid"] = self.episode_uuid
        if self.source_system:
            out["prov_source_system"] = self.source_system
        if self.source_kind:
            out["prov_source_kind"] = self.source_kind
        if self.source_ref:
            out["prov_source_ref"] = self.source_ref
        if self.event_occurred_at is not None:
            out["prov_event_occurred_at"] = self.event_occurred_at.isoformat()
        if self.event_received_at is not None:
            out["prov_event_received_at"] = self.event_received_at.isoformat()
        if self.graph_updated_at is not None:
            out["prov_graph_updated_at"] = self.graph_updated_at.isoformat()
        if self.valid_from is not None:
            out["prov_valid_from"] = self.valid_from.isoformat()
        if self.valid_to is not None:
            out["prov_valid_to"] = self.valid_to.isoformat()
        if self.confidence is not None:
            out["prov_confidence"] = float(self.confidence)
        if self.created_by_agent:
            out["prov_created_by_agent"] = self.created_by_agent
        if self.reconciliation_run_id:
            out["prov_reconciliation_run_id"] = self.reconciliation_run_id
        return out


@dataclass(frozen=True, slots=True)
class ProvenanceContext:
    """Caller-supplied context threaded into ``apply_plan``.

    Carries the optional provenance fields the ``ReconciliationPlan``
    cannot reconstruct on its own (source event times, resolved source
    ref/kind, agent identity, reconciliation run id). Combined with
    ``plan.event_ref`` and ``plan.confidence`` this produces a full
    :class:`ProvenanceRef` to stamp onto every mutation.
    """

    source_kind: str | None = None
    source_ref: str | None = None
    event_occurred_at: datetime | None = None
    event_received_at: datetime | None = None
    created_by_agent: str | None = None
    reconciliation_run_id: str | None = None


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

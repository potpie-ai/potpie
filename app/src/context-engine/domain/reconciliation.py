"""Reconciliation requests, plans, and results.

Post-refactor the graph layer has a single writer; the agent emits a
constrained mutation plan and ``apply_reconciliation_plan`` applies it
directly. There is no episodic narrative tier — plans are
entity/edge upserts, deletes, and invalidations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from domain.context_events import ContextEvent, EventRef
from domain.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
)


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """Pointer to evidence used by the planner."""

    kind: str
    ref: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReconciliationRequest:
    """Input to ``ReconciliationAgentPort``."""

    event: ContextEvent
    pot_id: str
    repo_name: str
    prior_attempts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ReconciliationPlan:
    """Constrained mutation plan produced by an agent or deterministic planner.

    A plan is a typed bundle of graph mutations targeting specific entities
    and edges. The agent inspects the event, decides which parts of the
    graph to touch, and emits this plan; ``apply_reconciliation_plan``
    executes the four mutation kinds in order against the single writer.
    """

    event_ref: EventRef
    summary: str
    entity_upserts: list[EntityUpsert] = field(default_factory=list)
    edge_upserts: list[EdgeUpsert] = field(default_factory=list)
    edge_deletes: list[EdgeDelete] = field(default_factory=list)
    invalidations: list[InvalidationOp] = field(default_factory=list)
    evidence: list[EvidenceRef] = field(default_factory=list)
    confidence: float | None = None
    warnings: list[str] = field(default_factory=list)
    ontology_downgrades: list[dict[str, str]] = field(default_factory=list)
    """Populated when soft ontology downgrade runs (API surface); not persisted on plan slices."""


@dataclass(slots=True)
class MutationSummary:
    """Counts applied in one reconciliation run."""

    entity_upserts_applied: int = 0
    edge_upserts_applied: int = 0
    edge_deletes_applied: int = 0
    invalidations_applied: int = 0
    stamp_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class ReconciliationResult:
    """Outcome of ``apply_reconciliation_plan`` or full ``reconcile_event``.

    ``mutation_id`` is a per-apply UUID stamped onto every mutation as
    write provenance — it lets readers trace any edge/entity back to the
    single apply that produced it.
    """

    ok: bool
    mutation_id: str
    mutation_summary: MutationSummary
    error: str | None = None
    reconciliation_errors: list[dict[str, str]] = field(default_factory=list)
    """Structured validation failures (entity + issue) when ``ok`` is false."""
    downgrades: list[dict[str, str]] = field(default_factory=list)
    """Ontology soft-fail rewrites applied (sync reconcile / apply)."""


# Phase A alias (INGESTION_ASYNC_PLAN — Ingestion Agent terminology)
IngestionPlan = ReconciliationPlan

"""Reconciliation requests, plans, and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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
class EpisodeDraft:
    """One episodic write to apply after validation."""

    name: str
    episode_body: str
    source_description: str
    reference_time: datetime


@dataclass(slots=True)
class ReconciliationRequest:
    """Input to ``ReconciliationAgentPort``."""

    event: ContextEvent
    pot_id: str
    repo_name: str
    prior_attempts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ReconciliationPlan:
    """Constrained mutation plan produced by an agent or deterministic planner."""

    event_ref: EventRef
    summary: str
    episodes: list[EpisodeDraft]
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

    episodes_written: int = 0
    entity_upserts_applied: int = 0
    edge_upserts_applied: int = 0
    edge_deletes_applied: int = 0
    invalidations_applied: int = 0
    stamp_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class ReconciliationResult:
    """Outcome of ``apply_reconciliation_plan`` or full ``reconcile_event``."""

    ok: bool
    episode_uuids: list[str | None]
    mutation_summary: MutationSummary
    error: str | None = None
    reconciliation_errors: list[dict[str, str]] = field(default_factory=list)
    """Structured validation failures (entity + issue) when ``ok`` is false."""
    downgrades: list[dict[str, str]] = field(default_factory=list)
    """Ontology soft-fail rewrites applied (sync reconcile / apply)."""


# Phase A alias (INGESTION_ASYNC_PLAN — Ingestion Agent terminology)
IngestionPlan = ReconciliationPlan

"""The internal, backend-bound structural write tier: :class:`MutationBatch`.

A ``MutationBatch`` is a typed, atomic bundle of graph mutations
(entity/edge upserts, deletes, invalidations) applied through the one write
door (``GraphMutationPort.apply`` → ``apply_mutation_batch``). It is *not* the
agent contract — the public, agent-facing tier is
:mod:`domain.semantic_mutations`, which lowers into a ``MutationBatch``.

Naming (Graph V1.5 Step 5a): the batch reconciles nothing — it is a mutation
batch, so the canonical name is ``MutationBatch`` (``ReconciliationPlan`` is
kept as a thin back-compat alias for existing importers). "Reconciliation" is
reserved for genuine source-state convergence (the ledger ``reconcile()`` path
and V2's ``reconcile_snapshot``).

``event_ref`` and ``summary`` are **optional**: they force an event frame onto
non-event writes (a recorded preference is not an event), so the lowerer no
longer fabricates an ``EventRef`` to record one. Write provenance flows through
:class:`~domain.graph_mutations.ProvenanceContext`, not through required batch
fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from potpie_context_engine.domain.context_events import EventRef
from potpie_context_engine.domain.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
)

# Step 11 parked the agentic-only types into ``domain.llm_reconciliation`` so
# this module holds only the live structural write tier. Re-exported here as a
# thin back-compat shim for one iteration; ``MutationBatch.evidence`` keeps its
# ``EvidenceRef`` element type through this import.
from potpie_context_engine.domain.llm_reconciliation import (  # noqa: F401  (re-export shim)
    EvidenceRef,
    ReconciliationRequest,
)


@dataclass(slots=True)
class MutationBatch:
    """A typed, atomic batch of graph mutations applied via the one write door.

    The semantic mutation lowerer decides which entities/edges to touch and
    emits this batch; ``apply_mutation_batch`` executes the mutation kinds in
    order against the single writer.

    ``event_ref`` / ``summary`` are optional — non-event writes leave them
    unset rather than fabricating an event frame.
    """

    event_ref: EventRef | None = None
    summary: str = ""
    entity_upserts: list[EntityUpsert] = field(default_factory=list)
    edge_upserts: list[EdgeUpsert] = field(default_factory=list)
    edge_deletes: list[EdgeDelete] = field(default_factory=list)
    invalidations: list[InvalidationOp] = field(default_factory=list)
    evidence: list[EvidenceRef] = field(default_factory=list)
    confidence: float | None = None
    warnings: list[str] = field(default_factory=list)
    ontology_downgrades: list[dict[str, str]] = field(default_factory=list)
    """Populated when soft ontology downgrade runs (API surface); not persisted on plan slices."""


# Back-compat alias — the public ``SemanticMutationPlan`` keeps its name;
# ``MutationBatch`` is the internal tier it lowers into. Existing importers of
# ``ReconciliationPlan`` keep working; new code uses ``MutationBatch``.
ReconciliationPlan = MutationBatch


@dataclass(slots=True)
class MutationSummary:
    """Counts applied in one mutation-batch apply."""

    entity_upserts_applied: int = 0
    edge_upserts_applied: int = 0
    edge_deletes_applied: int = 0
    invalidations_applied: int = 0
    stamp_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class MutationResult:
    """Outcome of ``apply_mutation_batch`` or full ``reconcile_event``.

    ``mutation_id`` is a per-apply UUID stamped onto every mutation as write
    provenance — it lets readers trace any edge/entity back to the single apply
    that produced it.
    """

    ok: bool
    mutation_id: str
    mutation_summary: MutationSummary
    error: str | None = None
    reconciliation_errors: list[dict[str, str]] = field(default_factory=list)
    """Structured validation failures (entity + issue) when ``ok`` is false."""
    downgrades: list[dict[str, str]] = field(default_factory=list)
    """Ontology soft-fail rewrites applied (sync reconcile / apply)."""


# Back-compat alias.
ReconciliationResult = MutationResult


__all__ = [
    "EvidenceRef",
    "MutationBatch",
    "MutationResult",
    "MutationSummary",
    "ReconciliationPlan",
    "ReconciliationRequest",
    "ReconciliationResult",
]

"""``DefaultGraphService`` — data plane over a ``GraphBackend``.

This is the real seam, with a thin body behind it. ``resolve``/``search`` run the
existing one-read-trunk (:class:`ReadOrchestrator`) over the backend's
``claim_query`` port; ``record`` lowers a durable record into a mutation plan and
applies it through the backend's ``mutation`` port. With the ``in_memory``
backend this gives a working resolve → record → resolve round trip — enough to
exercise the architecture end to end.

Record lowering maps ``record_type`` → its ontology predicate via
:data:`domain.ontology.RECORD_TYPES` (e.g. ``preference`` →
``POLICY_APPLIES_TO``), so a recorded preference surfaces in the
``coding_preferences`` reader; free-form types fall back to ``RELATED_TO``.

What is deliberately shallow (and marked TODO):

- ``mode`` (fast/balanced/verify/deep) is threaded into request metadata but
  does not yet change retrieval depth.
- The full async ingestion/reconciliation pipeline (validation, dedup,
  provenance, structured payload schemas) is bypassed; production ``record``
  should route through it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.services.read_orchestrator import ReadOrchestrator
from domain.agent_context_port import (
    build_context_record_source_id,
    normalize_record_type,
)
from domain.agent_envelope import AgentEnvelope
from domain.context_events import EventRef
from domain.errors import CapabilityNotImplemented
from domain.graph_mutations import EdgeUpsert, EntityUpsert
from domain.ontology import record_type_spec
from domain.ports.agent_context import (
    RecordReceipt,
    RecordRequest,
    ResolveRequest,
    SearchRequest,
)
from domain.ports.graph.backend import GraphBackend
from domain.ports.services.graph_service import DataPlaneStatus
from domain.reconciliation import ReconciliationPlan

# Scope keys the readers compute overlap against (see
# ``application.readers.coding_preferences._normalise_scope_for_overlap``).
_CODE_SCOPE_KEYS = ("language", "framework", "repo", "service", "file_path", "audience")


@dataclass(slots=True)
class DefaultGraphService:
    """Data-plane service backed by a swappable ``GraphBackend``."""

    backend: GraphBackend
    _orchestrator: ReadOrchestrator = field(init=False)

    def __post_init__(self) -> None:
        # One read trunk over the backend's canonical claim store.
        self._orchestrator = ReadOrchestrator(claim_query=self.backend.claim_query)

    # --- reads --------------------------------------------------------------
    def resolve(self, request: ResolveRequest) -> AgentEnvelope:
        return self._orchestrator.resolve(
            pot_id=request.pot_id,
            intent=request.intent,
            query=request.task,
            scope=dict(request.scope),
            include=list(request.include) or None,
            exclude=list(request.exclude) or None,
            as_of=request.as_of,
            max_items=request.max_items,
            metadata={"mode": request.mode, "source_policy": request.source_policy},
        )

    def search(self, request: SearchRequest) -> AgentEnvelope:
        return self._orchestrator.resolve(
            pot_id=request.pot_id,
            intent="unknown",
            query=request.query,
            scope=dict(request.scope),
            include=list(request.include) or None,
            max_items=request.max_items,
            metadata={"mode": request.mode, "search": True},
        )

    # --- writes -------------------------------------------------------------
    def record(self, request: RecordRequest) -> RecordReceipt:
        record_type = normalize_record_type(request.record_type)
        source_id = build_context_record_source_id(
            record_type=record_type,
            summary=request.summary,
            scope=dict(request.scope),
            source_refs=list(request.source_refs),
            idempotency_key=request.idempotency_key,
        )
        plan = self._lower_record(request, record_type=record_type, source_id=source_id)
        result = self.backend.mutation.apply(plan, expected_pot_id=request.pot_id)
        applied = (
            result.mutation_summary.entity_upserts_applied
            + result.mutation_summary.edge_upserts_applied
        )
        return RecordReceipt(
            pot_id=request.pot_id,
            record_type=record_type,
            accepted=result.ok,
            record_id=source_id,
            status="recorded" if result.ok else "rejected",
            mutations_applied=applied,
            detail=result.error,
        )

    def _lower_record(
        self, request: RecordRequest, *, record_type: str, source_id: str
    ) -> ReconciliationPlan:
        """Lower a durable record onto its ontology predicate.

        Drives off :data:`domain.ontology.RECORD_TYPES`: the record's anchor
        entity gets ``spec.anchor_label`` and the claim predicate is
        ``spec.emits_predicate`` — so e.g. a ``preference`` lands as a
        ``POLICY_APPLIES_TO`` claim that the ``coding_preferences`` reader
        serves. Free-form types (``emits_predicate`` is ``None``) fall back to
        the generic ``RELATED_TO`` soft edge. The reader computes scope overlap
        from ``properties['code_scope']`` and matches the task phrase against
        ``fact``, so both are carried on the edge.
        """
        spec = record_type_spec(record_type)
        anchor_label = spec.anchor_label if spec else record_type.capitalize()
        predicate = spec.emits_predicate if spec and spec.emits_predicate else "RELATED_TO"

        scope = dict(request.scope)
        code_scope = {
            key: str(scope[key]) for key in _CODE_SCOPE_KEYS if scope.get(key)
        }
        record_key = f"{anchor_label.lower()}:{source_id}"
        target_key = (
            f"service:{scope['service']}"
            if scope.get("service")
            else f"repo:{scope.get('repo_name', request.pot_id)}"
        )
        return ReconciliationPlan(
            event_ref=EventRef(
                event_id=source_id, source_system="agent", pot_id=request.pot_id
            ),
            summary=request.summary,
            entity_upserts=[
                EntityUpsert(
                    entity_key=record_key,
                    labels=(anchor_label,),
                    properties={"summary": request.summary},
                )
            ],
            edge_upserts=[
                EdgeUpsert(
                    edge_type=predicate,
                    from_entity_key=record_key,
                    to_entity_key=target_key,
                    properties={
                        "fact": request.summary,
                        "record_type": record_type,
                        "source_system": "agent",
                        "source_ref": source_id,
                        "code_scope": code_scope,
                    },
                )
            ],
        )

    # --- status -------------------------------------------------------------
    def data_plane_status(self, pot_id: str) -> DataPlaneStatus:
        readiness = self.backend.mutation.readiness(pot_id)
        counts = _safe(lambda: dict(self.backend.analytics.counts(pot_id)), {})
        freshness = _safe(lambda: dict(self.backend.analytics.freshness(pot_id)), {})
        quality = _safe(lambda: dict(self.backend.analytics.quality(pot_id)), {})
        return DataPlaneStatus(
            pot_id=pot_id,
            backend_profile=self.backend.profile,
            backend_ready=readiness.ready,
            reader_backed_includes=tuple(sorted(self._orchestrator.backed_includes)),
            counts=counts,
            freshness=freshness,
            quality=quality,
            detail=readiness.detail,
        )


def _safe(fn, default):
    """Run a backend analytics call, swallowing not-implemented projections."""
    try:
        return fn()
    except CapabilityNotImplemented:
        return default


__all__ = ["DefaultGraphService"]

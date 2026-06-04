"""Convert LLM plan models to domain ``ReconciliationPlan``."""

from __future__ import annotations

from datetime import datetime, timezone

from domain.context_events import EventRef
from domain.graph_mutations import EdgeDelete, EdgeUpsert, EntityUpsert, InvalidationOp
from domain.reconciliation import EpisodeDraft, EvidenceRef, ReconciliationPlan

from adapters.outbound.reconciliation.llm_plan_schema import (
    LlmEpisodeDraft,
    LlmInvalidationOp,
    LlmReconciliationPlan,
)


def _episode(d: LlmEpisodeDraft) -> EpisodeDraft:
    rt = d.reference_time or datetime.now(timezone.utc)
    return EpisodeDraft(
        name=d.name,
        episode_body=d.episode_body,
        source_description=d.source_description,
        reference_time=rt,
    )


def _invalidation(op: LlmInvalidationOp) -> InvalidationOp:
    if op.target_entity_key:
        return InvalidationOp(
            target_entity_key=op.target_entity_key,
            target_edge=None,
            reason=op.reason,
        )
    if op.edge_type and op.from_entity_key and op.to_entity_key:
        return InvalidationOp(
            target_entity_key=None,
            target_edge=(op.edge_type, op.from_entity_key, op.to_entity_key),
            reason=op.reason,
        )
    raise ValueError("invalidation requires target_entity_key or edge_type+from+to")


def llm_plan_to_reconciliation_plan(
    llm: LlmReconciliationPlan | dict,
    *,
    event_ref: EventRef,
) -> ReconciliationPlan:
    llm = LlmReconciliationPlan.model_validate(llm)

    inv: list[InvalidationOp] = []
    for raw in llm.invalidations:
        inv.append(_invalidation(raw))

    return ReconciliationPlan(
        event_ref=event_ref,
        summary=llm.summary,
        episodes=[_episode(e) for e in llm.episodes],
        entity_upserts=[
            EntityUpsert(
                entity_key=u.entity_key,
                labels=tuple(u.labels),
                properties=dict(u.properties),
            )
            for u in llm.entity_upserts
        ],
        edge_upserts=[
            EdgeUpsert(
                edge_type=u.edge_type,
                from_entity_key=u.from_entity_key,
                to_entity_key=u.to_entity_key,
                properties=dict(u.properties),
            )
            for u in llm.edge_upserts
        ],
        edge_deletes=[
            EdgeDelete(
                edge_type=u.edge_type,
                from_entity_key=u.from_entity_key,
                to_entity_key=u.to_entity_key,
            )
            for u in llm.edge_deletes
        ],
        invalidations=inv,
        evidence=[
            EvidenceRef(kind=e.kind, ref=e.ref, metadata=dict(e.metadata)) for e in llm.evidence
        ],
        confidence=llm.confidence,
        warnings=list(llm.warnings),
    )

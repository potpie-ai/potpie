"""Lower semantic mutations into the structural :class:`MutationBatch` (Step 5).

The public ``SemanticMutation*`` tier never names ``EntityUpsert`` / ``EdgeUpsert``
/ Cypher; this lowerer is the one place that bridges the agent-facing semantic
ops to the backend-bound mutation batch, stamping the full V1.5 claim metadata
(truth, confidence, evidence, retrieval-card description, environment, validity,
provenance, contract/ontology version, deterministic ``claim_key``) on every
edge so a claim is future-readable without a migration.

It produces no ``EventRef`` for non-event writes — provenance flows through
:class:`ProvenanceContext`.
"""

from __future__ import annotations

from datetime import datetime, timezone

from domain.graph_contract import (
    DEFAULT_TRUTH_CLASS,
    GRAPH_CONTRACT_VERSION,
    ONTOLOGY_VERSION,
    SemanticMutationOp,
    edge_identity_key,
    evidence_strength_for_truth,
    make_claim_key,
    normalize_entity_key,
)
from domain.graph_entity_summary import compact_entity_summary
from domain.graph_mutations import (
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceContext,
)
from domain.ontology import ENTITY_TYPES
from domain.reconciliation import MutationBatch
from domain.semantic_mutations import (
    GraphEntityRef,
    LoweredOperation,
    SemanticMutation,
    SemanticMutationPlan,
    SemanticMutationRequest,
)

# Reverse map: canonical key prefix → entity label, for inferring a label when
# an entity ref omits ``type`` but its key carries a known prefix.
_PREFIX_TO_LABEL: dict[str, str] = {
    spec.key_prefix: label for label, spec in ENTITY_TYPES.items()
}

# Default truth class per event/claim op when the caller omits one.
_EVENT_TRUTH = "timeline_event"


def lower_semantic_request(
    request: SemanticMutationRequest, plan: SemanticMutationPlan
) -> SemanticMutationPlan:
    """Lower the plan's accepted ops into ``plan.batch`` + ``plan.provenance``.

    Mutates and returns ``plan``. Only ``accepted_ops`` are lowered (review /
    deferred / rejected ops never reach the write tier). Each accepted op's
    ``claim_keys`` are filled in so the receipt can report them.
    """
    batch = MutationBatch()
    provenance = _provenance_from_request(request)

    # Dedup entity upserts by key (a later op may re-reference the same entity).
    entity_by_key: dict[str, EntityUpsert] = {}
    new_accepted: list[LoweredOperation] = []

    for outcome in plan.accepted_ops:
        op = request.operations[outcome.op_index]
        claim_keys = _lower_op(
            op,
            request=request,
            batch=batch,
            entity_by_key=entity_by_key,
        )
        new_accepted.append(
            LoweredOperation(
                op_index=outcome.op_index,
                op=outcome.op,
                risk=outcome.risk,
                status=outcome.status,
                subgraph=outcome.subgraph,
                truth=outcome.truth,
                claim_keys=tuple(claim_keys),
            )
        )

    batch.entity_upserts = list(entity_by_key.values())
    plan.batch = batch
    plan.provenance = provenance
    plan.accepted_ops = tuple(new_accepted)
    return plan


# ---------------------------------------------------------------------------
# Per-op lowering
# ---------------------------------------------------------------------------


def _lower_op(
    op: SemanticMutation,
    *,
    request: SemanticMutationRequest,
    batch: MutationBatch,
    entity_by_key: dict[str, EntityUpsert],
) -> list[str]:
    name = op.op
    if name == SemanticMutationOp.upsert_entity.value:
        _ensure_entity(op.subject, entity_by_key, default_label="Observation")
        return []
    if name in (
        SemanticMutationOp.link_entities.value,
        SemanticMutationOp.assert_claim.value,
    ):
        return _lower_claim(op, request=request, batch=batch, entity_by_key=entity_by_key)
    if name == SemanticMutationOp.append_event.value:
        return _lower_event(op, request=request, batch=batch, entity_by_key=entity_by_key)
    if name in (
        SemanticMutationOp.end_relation_validity.value,
        SemanticMutationOp.retract_claim.value,
    ):
        _lower_retract(op, batch=batch)
        return []
    return []


def _lower_claim(
    op: SemanticMutation,
    *,
    request: SemanticMutationRequest,
    batch: MutationBatch,
    entity_by_key: dict[str, EntityUpsert],
) -> list[str]:
    subject = _ensure_entity(op.subject, entity_by_key, default_label="Observation")
    subgraph = op.subgraph or "memory"
    predicate = (op.predicate or "RELATED_TO").strip().upper()

    # Object is an entity ref or a literal value (→ Observation entity).
    if op.object is not None:
        obj = _ensure_entity(op.object, entity_by_key, default_label="Observation")
        object_key = obj.entity_key
        object_component = object_key
    else:
        # assert_claim with a value object: mint an Observation carrying the
        # value, never an authoritative fact from raw text.
        value = "" if op.value is None else str(op.value)
        obj_key = normalize_entity_key(f"observation:{_short(value)}")
        _ensure_entity(
            GraphEntityRef(
                key=obj_key,
                type="Observation",
                properties={"value": value},
                description=op.description,
            ),
            entity_by_key,
            default_label="Observation",
        )
        object_key = obj_key
        object_component = value or obj_key

    claim_key = make_claim_key(
        pot_id=request.pot_id,
        subgraph=subgraph,
        subject_key=subject.entity_key,
        predicate=predicate,
        object_component=object_component,
        discriminator=_discriminator(op, request),
        environment=op.environment,
    )
    props = _claim_properties(
        op,
        request=request,
        subgraph=subgraph,
        claim_key=claim_key,
        subject_key=subject.entity_key,
        object_key=object_key,
        predicate=predicate,
    )
    batch.edge_upserts.append(
        EdgeUpsert(
            edge_type=predicate,
            from_entity_key=subject.entity_key,
            to_entity_key=object_key,
            properties=props,
        )
    )
    return [claim_key]


def _lower_event(
    op: SemanticMutation,
    *,
    request: SemanticMutationRequest,
    batch: MutationBatch,
    entity_by_key: dict[str, EntityUpsert],
) -> list[str]:
    # Anchor an Activity entity (the caller's subject, or one minted from the
    # verb + occurred_at so re-submits are idempotent).
    if op.subject is not None:
        activity = _ensure_entity(op.subject, entity_by_key, default_label="Activity")
    else:
        anchor = f"{op.verb}:{op.occurred_at or ''}:{op.description or ''}"
        activity = _ensure_entity(
            GraphEntityRef(
                key=normalize_entity_key(f"activity:{_short(anchor)}"),
                type="Activity",
                properties={"verb_class": op.verb, "occurred_at": op.occurred_at},
                description=op.description,
            ),
            entity_by_key,
            default_label="Activity",
        )
    # Stamp verb/occurred_at on the activity.
    if op.verb:
        activity.properties.setdefault("verb_class", op.verb)
    if op.occurred_at:
        activity.properties.setdefault("occurred_at", op.occurred_at)

    subgraph = op.subgraph or "recent_changes"
    claim_keys: list[str] = []

    def edge(from_key: str, predicate: str, to_key: str) -> None:
        ck = make_claim_key(
            pot_id=request.pot_id,
            subgraph=subgraph,
            subject_key=from_key,
            predicate=predicate,
            object_component=to_key,
            discriminator=_discriminator(op, request),
            environment=op.environment,
        )
        props = _claim_properties(
            op,
            request=request,
            subgraph=subgraph,
            claim_key=ck,
            subject_key=from_key,
            object_key=to_key,
            predicate=predicate,
            truth_override=op.truth or _EVENT_TRUTH,
        )
        batch.edge_upserts.append(
            EdgeUpsert(
                edge_type=predicate,
                from_entity_key=from_key,
                to_entity_key=to_key,
                properties=props,
            )
        )
        claim_keys.append(ck)

    if op.actor is not None:
        actor = _ensure_entity(op.actor, entity_by_key, default_label="Person")
        edge(actor.entity_key, "PERFORMED", activity.entity_key)
    for target in op.targets:
        t = _ensure_entity(target, entity_by_key, default_label="Observation")
        edge(activity.entity_key, "TOUCHED", t.entity_key)
    for mention in op.mentions:
        m = _ensure_entity(mention, entity_by_key, default_label="Observation")
        edge(activity.entity_key, "MENTIONS", m.entity_key)

    return claim_keys


def _lower_retract(op: SemanticMutation, *, batch: MutationBatch) -> None:
    reason = op.reason or "retracted via semantic mutation"
    valid_to = op.valid_until or _now_iso()
    if (
        op.op == SemanticMutationOp.end_relation_validity.value
        and not (op.subject is not None and op.predicate and op.object is not None)
    ):
        return
    if op.subject is not None and op.predicate and op.object is not None:
        target_edge = (
            op.predicate.strip().upper(),
            normalize_entity_key(op.subject.key),
            normalize_entity_key(op.object.key),
        )
        batch.invalidations.append(
            InvalidationOp(
                target_entity_key=None,
                target_edge=target_edge,
                reason=reason,
                valid_to=valid_to,
            )
        )
    elif op.subject is not None:
        batch.invalidations.append(
            InvalidationOp(
                target_entity_key=normalize_entity_key(op.subject.key),
                target_edge=None,
                reason=reason,
                valid_to=valid_to,
            )
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_entity(
    ref: GraphEntityRef | None,
    entity_by_key: dict[str, EntityUpsert],
    *,
    default_label: str,
) -> EntityUpsert:
    """Upsert (and dedup) an entity for a ref, returning the EntityUpsert."""
    # Should not happen for validated ops; mint a placeholder Observation.
    resolved: GraphEntityRef = ref or GraphEntityRef(
        key=normalize_entity_key("observation:_missing"), type=default_label
    )
    key = normalize_entity_key(resolved.key)
    label = _label_for(resolved, default_label)
    existing = entity_by_key.get(key)
    props = dict(existing.properties) if existing else {}
    props.update({k: v for k, v in resolved.properties.items()})
    if resolved.name:
        props.setdefault("name", resolved.name)
    if resolved.summary:
        props["summary"] = resolved.summary
    if resolved.description:
        # The agent-authored retrieval card on the entity.
        props["description"] = resolved.description
    # Derive a compact summary only from authored material — never from the
    # entity key. A bare re-reference (key + type only) must not carry a
    # key-derived summary downstream, where it would overwrite an authored
    # summary already stored on the node. Writers fill key-derived fallbacks
    # for genuinely-new nodes without clobbering existing values.
    summary = compact_entity_summary(
        props.get("summary"),
        props.get("description"),
        props.get("title"),
        props.get("name"),
    )
    if summary:
        props["summary"] = summary
    else:
        props.pop("summary", None)
    upsert = EntityUpsert(entity_key=key, labels=(label,), properties=props)
    entity_by_key[key] = upsert
    return upsert


def _label_for(ref: GraphEntityRef, default_label: str) -> str:
    if ref.type and ref.type in ENTITY_TYPES:
        return ref.type
    prefix = normalize_entity_key(ref.key).partition(":")[0]
    return _PREFIX_TO_LABEL.get(prefix, default_label)


def _claim_properties(
    op: SemanticMutation,
    *,
    request: SemanticMutationRequest,
    subgraph: str,
    claim_key: str,
    subject_key: str,
    object_key: str,
    predicate: str,
    truth_override: str | None = None,
) -> dict[str, object]:
    truth = truth_override or op.truth or DEFAULT_TRUTH_CLASS
    evidence_strength = evidence_strength_for_truth(truth)
    fact = op.description or _synthesize_fact(subject_key, predicate, object_key)
    source_refs = [ev.source_ref for ev in op.evidence]
    evidence_dicts = [
        {"source_ref": ev.source_ref, "authority": ev.authority, **dict(ev.metadata)}
        for ev in op.evidence
    ]
    valid_at = op.valid_from or _now_iso()
    props: dict[str, object] = {
        "claim_key": claim_key,
        "subgraph": subgraph,
        "truth": truth,
        "evidence_strength": evidence_strength,
        "confidence": op.confidence if op.confidence is not None else 1.0,
        "fact": fact,
        "description": op.description,
        "source_refs": source_refs,
        "evidence": evidence_dicts,
        "source_system": request.created_by.surface or "agent",
        "source_ref": source_refs[0] if source_refs else (request.idempotency_key or claim_key),
        "valid_at": valid_at,
        "valid_from": valid_at,
        "observed_at": op.observed_at or _now_iso(),
        "created_by": _actor_dict(request),
        "graph_contract_version": request.graph_contract_version or GRAPH_CONTRACT_VERSION,
        "ontology_version": ONTOLOGY_VERSION,
        "idempotency_key": request.idempotency_key,
        "identity_key": list(
            edge_identity_key(subject_key, predicate, object_key, environment=op.environment)
        ),
    }
    if op.environment:
        props["environment"] = op.environment.strip().lower()
    if op.verb:
        props["verb_class"] = op.verb
    if op.occurred_at:
        props["occurred_at"] = op.occurred_at
    if op.valid_until:
        props["valid_until"] = op.valid_until
    # Carry the scope hierarchy for the readers (R4) when the subject/object
    # properties or op extras name a code scope.
    code_scope = _code_scope_for(op)
    if code_scope:
        props["code_scope"] = code_scope
    return props


def _code_scope_for(op: SemanticMutation) -> dict[str, str]:
    """Collect scope hints (repo/service/file_path/...) from op + subject props."""
    keys = ("language", "framework", "repo", "service", "file_path", "audience", "environment")
    out: dict[str, str] = {}
    sources = [dict(op.extra)]
    if op.subject is not None:
        sources.append(dict(op.subject.properties))
    for src in sources:
        for key in keys:
            val = src.get(key)
            if isinstance(val, str) and val.strip():
                out.setdefault(key, val.strip())
    if op.environment:
        out.setdefault("environment", op.environment.strip().lower())
    return out


def _provenance_from_request(request: SemanticMutationRequest) -> ProvenanceContext:
    return ProvenanceContext(
        source_ref=request.idempotency_key,
        created_by_agent=request.created_by.harness,
        actor_user_id=request.created_by.user or request.approved_by,
        actor_surface=request.created_by.surface,
        actor_client_name=request.created_by.harness,
    )


def _actor_dict(request: SemanticMutationRequest) -> dict[str, str]:
    out: dict[str, str] = {}
    if request.created_by.surface:
        out["surface"] = request.created_by.surface
    if request.created_by.harness:
        out["harness"] = request.created_by.harness
    if request.created_by.user:
        out["user"] = request.created_by.user
    if request.approved_by:
        out["approved_by"] = request.approved_by
    return out


def _discriminator(op: SemanticMutation, request: SemanticMutationRequest) -> str | None:
    if op.evidence:
        return op.evidence[0].source_ref
    return request.idempotency_key


def _synthesize_fact(subject_key: str, predicate: str, object_key: str) -> str:
    return f"{subject_key} {predicate.lower().replace('_', ' ')} {object_key}"


def _short(text: str, *, length: int = 12) -> str:
    import hashlib

    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:length]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["lower_semantic_request"]

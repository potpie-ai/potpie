"""Semantic-mutation validation + risk policy (Graph V1.5 Step 4).

Reject unsafe or ungrounded semantic mutations *before* they become a
structural :class:`~domain.reconciliation.MutationBatch`. Validation is purely a
domain concern — it reads the ontology and the contract constants, never a
backend — so a payload can be checked without touching a store.

Two outputs per request:

1. **Structured issues** — per-op ``error`` / ``warning`` findings.
2. **Risk classification + decision** — each op gets a :class:`MutationRisk`
   and a status (``accepted`` / ``review_required`` / ``deferred`` /
   ``rejected``); the batch's overall ``decision`` (``apply`` /
   ``review_required`` / ``rejected``) follows the risk policy and the caller's
   approval flags. A batch applies atomically: if any op cannot be auto-applied
   under the current flags, the whole batch returns ``review_required`` and
   nothing is written.
"""

from __future__ import annotations

from datetime import datetime
from typing import Mapping

from domain.graph_contract import (
    DEFERRED_OPS,
    EVIDENCE_REQUIRED_TRUTH_CLASSES,
    REVIEW_REQUIRED_OPS,
    MutationRisk,
    SemanticMutationOp,
    SourceAuthority,
    entity_key_matches_type,
    entity_key_prefix,
    is_known_op,
    is_source_authority,
    is_supported_contract_version,
    is_truth_class,
    normalize_entity_key,
)
from domain.ontology import EDGE_TYPES, ENTITY_TYPES, edge_spec
from domain.semantic_mutations import (
    GraphEntityRef,
    LoweredOperation,
    SemanticMutation,
    SemanticMutationPlan,
    SemanticMutationRequest,
    SemanticMutationValidationIssue,
)

# Ops that establish a durable typed claim (subject-predicate-object). These
# carry the "evidence-or-low-authority" and endpoint rules.
_CLAIM_OPS = frozenset(
    {SemanticMutationOp.link_entities.value, SemanticMutationOp.assert_claim.value}
)
# Ops that end / retract validity — state changes, classified medium.
_RETRACT_OPS = frozenset(
    {
        SemanticMutationOp.end_relation_validity.value,
        SemanticMutationOp.retract_claim.value,
    }
)
_PREFIX_TO_ENTITY_TYPE: dict[str, str] = {
    spec.key_prefix: label for label, spec in ENTITY_TYPES.items()
}
_STATE_PATCH_FIELDS = frozenset({"lifecycle_state", "status", "state"})


def validate_semantic_request(request: SemanticMutationRequest) -> SemanticMutationPlan:
    """Validate + risk-classify a parsed semantic mutation request.

    Returns a :class:`SemanticMutationPlan` with ``batch=None`` (lowering is a
    separate step) carrying issues, per-op classification, the overall risk and
    the batch decision.
    """
    issues: list[SemanticMutationValidationIssue] = []

    if not is_supported_contract_version(request.graph_contract_version):
        issues.append(
            SemanticMutationValidationIssue(
                code="unsupported_contract_version",
                message=(
                    f"graph_contract_version {request.graph_contract_version!r} "
                    f"is not supported by this build"
                ),
            )
        )
    if not request.pot_id.strip():
        issues.append(
            SemanticMutationValidationIssue(code="missing_pot_id", message="pot_id is required")
        )

    accepted: list[LoweredOperation] = []
    review: list[LoweredOperation] = []
    deferred: list[LoweredOperation] = []

    for index, op in enumerate(request.operations):
        op_issues, outcome = _validate_op(op, index)
        issues.extend(op_issues)
        if outcome.status == "deferred":
            deferred.append(outcome)
        elif outcome.status == "review_required":
            review.append(outcome)
        elif outcome.status == "accepted":
            accepted.append(outcome)
        # "rejected" ops contribute their errors but no applicable op.

    has_errors = any(i.is_error for i in issues)
    overall_risk = _overall_risk(accepted, review)
    decision = _decide(
        has_errors=has_errors,
        accepted=accepted,
        review=review,
        allow_review_required=request.allow_review_required,
        approved_by=request.approved_by,
    )

    # If the decision is review_required because of an unapproved medium op,
    # surface a non-blocking warning so the harness knows how to proceed.
    if decision == "review_required" and not has_errors and not review:
        issues.append(
            SemanticMutationValidationIssue(
                code="approval_required",
                message=(
                    "batch contains medium- or high-risk operations; re-submit with "
                    "--allow-review-required --approved-by <user-ref> to apply"
                ),
                severity="warning",
            )
        )

    return SemanticMutationPlan(
        pot_id=request.pot_id,
        ok=not has_errors,
        risk=overall_risk.value,
        decision=decision,
        issues=tuple(issues),
        accepted_ops=tuple(accepted),
        review_required_ops=tuple(review),
        deferred_ops=tuple(deferred),
    )


# ---------------------------------------------------------------------------
# Per-op validation
# ---------------------------------------------------------------------------


def _validate_op(
    op: SemanticMutation, index: int
) -> tuple[list[SemanticMutationValidationIssue], LoweredOperation]:
    issues: list[SemanticMutationValidationIssue] = []

    def err(code: str, message: str) -> None:
        issues.append(
            SemanticMutationValidationIssue(
                code=code, message=message, severity="error", op_index=index
            )
        )

    def warn(code: str, message: str) -> None:
        issues.append(
            SemanticMutationValidationIssue(
                code=code, message=message, severity="warning", op_index=index
            )
        )

    name = op.op
    subgraph = op.subgraph or _subgraph_for(op)

    # 1. op known
    if not is_known_op(name):
        err("unknown_op", f"unknown mutation op {name!r}")
        return issues, LoweredOperation(
            op_index=index, op=name, risk=MutationRisk.high.value, status="rejected"
        )

    # 2. deferred ops — honest dead end, not silent
    if name in DEFERRED_OPS:
        err(
            "op_deferred",
            f"op {name!r} is deferred to V2 — model the change as a new claim "
            f"(assert_claim) or append_event in V1.5",
        )
        return issues, LoweredOperation(
            op_index=index, op=name, risk=MutationRisk.high.value, status="deferred"
        )

    # 3. truth class (default applied downstream; only validate when present)
    if op.truth is not None and not is_truth_class(op.truth):
        err("bad_truth_class", f"unknown truth class {op.truth!r}")

    # 4. confidence range
    if op.confidence is not None and not (0.0 <= op.confidence <= 1.0):
        err("bad_confidence", "confidence must be between 0.0 and 1.0")

    # 5. timestamps parse ISO 8601
    for field_name, raw in (
        ("valid_from", op.valid_from),
        ("valid_until", op.valid_until),
        ("observed_at", op.observed_at),
        ("occurred_at", op.occurred_at),
    ):
        if raw and not _parses_iso(raw):
            err("bad_timestamp", f"{field_name} {raw!r} is not a valid ISO 8601 timestamp")

    # 6. evidence authorities (when present) must be known
    for ev in op.evidence:
        if ev.authority is not None and not is_source_authority(ev.authority):
            err(
                "bad_authority",
                f"evidence authority {ev.authority!r} is not a known source authority "
                f"({', '.join(sorted(a.value for a in SourceAuthority))})",
            )

    # 7. per-op structural rules
    if name == SemanticMutationOp.upsert_entity.value:
        _validate_entity_ref(op.subject, "subject", required=True, err=err, warn=warn)
    elif name in _CLAIM_OPS:
        _validate_claim_op(op, err=err, warn=warn)
    elif name == SemanticMutationOp.append_event.value:
        _validate_event_op(op, err=err, warn=warn)
    elif name in _RETRACT_OPS:
        _validate_retract_op(op, err=err, warn=warn)
    elif name == SemanticMutationOp.patch_entity.value:
        _validate_patch_entity_op(op, err=err, warn=warn)
    elif name == SemanticMutationOp.transition_state.value:
        _validate_transition_state_op(op, err=err, warn=warn)
    elif name == SemanticMutationOp.supersede_claim.value:
        _validate_supersede_claim_op(op, err=err, warn=warn)
    elif name == SemanticMutationOp.merge_duplicate_entities.value:
        _validate_merge_duplicate_entities_op(op, err=err, warn=warn)

    has_errors = any(i.is_error and i.op_index == index for i in issues)
    if has_errors:
        return issues, LoweredOperation(
            op_index=index, op=name, risk=_op_risk(op).value, status="rejected",
            subgraph=subgraph, truth=op.truth,
        )

    # 8. review-required ops (always)
    if name in REVIEW_REQUIRED_OPS:
        return issues, LoweredOperation(
            op_index=index, op=name, risk=MutationRisk.high.value,
            status="review_required", subgraph=subgraph, truth=op.truth,
        )

    return issues, LoweredOperation(
        op_index=index, op=name, risk=_op_risk(op).value, status="accepted",
        subgraph=subgraph, truth=op.truth,
    )


def _validate_entity_ref(ref, role, *, required, err, warn) -> None:
    del warn
    if ref is None:
        if required:
            err("missing_entity", f"{role} entity is required for this op")
        return
    if ref.type is not None and ref.type not in ENTITY_TYPES:
        err("unknown_entity_type", f"{role} type {ref.type!r} is not a known entity type")
        return
    if ref.type is not None and not entity_key_matches_type(ref.key, ref.type):
        err(
            "key_prefix_mismatch",
            f"{role} key {ref.key!r} does not match the canonical key prefix for "
            f"{ref.type!r}",
        )


def _validate_claim_op(op, *, err, warn) -> None:
    _validate_entity_ref(op.subject, "subject", required=True, err=err, warn=warn)

    # Object is either an entity ref or a literal value.
    if op.object is None and op.value is None:
        err("missing_object", "claim op requires an 'object' entity or a 'value'")
    elif op.object is not None:
        _validate_entity_ref(op.object, "object", required=False, err=err, warn=warn)

    # Predicate must be a known canonical edge type.
    if not op.predicate:
        err("missing_predicate", "claim op requires a 'predicate'")
    elif op.predicate not in EDGE_TYPES:
        err("unknown_predicate", f"predicate {op.predicate!r} is not a known edge type")
    elif op.object is not None and op.subject is not None:
        # Endpoint rules: when both endpoint types are declared, enforce them.
        spec = edge_spec(op.predicate)
        s_type, o_type = op.subject.type, op.object.type
        if spec and s_type and o_type and not spec.allows([s_type], [o_type]):
            allowed = ", ".join(f"{a}->{b}" for a, b in spec.allowed_pairs)
            err(
                "invalid_endpoints",
                f"{op.predicate} does not allow {s_type} -> {o_type}; allowed: {allowed}",
            )

    # Durable writes need evidence OR an explicit low-authority truth class.
    _check_evidence_or_low_authority(op, err)

    # Recall depends on the agent-authored description; warn (don't reject).
    if not (op.description and op.description.strip()):
        warn(
            "missing_description",
            "claim has no agent-authored description; retrieval recall depends on a "
            "description written for search (symptoms, synonyms, scope)",
        )


def _validate_event_op(op, *, err, warn) -> None:
    _validate_entity_ref(op.subject, "subject", required=False, err=err, warn=warn)
    _validate_entity_ref(op.actor, "actor", required=False, err=err, warn=warn)
    for index, target in enumerate(op.targets):
        _validate_entity_ref(target, f"targets[{index}]", required=False, err=err, warn=warn)
    for index, mention in enumerate(op.mentions):
        _validate_entity_ref(mention, f"mentions[{index}]", required=False, err=err, warn=warn)

    if op.subject is not None:
        _validate_activity_anchor(op.subject, "subject", err=err)
    if op.actor is not None:
        _validate_edge_endpoints(
            "PERFORMED",
            op.actor,
            op.subject,
            from_role="actor",
            to_role="subject",
            from_default="Person",
            to_default="Activity",
            err=err,
        )
    for index, target in enumerate(op.targets):
        _validate_edge_endpoints(
            "TOUCHED",
            op.subject,
            target,
            from_role="subject",
            to_role=f"targets[{index}]",
            from_default="Activity",
            to_default="Observation",
            err=err,
        )
    for index, mention in enumerate(op.mentions):
        _validate_edge_endpoints(
            "MENTIONS",
            op.subject,
            mention,
            from_role="subject",
            to_role=f"mentions[{index}]",
            from_default="Activity",
            to_default="Observation",
            err=err,
        )

    if not (op.verb and op.verb.strip()):
        err("missing_verb", "append_event requires a 'verb' (verb_class)")
    if op.occurred_at and not _parses_iso(op.occurred_at):
        err("bad_timestamp", f"occurred_at {op.occurred_at!r} is not valid ISO 8601")
    # An event with no target / actor / mention is allowed (it can still anchor
    # a timeline entry), but description aids recall.
    if not (op.description and op.description.strip()):
        warn(
            "missing_description",
            "event has no description; timeline recall improves with one",
        )


def _validate_retract_op(op, *, err, warn) -> None:
    # Target identity: subject+predicate(+object) or an explicit subject key.
    _validate_entity_ref(op.subject, "subject", required=True, err=err, warn=warn)
    if op.object is not None:
        _validate_entity_ref(op.object, "object", required=False, err=err, warn=warn)
    if op.op == SemanticMutationOp.end_relation_validity.value and not op.predicate:
        err("missing_predicate", "end_relation_validity requires a 'predicate'")
    if op.op == SemanticMutationOp.end_relation_validity.value and op.object is None:
        err(
            "missing_object",
            "end_relation_validity requires an 'object' entity to target an exact relation",
        )
    if op.predicate:
        _validate_predicate_and_endpoints(op, err=err)
    if not (op.reason and op.reason.strip()):
        warn("missing_reason", "retraction has no 'reason'; recording one aids audit")


def _validate_supersede_claim_op(op, *, err, warn) -> None:
    _validate_relation_target_op(
        op,
        op_name=SemanticMutationOp.supersede_claim.value,
        require_object=True,
        err=err,
        warn=warn,
    )
    _validate_entity_ref(
        op.superseded_by, "superseded_by", required=True, err=err, warn=warn
    )
    if op.predicate and op.subject is not None and op.superseded_by is not None:
        _validate_edge_endpoints(
            op.predicate,
            op.subject,
            op.superseded_by,
            from_role="subject",
            to_role="superseded_by",
            from_default=None,
            to_default=None,
            err=err,
        )
    if not (op.reason and op.reason.strip()):
        err("missing_reason", "supersede_claim requires a reason for audit history")
    _check_evidence_or_low_authority(op, err)
    if not (op.description and op.description.strip()):
        warn(
            "missing_description",
            "supersede_claim writes a replacement claim; retrieval recall improves with a description",
        )


def _validate_merge_duplicate_entities_op(op, *, err, warn) -> None:
    _validate_entity_ref(op.subject, "subject", required=True, err=err, warn=warn)
    _validate_entity_ref(op.object, "object", required=True, err=err, warn=warn)
    if op.subject is not None and op.object is not None:
        if normalize_entity_key(op.subject.key) == normalize_entity_key(op.object.key):
            err("self_merge", "merge_duplicate_entities requires two distinct entity keys")
        subject_type = _effective_entity_type(op.subject, None)
        object_type = _effective_entity_type(op.object, None)
        if subject_type and object_type and subject_type != object_type:
            err(
                "merge_type_mismatch",
                "merge_duplicate_entities requires subject and object to have the same entity type",
            )
    if not _merge_external_ids(op):
        err(
            "missing_external_ids",
            "merge_duplicate_entities requires external_ids or extra.external_ids for audit",
        )
    if not (op.reason and op.reason.strip()):
        err("missing_reason", "merge_duplicate_entities requires a reason for audit history")
    if not (op.description and op.description.strip()):
        warn(
            "missing_description",
            "merge_duplicate_entities writes a merge record; retrieval recall improves with a description",
        )


def _validate_patch_entity_op(op, *, err, warn) -> None:
    _validate_entity_ref(op.subject, "subject", required=True, err=err, warn=warn)
    entity_type = _effective_entity_type(op.subject, None)
    if entity_type not in ENTITY_TYPES:
        err("missing_entity_type", "patch_entity requires a known subject entity type")
        return

    if not op.patch:
        err("missing_patch", "patch_entity requires a non-empty 'patch' object")
        return

    allowed = ENTITY_TYPES[entity_type].patchable_properties
    for field, value in op.patch.items():
        if field not in allowed:
            err(
                "patch_field_not_allowed",
                f"{entity_type} does not allow patching field {field!r}; "
                f"allowed: {', '.join(sorted(allowed))}",
            )
            continue
        if field in _STATE_PATCH_FIELDS:
            err(
                "state_patch_not_allowed",
                "state fields must use transition_state so lifecycle history is preserved",
            )
        if field == "description" and not _retrieval_description_strong(value):
            err(
                "weak_description",
                "patch_entity description must be retrieval-grade and cannot be a weak overwrite",
            )

    if not op.expected_entity_version:
        warn(
            "missing_expected_entity_version",
            "patch_entity has no expected_entity_version; commit still checks subgraph versions",
        )


def _validate_transition_state_op(op, *, err, warn) -> None:
    _validate_entity_ref(op.subject, "subject", required=True, err=err, warn=warn)
    entity_type = _effective_entity_type(op.subject, None)
    if entity_type not in ENTITY_TYPES:
        err("missing_entity_type", "transition_state requires a known subject entity type")
        return

    spec = ENTITY_TYPES[entity_type]
    if not spec.lifecycle_states:
        err(
            "lifecycle_not_declared",
            f"{entity_type} does not declare lifecycle states",
        )
        return

    if not op.from_state:
        err("missing_from_state", "transition_state requires 'from_state'")
    elif op.from_state not in spec.lifecycle_states:
        err(
            "invalid_from_state",
            f"{entity_type} state {op.from_state!r} is not declared; "
            f"allowed: {', '.join(sorted(spec.lifecycle_states))}",
        )

    if not op.to_state:
        err("missing_to_state", "transition_state requires 'to_state'")
    elif op.to_state not in spec.lifecycle_states:
        err(
            "invalid_to_state",
            f"{entity_type} state {op.to_state!r} is not declared; "
            f"allowed: {', '.join(sorted(spec.lifecycle_states))}",
        )

    if op.from_state and op.to_state:
        allowed_targets = spec.lifecycle_transitions.get(op.from_state, frozenset())
        if op.from_state == op.to_state or op.to_state not in allowed_targets:
            allowed = ", ".join(sorted(allowed_targets)) or "(none)"
            err(
                "invalid_state_transition",
                f"{entity_type} cannot transition {op.from_state!r} -> "
                f"{op.to_state!r}; allowed from {op.from_state!r}: {allowed}",
            )

    if not (op.reason and op.reason.strip()):
        err("missing_reason", "transition_state requires a reason for audit history")
    if not op.expected_entity_version:
        warn(
            "missing_expected_entity_version",
            "transition_state has no expected_entity_version; commit still checks subgraph versions",
        )


def _validate_relation_target_op(
    op,
    *,
    op_name: str,
    require_object: bool,
    err,
    warn,
) -> None:
    _validate_entity_ref(op.subject, "subject", required=True, err=err, warn=warn)
    if require_object:
        _validate_entity_ref(op.object, "object", required=True, err=err, warn=warn)
    elif op.object is not None:
        _validate_entity_ref(op.object, "object", required=False, err=err, warn=warn)
    if not op.predicate:
        err("missing_predicate", f"{op_name} requires a 'predicate'")
    else:
        _validate_predicate_and_endpoints(op, err=err)


def _validate_predicate_and_endpoints(op, *, err) -> None:
    predicate = op.predicate
    if not predicate:
        return
    if predicate not in EDGE_TYPES:
        err("unknown_predicate", f"predicate {predicate!r} is not a known edge type")
        return
    if op.subject is not None and op.object is not None:
        _validate_edge_endpoints(
            predicate,
            op.subject,
            op.object,
            from_role="subject",
            to_role="object",
            from_default=None,
            to_default=None,
            err=err,
        )


def _validate_activity_anchor(ref: GraphEntityRef, role: str, *, err) -> None:
    spec = edge_spec("MENTIONS")
    ref_type = _effective_entity_type(ref, "Activity")
    if spec and ref_type and not spec.allows([ref_type], ["Observation"]):
        err(
            "invalid_endpoints",
            f"append_event {role} must be Activity-like; got {ref_type}",
        )


def _validate_edge_endpoints(
    predicate: str,
    from_ref: GraphEntityRef | None,
    to_ref: GraphEntityRef | None,
    *,
    from_role: str,
    to_role: str,
    from_default: str | None,
    to_default: str | None,
    err,
) -> None:
    spec = edge_spec(predicate)
    from_type = _effective_entity_type(from_ref, from_default)
    to_type = _effective_entity_type(to_ref, to_default)
    if not (spec and from_type and to_type):
        return
    if spec.allows([from_type], [to_type]):
        return
    allowed = ", ".join(f"{a}->{b}" for a, b in spec.allowed_pairs)
    err(
        "invalid_endpoints",
        f"{predicate} does not allow {from_role} {from_type} -> "
        f"{to_role} {to_type}; allowed: {allowed}",
    )


def _effective_entity_type(ref: GraphEntityRef | None, default: str | None) -> str | None:
    if ref is None:
        return default
    if ref.type in ENTITY_TYPES:
        return ref.type
    prefix = entity_key_prefix(ref.key)
    if prefix is None:
        return default
    return _PREFIX_TO_ENTITY_TYPE.get(prefix, default)


def _check_evidence_or_low_authority(op, err) -> None:
    truth = op.truth or ""
    # Only fact-asserting truth classes (authoritative_fact / source_observation)
    # must be grounded in evidence. Subjective/attributed classes (preference,
    # user_decision, timeline_event, agent_claim, …) carry their own authority.
    if truth not in EVIDENCE_REQUIRED_TRUTH_CLASSES:
        return
    if not op.evidence:
        err(
            "missing_evidence",
            f"a {truth!r} claim asserts an objective fact and must carry evidence; "
            f"use a low-authority truth class (agent_claim) for an ungrounded claim",
        )


def _retrieval_description_strong(value) -> bool:
    if not isinstance(value, str):
        return False
    text = " ".join(value.strip().split())
    if len(text) < 40:
        return False
    return len([token for token in text.split(" ") if token]) >= 6


# ---------------------------------------------------------------------------
# Risk policy
# ---------------------------------------------------------------------------


def _op_risk(op: SemanticMutation) -> MutationRisk:
    name = op.op
    if name in REVIEW_REQUIRED_OPS:
        return MutationRisk.high
    if name in (
        SemanticMutationOp.supersede_claim.value,
        SemanticMutationOp.merge_duplicate_entities.value,
    ):
        return MutationRisk.high
    if name in _RETRACT_OPS:
        return MutationRisk.medium
    if name in (
        SemanticMutationOp.patch_entity.value,
        SemanticMutationOp.transition_state.value,
    ):
        return MutationRisk.medium
    if name in _CLAIM_OPS:
        if op.truth == "user_decision":
            return MutationRisk.medium
        return MutationRisk.low
    if name in (
        SemanticMutationOp.append_event.value,
        SemanticMutationOp.upsert_entity.value,
    ):
        return MutationRisk.low
    return MutationRisk.high


def _overall_risk(
    accepted: list[LoweredOperation], review: list[LoweredOperation]
) -> MutationRisk:
    order = {MutationRisk.low.value: 0, MutationRisk.medium.value: 1, MutationRisk.high.value: 2}
    worst = MutationRisk.low
    for outcome in (*accepted, *review):
        if order[outcome.risk] > order[worst.value]:
            worst = MutationRisk(outcome.risk)
    return worst


def _decide(
    *,
    has_errors: bool,
    accepted: list[LoweredOperation],
    review: list[LoweredOperation],
    allow_review_required: bool,
    approved_by: str | None,
) -> str:
    if has_errors:
        return "rejected"
    if review:
        return "review_required"
    if not accepted:
        # Nothing applicable and no errors (e.g. an empty-after-filter batch).
        return "rejected"
    # Any medium/high-risk op requires explicit approval; otherwise auto-apply.
    needs_approval = any(
        o.risk in {MutationRisk.medium.value, MutationRisk.high.value}
        for o in accepted
    )
    if needs_approval and not (allow_review_required and approved_by):
        return "review_required"
    return "apply"


def _subgraph_for(op: SemanticMutation) -> str:
    """Best-effort subgraph for an op when not explicitly provided."""
    if op.predicate:
        return subgraph_for_predicate(op.predicate)
    if op.op == SemanticMutationOp.append_event.value:
        return "recent_changes"
    if op.op in (
        SemanticMutationOp.patch_entity.value,
        SemanticMutationOp.transition_state.value,
        SemanticMutationOp.merge_duplicate_entities.value,
    ):
        entity_type = _effective_entity_type(op.subject, None)
        if entity_type in _ENTITY_TYPE_SUBGRAPH:
            return _ENTITY_TYPE_SUBGRAPH[entity_type]
        if entity_type in ENTITY_TYPES:
            return _CATEGORY_SUBGRAPH.get(ENTITY_TYPES[entity_type].category, "memory")
    return "memory"


def _merge_external_ids(op: SemanticMutation) -> Mapping[str, object]:
    if op.external_ids:
        return op.external_ids
    raw = op.extra.get("external_ids") if isinstance(op.extra, Mapping) else None
    return raw if isinstance(raw, Mapping) else {}


# Predicate → subgraph (the named slice a claim belongs to).
_MEMORY_PREDICATE_SUBGRAPH = {
    "PROVIDES": "features",
    "IMPLEMENTED_IN": "features",
    "POLICY_APPLIES_TO": "decisions",
    "REPRODUCES": "debugging",
    "RESOLVED": "debugging",
    "ATTEMPTED_FIX_FAILED": "debugging",
    "VERIFIED": "debugging",
    "DECIDED": "decisions",
    "AFFECTS": "decisions",
}
_CATEGORY_SUBGRAPH = {
    "topology": "infra_topology",
    "ownership": "code_topology",
    "people": "code_topology",
    "timeline": "recent_changes",
    "generic": "admin",
}
_ENTITY_TYPE_SUBGRAPH = {
    "Preference": "decisions",
    "Policy": "decisions",
    "Decision": "decisions",
    "BugPattern": "debugging",
    "Fix": "debugging",
    "Activity": "recent_changes",
}


def subgraph_for_predicate(predicate: str) -> str:
    pred = (predicate or "").strip().upper()
    if pred in _MEMORY_PREDICATE_SUBGRAPH:
        return _MEMORY_PREDICATE_SUBGRAPH[pred]
    spec = edge_spec(pred)
    if spec is not None:
        return _CATEGORY_SUBGRAPH.get(spec.category, "memory")
    return "memory"


def _parses_iso(value: str) -> bool:
    try:
        datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return True
    except (ValueError, AttributeError):
        return False


__all__ = ["subgraph_for_predicate", "validate_semantic_request"]

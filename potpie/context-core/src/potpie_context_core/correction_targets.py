"""Resolve semantic corrections to immutable, previewable claim selections."""

from potpie_context_core.graph_contract import normalize_entity_key
from potpie_context_core.ports.claim_query import ClaimQueryFilter
from potpie_context_core.semantic_mutations import SemanticMutationValidationIssue


CORRECTION_OPS = frozenset(
    {"end_relation_validity", "retract_claim", "supersede_claim"}
)


def resolve_correction_targets(request, plan, claim_query) -> None:
    """Bind corrections once, rejecting ambiguous endpoint-only selections.

    Offline validation can still check payload shape. Runtime validation supplies
    a query port, and the resulting exact keys travel with the lowered plan.
    Entity retraction intentionally selects all claims incident to that entity.
    """
    if claim_query is None or plan.errors:
        return
    issues = []
    for index, op in enumerate(request.operations):
        if op.op not in CORRECTION_OPS or op.subject is None:
            continue
        subject = normalize_entity_key(op.subject.key)
        predicate = (op.predicate or "").strip().upper()
        object_key = normalize_entity_key(op.object.key) if op.object else None
        filters = dict(
            pot_id=request.pot_id,
            predicate_in=(predicate,) if predicate else (),
            claim_key_in=op.target_claim_keys,
            subgraph_in=(op.subgraph,) if op.subgraph else (),
        )
        rows = claim_query.find_claims(
            ClaimQueryFilter(
                **filters,
                subject_key_in=(subject,),
                object_key_in=(object_key,) if object_key else (),
            )
        )
        if not predicate and object_key is None:
            rows += claim_query.find_claims(
                ClaimQueryFilter(**filters, object_key_in=(subject,))
            )
        if op.environment is not None:
            rows = [row for row in rows if row.environment == op.environment]
        keys = tuple(sorted({row.claim_key for row in rows if row.claim_key}))
        error = None
        if any(not row.claim_key for row in rows):
            error = "Selected legacy claims have no claim key; correct their identity before retrying."
        elif op.target_claim_keys and set(keys) != set(op.target_claim_keys):
            error = "target_claim_keys must all identify active claims matching the supplied scope and endpoints."
        elif predicate and not op.target_claim_keys and len(keys) > 1:
            error = "Correction selects multiple claims; supply target_claim_keys from the read result."
        elif predicate and not keys:
            error = "Correction did not select an active claim matching the supplied scope and endpoints."
        if error:
            issues.append(
                SemanticMutationValidationIssue(
                    code="ambiguous_correction_target",
                    message=error,
                    op_index=index,
                )
            )
        else:
            plan.correction_targets[index] = keys
    if issues:
        plan.issues = (*plan.issues, *issues)
        plan.decision = "rejected"
        plan.ok = False

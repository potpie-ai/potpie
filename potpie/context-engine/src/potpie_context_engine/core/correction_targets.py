"""Resolve semantic corrections to immutable, previewable claim selections."""

from .claim_corrections import ClaimTarget
from .claim_lifecycle import claim_scope

from potpie_context_engine.core.graph_contract import normalize_entity_key
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.semantic_mutations import (
    SemanticMutationValidationIssue,
)

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
        if any(
            len(values) != len(set(values))
            for values in (op.target_claim_keys, op.target_claim_record_ids)
        ):
            issues.append(
                SemanticMutationValidationIssue(
                    code="ambiguous_correction_target",
                    message="Correction selectors must not repeat claim or incarnation identities.",
                    op_index=index,
                )
            )
            continue
        subject = normalize_entity_key(op.subject.key)
        predicate = (op.predicate or "").strip().upper()
        object_key = normalize_entity_key(op.object.key) if op.object else None
        filters = dict(
            pot_id=request.pot_id,
            predicate_in=(predicate,) if predicate else (),
            claim_key_in=op.target_claim_keys,
            subgraph_in=(op.subgraph,) if op.subgraph else (),
            record_id_in=op.target_claim_record_ids,
            include_invalidated=bool(op.target_claim_record_ids),
        )
        rows = claim_query.find_claims(
            ClaimQueryFilter(
                **filters,
                subject_key_in=(subject,),
                object_key_in=(object_key,) if object_key else (),
            )
        )
        if not predicate and object_key is None:
            rows += [
                row
                for row in claim_query.find_claims(
                    ClaimQueryFilter(**filters, object_key_in=(subject,))
                )
                if row.subject_key != subject
            ]
        environment = op.environment.strip().lower() if op.environment else None
        rows = [row for row in rows if claim_scope(row)[4] == environment]
        if op.target_source_system is not None:
            rows = [row for row in rows if row.source_system == op.target_source_system]
        if op.target_source_ref is not None:
            rows = [row for row in rows if row.source_ref == op.target_source_ref]
        keys = {row.claim_key for row in rows if row.claim_key}
        identities = [row.record_id for row in rows]
        error = None
        if any(not row.claim_key or not row.record_id for row in rows):
            error = "Selected legacy claims lack claim/incarnation identity; repair identity before retrying."
        elif len(identities) != len(set(identities)):
            error = "Selected incarnation identity is duplicated."
        elif op.target_claim_record_ids and set(identities) != set(
            op.target_claim_record_ids
        ):
            error = "target_claim_record_ids must all match the supplied scope and endpoints."
        elif op.target_claim_keys and keys != set(op.target_claim_keys):
            error = "target_claim_keys must all identify active claims matching the supplied scope and endpoints."
        elif not op.target_claim_record_ids and len(rows) != len(keys):
            error = "Correction selects multiple incarnations of one claim; supply target_claim_record_ids."
        elif (
            predicate
            and not (op.target_claim_keys or op.target_claim_record_ids)
            and len(rows) > 1
        ):
            error = "Correction selects multiple claims; supply exact claim and incarnation identities."
        elif predicate and not rows:
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
            plan.correction_targets[index] = tuple(
                sorted(
                    (ClaimTarget(row.claim_key, row.record_id) for row in rows),
                    key=lambda target: target.record_id,
                )
            )
    if issues:
        plan.issues = (*plan.issues, *issues)
        plan.decision = "rejected"
        plan.ok = False

"""Build evidence review annotations inside resource graph preparation."""

from potpie_context_engine.core.claim_codec import (
    claim_preservation_snapshot as _claim_snapshot,
)
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.ports.resource_store import (
    parse_resource_id,
    ResourceStoreError,
)
from potpie_context_engine.core.resource_to_semantic import SECTION_PREDICATE
from potpie_context_engine.core.semantic_mutations import SemanticMutation


def review_operations(claims, manifest):
    refs, sections, reason = (
        manifest.pending_review_refs,
        manifest.pending_review_sections,
        manifest.pending_review_reason,
    )
    if not refs or not sections or not reason:
        return ()
    pot_id, doc = manifest.pot_id, manifest.doc
    rows = claims.find_claims(ClaimQueryFilter(pot_id=pot_id, include_invalidated=True))
    affected = []
    section_set = set(sections)
    exact = set(refs)
    for row in rows:
        cited = tuple(
            dict.fromkeys(
                (
                    *row.source_refs,
                    *(str(item.get("source_ref") or "") for item in row.evidence),
                )
            )
        )
        matched = False
        for source_ref in cited:
            if source_ref in exact:
                matched = True
                break
            try:
                parsed = parse_resource_id(source_ref)
            except ResourceStoreError:
                continue
            if parsed.doc == doc and parsed.section in section_set:
                matched = True
                break
        if matched and row.predicate != SECTION_PREDICATE:
            affected.append(row)
    return build_review_operations(
        claims, pot_id=pot_id, rows=affected, refs=refs, reason=reason
    )


def build_review_operations(claims, *, pot_id, rows, refs, reason):
    keys = {key for row in rows for key in (row.subject_key, row.object_key)}
    labels = claims.entity_labels(pot_id=pot_id, entity_keys=keys)

    def concrete_type(key: str) -> str | None:
        concrete = sorted(label for label in labels.get(key, ()) if label != "Entity")
        return concrete[0] if concrete else None

    operations = []
    for row in rows:
        evidence = [dict(item) for item in row.evidence]
        if not evidence:
            evidence = [{"source_ref": ref} for ref in row.source_refs]
        raw = {
            "op": "assert_claim",
            "subgraph": row.subgraph or "knowledge",
            "predicate": row.predicate,
            "subject": {
                "key": row.subject_key,
                "type": concrete_type(row.subject_key),
            },
            "object": {
                "key": row.object_key,
                "type": concrete_type(row.object_key),
            },
            "truth": row.truth or "agent_claim",
            "confidence": row.confidence,
            "description": row.description or row.fact,
            "environment": row.environment,
            "valid_from": row.valid_at.isoformat() if row.valid_at else None,
            "valid_until": row.valid_until.isoformat() if row.valid_until else None,
            "observed_at": row.observed_at.isoformat() if row.observed_at else None,
            "evidence": evidence,
            "extra": {
                "evidence_review_required": True,
                "evidence_review_reason": reason,
                "evidence_review_refs": list(refs),
                "_preserve_claim_key": row.claim_key,
                "_preserve_claim_record_id": row.record_id,
                "_preserve_source_ref": row.source_ref,
                "_preserve_claim_properties": dict(row.properties),
                "_preserve_claim_fact": row.fact,
                "_preserve_source_system": row.source_system,
                "_expected_claim_snapshot": _claim_snapshot(row),
            },
        }
        operations.append(SemanticMutation.parse(raw))
    return tuple(operations)

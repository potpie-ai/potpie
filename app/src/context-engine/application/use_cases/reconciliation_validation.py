"""Validate ``ReconciliationPlan`` before deterministic apply.

Graphiti episodic ingest runs predicate-family auto-supersede separately (see
``adapters/outbound/graphiti/temporal_supersede.py`` and
``domain.ontology.predicate_family_for_episodic_supersede`` for cross-type edges
such as ``CHOSE`` vs ``MIGRATED_TO``); it is not part of this validation path.

Episodic edge-type collapse and high ``MODIFIED`` regression guards live in
``adapters/outbound/graphiti/edge_extraction_normalize.py`` (see
docs/context-graph-improvements/02-edge-type-collapse.md).

When ``CONTEXT_ENGINE_ONTOLOGY_SOFT_FAIL=1`` and strict mode is off, unknown
labels, non-catalog edge types, and invalid lifecycle values are coerced instead
of failing the batch (see docs/context-graph-fixes/03-graceful-ontology-downgrade.md).
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from domain.canonical_label_inference import (
    _ensure_required_properties_for_label,
    enrich_reconciliation_plan_entity_labels,
)
from domain.errors import ReconciliationPlanValidationError
from domain.graph_mutations import EdgeUpsert, EntityUpsert, InvalidationOp
from domain.ontology import (
    BASE_GRAPH_LABELS,
    CODE_GRAPH_LABELS,
    EDGE_TYPES,
    ENTITY_TYPES,
    canonical_entity_labels,
    is_canonical_edge_type,
    validate_structural_mutations,
)
from domain.reconciliation import ReconciliationPlan
from domain.reconciliation_flags import (
    infer_canonical_labels_enabled,
    ontology_soft_fail_enabled,
    ontology_strict_enabled,
)
from domain.reconciliation_issues import validation_lines_to_issues

MAX_EPISODES = 32
MAX_GENERIC_ENTITY_UPSERTS = 5000
MAX_GENERIC_EDGES = 10000
MAX_INVALIDATIONS = 2000

_TEMPORAL_PROPERTY_KEYS = frozenset(
    {"valid_at", "valid_from", "valid_to", "observed_at", "deployed_at"}
)
_OBSERVATION_FALLBACK = "Observation"


def validate_reconciliation_plan(
    plan: ReconciliationPlan, expected_pot_id: str
) -> None:
    plan.ontology_downgrades.clear()
    _validate_hard(plan, expected_pot_id)

    if infer_canonical_labels_enabled():
        enrich_reconciliation_plan_entity_labels(plan)

    soft = ontology_soft_fail_enabled() and not ontology_strict_enabled()
    if soft:
        _apply_soft_ontology_downgrades(plan)

    if plan.ontology_downgrades and _quality_issue_attach_room(plan):
        _attach_ontology_downgrade_quality_issues(plan)

    ontology_errors = validate_structural_mutations(
        plan.entity_upserts,
        plan.edge_upserts,
        plan.edge_deletes,
    )
    ontology_errors.extend(_validate_invalidations(plan.invalidations))
    if ontology_errors:
        sample = "; ".join(ontology_errors[:8])
        suffix = (
            ""
            if len(ontology_errors) <= 8
            else f"; ... {len(ontology_errors) - 8} more"
        )
        structured = validation_lines_to_issues(ontology_errors)
        raise ReconciliationPlanValidationError(
            f"ontology validation failed: {sample}{suffix}",
            structured_issues=tuple(structured),
        )


def _validate_hard(plan: ReconciliationPlan, expected_pot_id: str) -> None:
    if plan.event_ref.pot_id != expected_pot_id:
        raise ReconciliationPlanValidationError(
            "plan event_ref.pot_id does not match expected pot"
        )

    if len(plan.episodes) > MAX_EPISODES:
        raise ReconciliationPlanValidationError("too many episodes in plan")

    if len(plan.entity_upserts) > MAX_GENERIC_ENTITY_UPSERTS:
        raise ReconciliationPlanValidationError("entity upsert cap exceeded")
    if len(plan.edge_upserts) + len(plan.edge_deletes) > MAX_GENERIC_EDGES:
        raise ReconciliationPlanValidationError("edge mutation cap exceeded")
    if len(plan.invalidations) > MAX_INVALIDATIONS:
        raise ReconciliationPlanValidationError("invalidation cap exceeded")

    _validate_duplicate_entity_keys(plan.entity_upserts)
    _validate_temporal_strings(plan)


def _validate_duplicate_entity_keys(items: list[EntityUpsert]) -> None:
    seen: set[str] = set()
    for item in items:
        key = item.entity_key
        if not key or not key.strip():
            continue
        if key in seen:
            raise ReconciliationPlanValidationError(
                f"duplicate entity_key in batch: {key!r}"
            )
        seen.add(key)


def _validate_temporal_strings(plan: ReconciliationPlan) -> None:
    for eu in plan.entity_upserts:
        _assert_iso_temporal_props(eu.properties, eu.entity_key or "<missing>")
    for ed in plan.edge_upserts:
        _assert_iso_temporal_props(
            ed.properties, f"edge:{ed.edge_type}:{ed.from_entity_key}->{ed.to_entity_key}"
        )


def _assert_iso_temporal_props(props: dict[str, object], ref: str) -> None:
    for key in _TEMPORAL_PROPERTY_KEYS:
        raw = props.get(key)
        if raw is None:
            continue
        if isinstance(raw, datetime):
            continue
        s = str(raw).strip()
        if not s:
            continue
        try:
            datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ReconciliationPlanValidationError(
                f"{ref}: {key!r} must be a valid ISO 8601 timestamp"
            ) from exc


def _apply_soft_ontology_downgrades(plan: ReconciliationPlan) -> None:
    out = plan.ontology_downgrades
    allowed_nc = BASE_GRAPH_LABELS | CODE_GRAPH_LABELS
    for ent in plan.entity_upserts:
        props = dict(ent.properties)
        ent.properties = props
        removed = [
            lb
            for lb in ent.labels
            if lb not in ENTITY_TYPES and lb not in allowed_nc
        ]
        if removed:
            ent.labels = tuple(lb for lb in ent.labels if lb not in removed)
            out.append(
                {
                    "entity_uuid": ent.entity_key,
                    "kind": "unknown_labels",
                    "from": ",".join(removed),
                    "to": "dropped",
                }
            )
        if not canonical_entity_labels(ent.labels):
            if not _try_adr_document_fallback(ent, removed, out):
                _observation_fallback(ent, out)
        for label in canonical_entity_labels(ent.labels):
            _coerce_lifecycle_for_label(ent, label, out)
        for label in canonical_entity_labels(ent.labels):
            _ensure_required_properties_for_label(
                ent.properties, label, ent.entity_key
            )

    for edge in plan.edge_upserts:
        _maybe_rewrite_unknown_edge(edge, out)


def _try_adr_document_fallback(
    ent: EntityUpsert, removed: list[str], out: list[dict[str, str]]
) -> bool:
    """Map extractor ``ADR`` / unknown doc labels onto canonical ``Document`` when possible."""
    pool = {str(x) for x in removed} | {str(x) for x in ent.labels}
    if not any(x.upper() == "ADR" for x in pool):
        return False
    labels = list(ent.labels)
    for mark in ("Document", "Entity"):
        if mark not in labels:
            labels.append(mark)
    ent.labels = tuple(labels)
    ent.properties.setdefault("title", (ent.entity_key or "untitled")[:500])
    ent.properties.setdefault(
        "source_uri",
        str(ent.properties.get("source_uri") or "unknown:")[:2000],
    )
    _ensure_required_properties_for_label(
        ent.properties, "Document", ent.entity_key
    )
    out.append(
        {
            "entity_uuid": ent.entity_key,
            "kind": "canonical_label",
            "from": "ADR",
            "to": "Document",
        }
    )
    return True


def _observation_fallback(ent: EntityUpsert, out: list[dict[str, str]]) -> None:
    labels = list(ent.labels)
    for mark in (_OBSERVATION_FALLBACK, "Entity"):
        if mark not in labels:
            labels.append(mark)
    ent.labels = tuple(labels)
    if not ent.properties.get("summary"):
        ent.properties["summary"] = (ent.entity_key or "unknown")[:500]
    if not ent.properties.get("observed_at"):
        ent.properties["observed_at"] = datetime.now(timezone.utc).isoformat()
    out.append(
        {
            "entity_uuid": ent.entity_key,
            "kind": "canonical_label",
            "from": "none",
            "to": _OBSERVATION_FALLBACK,
        }
    )


def _coerce_lifecycle_for_label(
    ent: EntityUpsert, label: str, out: list[dict[str, str]]
) -> None:
    spec = ENTITY_TYPES.get(label)
    if not spec or not spec.lifecycle_states:
        return
    if label == "Decision":
        key = "status"
        value = ent.properties.get("status")
    elif "lifecycle_state" in ent.properties:
        key = "lifecycle_state"
        value = ent.properties.get("lifecycle_state")
    else:
        key = "status"
        value = ent.properties.get("status")
    if value is None:
        return
    if str(value) in spec.lifecycle_states:
        return
    old = str(value)
    ent.properties[key] = "unknown"
    out.append(
        {
            "entity_uuid": ent.entity_key,
            "kind": "lifecycle_status",
            "from": old,
            "to": "unknown",
        }
    )


def _maybe_rewrite_unknown_edge(edge: EdgeUpsert, out: list[dict[str, str]]) -> None:
    et = edge.edge_type
    if et in EDGE_TYPES:
        return
    old = et
    props = dict(edge.properties)
    props["original_edge_type"] = old
    props["confidence"] = 0.3
    edge.edge_type = "RELATED_TO"
    edge.properties = props
    out.append(
        {
            "entity_uuid": f"{edge.from_entity_key}->{edge.to_entity_key}",
            "kind": "edge_type",
            "from": old,
            "to": "RELATED_TO",
        }
    )


def _quality_issue_attach_room(plan: ReconciliationPlan) -> bool:
    extra = len(plan.ontology_downgrades)
    return len(plan.entity_upserts) + extra <= MAX_GENERIC_ENTITY_UPSERTS


def _attach_ontology_downgrade_quality_issues(plan: ReconciliationPlan) -> None:
    ev_ref = plan.event_ref.event_id
    for d in plan.ontology_downgrades:
        qid = str(uuid4())
        plan.entity_upserts.append(
            EntityUpsert(
                entity_key=f"quality:ontology_downgrade:{qid}",
                labels=("Entity", "QualityIssue"),
                properties={
                    "code": "ontology_downgrade",
                    "severity": "info",
                    "status": "proposed",
                    "kind": "ontology_downgrade",
                    "source_event_id": ev_ref,
                    "downgrade_kind": d.get("kind", ""),
                    "downgrade_from": d.get("from", ""),
                    "downgrade_to": d.get("to", ""),
                    "affected_entity": d.get("entity_uuid", ""),
                },
            )
        )


def _validate_invalidations(items: list[InvalidationOp]) -> list[str]:
    errors: list[str] = []
    for item in items:
        if not item.target_entity_key and not item.target_edge:
            errors.append("invalidation must set target_entity_key or target_edge")
        if item.target_edge:
            edge_type = item.target_edge[0]
            if not is_canonical_edge_type(edge_type):
                errors.append(
                    f"invalidation target_edge has non-canonical edge_type: {edge_type!r}"
                )
    return errors


"""Merge canonical node labels from episodic edge patterns into reconciliation plans.

See docs/context-graph-improvements/03-canonical-node-labels.md
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from domain.graph_mutations import EntityUpsert
from domain.ontology import (
    ENTITY_TYPES,
    canonical_entity_labels,
    inferred_labels_for_episodic_edge_endpoint,
    is_canonical_entity_label,
)
from domain.reconciliation import ReconciliationPlan


def _default_for_required_property(prop_name: str, entity_key: str) -> Any:
    if prop_name in ("title", "name", "statement"):
        return (entity_key or "unknown")[:500]
    if prop_name in ("summary", "description", "rationale", "alternatives_rejected"):
        return ""
    if prop_name in ("status", "lifecycle_state"):
        return "unknown"
    if prop_name == "severity":
        return "unknown"
    if prop_name == "component_type":
        return "unknown"
    if prop_name == "fix_type":
        return "unknown"
    if prop_name == "version":
        return "unknown"
    if prop_name == "deployed_at":
        return ""
    return ""


def _ensure_required_properties_for_label(
    properties: dict[str, Any], label: str, entity_key: str
) -> None:
    spec = ENTITY_TYPES.get(label)
    if spec is None:
        return
    for prop in spec.required_properties:
        if prop not in properties or properties[prop] is None:
            properties[prop] = _default_for_required_property(prop, entity_key)


def enrich_reconciliation_plan_entity_labels(plan: ReconciliationPlan) -> None:
    """Augment ``entity_upserts`` with labels implied by ``edge_upserts`` (in-place)."""
    additions: dict[str, set[str]] = defaultdict(set)
    for edge in plan.edge_upserts:
        for lbl in inferred_labels_for_episodic_edge_endpoint(edge.edge_type, "source"):
            additions[edge.from_entity_key].add(lbl)
        for lbl in inferred_labels_for_episodic_edge_endpoint(edge.edge_type, "target"):
            additions[edge.to_entity_key].add(lbl)
    if not additions:
        return

    by_key = {eu.entity_key: eu for eu in plan.entity_upserts}
    for entity_key, new_labels in additions.items():
        ent = by_key.get(entity_key)
        if ent is None:
            continue
        prior_canonical = frozenset(canonical_entity_labels(ent.labels))
        merged = set(ent.labels)
        for lbl in new_labels:
            if not is_canonical_entity_label(lbl):
                continue
            merged.add(lbl)
            if lbl not in prior_canonical:
                props = dict(ent.properties)
                _ensure_required_properties_for_label(props, lbl, entity_key)
                ent.properties = props
        ent.labels = tuple(sorted(merged))

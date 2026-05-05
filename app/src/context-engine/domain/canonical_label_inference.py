"""Enrich a ``ReconciliationPlan`` with canonical labels inferred by the classifier.

Plan path runs before structural writes; it shares the same
``ontology_classifier.classify_entity`` engine as the post-Graphiti Neo4j pass
so the vocabulary pinning is identical regardless of which path produced the
mutation.

See docs/context-graph-improvements/03-canonical-node-labels.md.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from domain.ontology import ENTITY_TYPES, canonical_entity_labels
from domain.ontology_classifier import build_signals, classify_entity
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
    """Augment ``entity_upserts`` with labels inferred by the shared classifier (in-place)."""
    outgoing: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, set[str]] = defaultdict(set)
    for edge in plan.edge_upserts:
        outgoing[edge.from_entity_key].add(edge.edge_type)
        incoming[edge.to_entity_key].add(edge.edge_type)

    for ent in plan.entity_upserts:
        signals = build_signals(
            labels=ent.labels,
            properties=ent.properties,
            outgoing_edge_names=outgoing.get(ent.entity_key, ()),
            incoming_edge_names=incoming.get(ent.entity_key, ()),
        )
        additions = classify_entity(signals)
        if not additions:
            continue

        prior_canonical = frozenset(canonical_entity_labels(ent.labels))
        merged = set(ent.labels)
        props = dict(ent.properties)
        for label in additions:
            merged.add(label)
            if label not in prior_canonical:
                _ensure_required_properties_for_label(props, label, ent.entity_key)
        ent.properties = props
        ent.labels = tuple(sorted(merged))

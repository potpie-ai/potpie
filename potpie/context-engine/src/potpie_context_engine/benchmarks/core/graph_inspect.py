"""Reconstruct a bounded structural sample from canonical graph evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from potpie_context_engine.application.services.envelope_builder import envelope_to_dict
from potpie_context_engine.core.ontology import ENTITY_TYPES
from potpie_context_engine.core.ports.agent_context import ResolveRequest
from potpie_context_engine.core.ports.graph_service import GraphService


@dataclass(frozen=True)
class GraphEntity:
    label: str
    key: str
    properties: dict[str, Any]


@dataclass(frozen=True)
class GraphEdge:
    type: str
    from_label: str
    from_key: str
    to_label: str
    to_key: str
    properties: dict[str, Any]


@dataclass
class GraphSnapshot:
    entities: list[GraphEntity] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    def entities_by_label(self, label: str) -> list[GraphEntity]:
        return [e for e in self.entities if e.label == label]

    def edges_by_type(self, edge_type: str) -> list[GraphEdge]:
        return [e for e in self.edges if e.type == edge_type]


def snapshot_graph(
    service: GraphService, pot_id: str, *, limit: int = 200
) -> GraphSnapshot:
    """Sample reader claims, bounded by ``limit``; this is not a full graph export."""
    envelope = service.resolve(
        ResolveRequest(
            pot_id=pot_id,
            intent="operations",
            include=("infra_topology", "timeline", "prior_bugs", "coding_preferences"),
            max_items=min(max(limit, 0), 200),
        )
    )
    return _parse_envelope_claims({"result": envelope_to_dict(envelope)})


_KEY_PREFIX_LABELS = {spec.key_prefix: label for label, spec in ENTITY_TYPES.items()}


def _label_from_key(key: str) -> str:
    prefix = key.split(":", 1)[0].strip().lower() if ":" in key else ""
    if prefix in _KEY_PREFIX_LABELS:
        return _KEY_PREFIX_LABELS[prefix]
    return prefix.capitalize() if prefix else "Entity"


def _parse_envelope_claims(response: dict[str, Any]) -> GraphSnapshot:
    """Reconstruct entities + edges from the AgentEnvelope's claim items.

    Each ``result.items[]`` whose ``payload`` carries ``subject_key`` /
    ``object_key`` / ``predicate`` is one canonical claim edge. The two
    endpoints become entities (label from key prefix); the claim becomes an
    edge. Items without that triple (e.g. non-topology reader output) are
    skipped — they're graded via retrieval / the judge, not structurally.
    """
    result = response.get("result")
    if not isinstance(result, dict):
        return GraphSnapshot()
    items = result.get("items")
    if not isinstance(items, list):
        return GraphSnapshot()

    entities_by_key: dict[tuple[str, str], GraphEntity] = {}
    edges: list[GraphEdge] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        subj = payload.get("subject_key")
        obj = payload.get("object_key")
        pred = payload.get("predicate")
        if not (
            isinstance(subj, str) and isinstance(obj, str) and isinstance(pred, str)
        ):
            continue
        if not subj or not obj or not pred:
            continue
        s_label = _label_from_key(subj)
        o_label = _label_from_key(obj)
        env_prop = payload.get("environment")
        entities_by_key.setdefault(
            (s_label, subj), GraphEntity(label=s_label, key=subj, properties={})
        )
        entities_by_key.setdefault(
            (o_label, obj),
            GraphEntity(
                label=o_label,
                key=obj,
                properties={"environment": env_prop}
                if isinstance(env_prop, str)
                else {},
            ),
        )
        edges.append(
            GraphEdge(
                type=pred.upper(),
                from_label=s_label,
                from_key=subj,
                to_label=o_label,
                to_key=obj,
                properties={
                    k: payload[k]
                    for k in (
                        "environment",
                        "fact",
                        "source_ref",
                        "source_system",
                        "valid_at",
                    )
                    if payload.get(k) is not None
                },
            )
        )
    return GraphSnapshot(entities=list(entities_by_key.values()), edges=edges)

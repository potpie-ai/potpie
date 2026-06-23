"""Snapshot the post-ingest graph for ontology / structural assertions.

After all events are reconciled, the bench needs to ask: "given that
these events were ingested, what entities + edges exist in the graph?"
This module wraps the engine's existing query surface to return a
flattened, easy-to-assert-on snapshot.

We reuse ``POST /api/v2/context/query/context-graph`` with a permissive
query body so the assertion code can filter in-process. That keeps the
engine HTTP surface unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from context_engine.adapters.outbound.http.potpie_context_api_client import PotpieContextApiClient


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
    client: PotpieContextApiClient,
    pot_id: str,
    *,
    limit: int = 200,
) -> GraphSnapshot:
    """Pull entities + edges for a pot via the engine's query API.

    The read trunk (rebuild plan P8/P9) answers every non-agentic read
    through the single :class:`AgentEnvelope` shape: ``result.items[]``, where
    each topology item carries ``payload.{subject_key, object_key, predicate}``
    (the canonical claim row surfaced by the ``infra_topology`` reader). There
    is no longer a ``goal=neighborhood`` path that dumps ``result.nodes`` /
    ``result.edges`` — so we reconstruct the structural snapshot from the
    envelope's claim rows instead.

    We query **unscoped** with the reader-backed includes so the
    InfraTopologyReader returns the whole pot's topology (an unanchored
    ``find_claims`` over the infra predicates) rather than a 2-hop
    neighbourhood around a single service. Entity labels are derived from the
    canonical key prefix (``service:…`` → ``Service``, ``datastore:…`` →
    ``DataStore``); the claim predicate (``DEPENDS_ON`` / ``USES`` / …) becomes
    the edge ``type``. The bench uses an admin-scoped key so visibility is not
    an issue.

    Older engines that still answer the structural ``nodes`` / ``edges`` shape
    are handled by the fallbacks.
    """
    body = {
        "pot_id": pot_id,
        "goal": "retrieve",
        "strategy": "auto",
        "intent": "operations",
        # Reader-backed include vocabulary (READER_BACKED_INCLUDES). Passing the
        # includes explicitly routes every P9 reader; unscoped → full topology.
        "include": ["infra_topology", "timeline", "prior_bugs", "coding_preferences"],
        "budget": {"max_items": 50},
        "limit": min(limit, 200),
    }
    response = client.context_graph_query(body)
    snap = _parse_envelope_claims(response)
    if snap.entities or snap.edges:
        return snap
    # Fallbacks for older engines that answer the structural nodes/edges shape.
    snap = _parse_neighborhood(response)
    if snap.entities or snap.edges:
        return snap
    return _parse_snapshot(response)


# Canonical entity-key prefix → ontology label. Keys look like
# ``service:checkout-api`` or, with an org/repo segment,
# ``service:acme-platform:inventory-svc`` — we split on the FIRST ``:`` so the
# prefix is recovered regardless of how many segments follow.
_KEY_PREFIX_LABELS: dict[str, str] = {
    "service": "Service",
    "datastore": "DataStore",
    "environment": "Environment",
    "cluster": "Cluster",
    "team": "Team",
    "person": "Person",
    "repository": "Repository",
    "feature": "Feature",
    "topic": "Topic",
    "secret": "Secret",
    "document": "Document",
    "decision": "Decision",
    "issue": "Issue",
    "deployment": "Deployment",
    "alert": "Alert",
}


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


_BASE_LABELS = frozenset({"Entity", "Node"})


def _pick_label(labels: Any) -> str:
    """Choose the most specific label from a node's label list."""
    if isinstance(labels, str):
        return labels
    if not isinstance(labels, (list, tuple)):
        return ""
    specific = [str(label) for label in labels if str(label) not in _BASE_LABELS]
    if specific:
        return specific[0]
    return str(labels[0]) if labels else ""


def _edge_predicate(
    from_key: str, to_key: str, props: dict[str, Any], raw_type: str
) -> str:
    """Recover the semantic predicate for a canonical RELATES_TO edge."""
    for prop in ("name", "predicate", "relation", "edge_name"):
        val = props.get(prop)
        if val:
            return str(val)
    # ``fact`` is "<from_key> <PREDICATE> <to_key>"; extract the middle token.
    fact = str(props.get("fact") or "")
    if fact and from_key and to_key and from_key in fact and to_key in fact:
        middle = fact.split(from_key, 1)[-1].rsplit(to_key, 1)[0].strip()
        if middle:
            return middle.replace(" ", "_").upper()
    return raw_type or "RELATES_TO"


def _parse_neighborhood(response: dict[str, Any]) -> GraphSnapshot:
    """Parse a ``goal=neighborhood`` response (``result.nodes`` / ``result.edges``)."""
    result = response.get("result")
    if not isinstance(result, dict):
        return GraphSnapshot()

    entities: list[GraphEntity] = []
    label_by_key: dict[str, str] = {}
    for raw in result.get("nodes") or []:
        if not isinstance(raw, dict):
            continue
        key = str(raw.get("entity_key") or raw.get("key") or raw.get("id") or "")
        label = _pick_label(raw.get("labels") or raw.get("label"))
        if not label:
            continue
        props = dict(raw.get("properties") or {})
        entities.append(GraphEntity(label=label, key=key, properties=props))
        if key:
            label_by_key[key] = label

    edges: list[GraphEdge] = []
    for raw in result.get("edges") or result.get("relationships") or []:
        if not isinstance(raw, dict):
            continue
        from_key = str(raw.get("from") or raw.get("source") or "")
        to_key = str(raw.get("to") or raw.get("target") or "")
        props = dict(raw.get("properties") or {})
        predicate = _edge_predicate(from_key, to_key, props, str(raw.get("type") or ""))
        edges.append(
            GraphEdge(
                type=predicate,
                from_label=label_by_key.get(from_key, ""),
                from_key=from_key,
                to_label=label_by_key.get(to_key, ""),
                to_key=to_key,
                properties=props,
            )
        )

    return GraphSnapshot(entities=entities, edges=edges)


def _parse_snapshot(response: dict[str, Any]) -> GraphSnapshot:
    entities: list[GraphEntity] = []
    edges: list[GraphEdge] = []

    raw_entities = (
        response.get("entities")
        or response.get("nodes")
        or (response.get("result") or {}).get("entities")
        or []
    )
    for raw in raw_entities:
        if not isinstance(raw, dict):
            continue
        label = str(raw.get("label") or raw.get("type") or raw.get("entity_type") or "")
        key = str(raw.get("entity_key") or raw.get("key") or raw.get("id") or "")
        if not label:
            continue
        props = dict(raw.get("properties") or raw.get("attributes") or {})
        # Some adapters flatten properties onto the entity itself.
        for k, v in raw.items():
            if k in {
                "label",
                "type",
                "entity_type",
                "entity_key",
                "key",
                "id",
                "properties",
                "attributes",
            }:
                continue
            props.setdefault(k, v)
        entities.append(GraphEntity(label=label, key=key, properties=props))

    raw_edges = (
        response.get("edges")
        or response.get("relationships")
        or (response.get("result") or {}).get("edges")
        or []
    )
    for raw in raw_edges:
        if not isinstance(raw, dict):
            continue
        edge_type = str(
            raw.get("type") or raw.get("relationship") or raw.get("edge_type") or ""
        )
        if not edge_type:
            continue
        from_node = raw.get("from") or raw.get("source") or {}
        to_node = raw.get("to") or raw.get("target") or {}
        edges.append(
            GraphEdge(
                type=edge_type,
                from_label=str(from_node.get("label") or from_node.get("type") or ""),
                from_key=str(
                    from_node.get("entity_key")
                    or from_node.get("key")
                    or from_node.get("id")
                    or ""
                ),
                to_label=str(to_node.get("label") or to_node.get("type") or ""),
                to_key=str(
                    to_node.get("entity_key")
                    or to_node.get("key")
                    or to_node.get("id")
                    or ""
                ),
                properties=dict(raw.get("properties") or {}),
            )
        )

    return GraphSnapshot(entities=entities, edges=edges)

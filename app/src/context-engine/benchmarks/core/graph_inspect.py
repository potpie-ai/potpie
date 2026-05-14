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

from adapters.outbound.http.potpie_context_api_client import PotpieContextApiClient


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
    """Pull all entities + edges for a pot via the existing query API.

    Uses a broad ``goal=retrieve`` request to get every node/edge the
    server is willing to return. The bench uses an admin-scoped key so
    visibility is not an issue.
    """
    body = {
        "pot_id": pot_id,
        "goal": "retrieve",
        "strategy": "traversal",
        "include_entities": True,
        "include_edges": True,
        "limit": min(limit, 200),
    }
    response = client.context_graph_query(body)
    return _parse_snapshot(response)


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
            if k in {"label", "type", "entity_type", "entity_key", "key", "id", "properties", "attributes"}:
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
        edge_type = str(raw.get("type") or raw.get("relationship") or raw.get("edge_type") or "")
        if not edge_type:
            continue
        from_node = raw.get("from") or raw.get("source") or {}
        to_node = raw.get("to") or raw.get("target") or {}
        edges.append(
            GraphEdge(
                type=edge_type,
                from_label=str(from_node.get("label") or from_node.get("type") or ""),
                from_key=str(from_node.get("entity_key") or from_node.get("key") or from_node.get("id") or ""),
                to_label=str(to_node.get("label") or to_node.get("type") or ""),
                to_key=str(to_node.get("entity_key") or to_node.get("key") or to_node.get("id") or ""),
                properties=dict(raw.get("properties") or {}),
            )
        )

    return GraphSnapshot(entities=entities, edges=edges)

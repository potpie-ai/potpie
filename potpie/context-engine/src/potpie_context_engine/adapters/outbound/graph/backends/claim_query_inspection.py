"""Structural inspection derived entirely from a canonical ClaimQueryPort."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from potpie_context_core.graph_entity_summary import normalize_entity_properties
from potpie_context_core.ports.claim_query import (
    ClaimQueryFilter,
    ClaimQueryPort,
    ClaimRow,
)
from potpie_context_core.ports.graph.inspection import GraphEdge, GraphNode, GraphSlice

_SCAN_LIMIT = 100_000


@dataclass(slots=True)
class ClaimQueryInspection:
    """Backend-neutral traversal over current canonical claim rows."""

    claim_query: ClaimQueryPort

    def neighborhood(
        self,
        *,
        pot_id: str,
        entity_key: str,
        depth: int = 1,
        direction: str = "both",
        predicates: tuple[str, ...] = (),
        limit: int | None = None,
    ) -> GraphSlice:
        rows = self._current_rows(pot_id)
        seen_node_keys: dict[str, None] = {}
        edges: list[GraphEdge] = []
        seen_edges: set[tuple[str, str, str]] = set()
        frontier = {entity_key}
        max_edges = max(0, int(limit)) if limit is not None else None
        predicate_set = {predicate.upper() for predicate in predicates if predicate}
        walk_out = direction in ("out", "both")
        walk_in = direction in ("in", "both")
        truncated = False
        visited_frontier: set[str] = set()
        for _ in range(max(1, depth)):
            current = frontier - visited_frontier
            if not current:
                break
            visited_frontier.update(current)
            next_frontier: set[str] = set()
            for row in rows:
                if predicate_set and row.predicate.upper() not in predicate_set:
                    continue
                follows_out = walk_out and row.subject_key in current
                follows_in = walk_in and row.object_key in current
                if not (follows_out or follows_in):
                    continue
                edge_key = (row.subject_key, row.predicate, row.object_key)
                if edge_key not in seen_edges:
                    if max_edges is not None and len(edges) >= max_edges:
                        truncated = True
                        break
                    seen_edges.add(edge_key)
                    edges.append(_edge(row))
                for key in (row.subject_key, row.object_key):
                    seen_node_keys.setdefault(key, None)
                if follows_out:
                    next_frontier.add(row.object_key)
                if follows_in:
                    next_frontier.add(row.subject_key)
            if truncated:
                break
            frontier = next_frontier - visited_frontier
            if not frontier:
                break
        return GraphSlice(
            pot_id=pot_id,
            nodes=self._nodes(pot_id, seen_node_keys),
            edges=tuple(edges),
            truncated=truncated,
        )

    def path(
        self, *, pot_id: str, from_key: str, to_key: str, max_depth: int = 4
    ) -> GraphSlice:
        adjacency: dict[str, list[ClaimRow]] = {}
        for row in self._current_rows(pot_id):
            adjacency.setdefault(row.subject_key, []).append(row)
            adjacency.setdefault(row.object_key, []).append(row)
        queue: list[tuple[str, list[ClaimRow]]] = [(from_key, [])]
        visited = {from_key}
        while queue:
            node, trail = queue.pop(0)
            if node == to_key:
                node_keys = {from_key, to_key}
                for row in trail:
                    node_keys.update((row.subject_key, row.object_key))
                return GraphSlice(
                    pot_id=pot_id,
                    nodes=self._nodes(pot_id, sorted(node_keys)),
                    edges=tuple(_edge(row) for row in trail),
                )
            if len(trail) >= max(0, max_depth):
                continue
            for row in adjacency.get(node, ()):
                next_key = (
                    row.object_key if row.subject_key == node else row.subject_key
                )
                if next_key not in visited:
                    visited.add(next_key)
                    queue.append((next_key, [*trail, row]))
        return GraphSlice(pot_id=pot_id)

    def labels(
        self, *, pot_id: str, entity_keys: Iterable[str]
    ) -> Mapping[str, tuple[str, ...]]:
        return self.claim_query.entity_labels(
            pot_id=pot_id, entity_keys=entity_keys
        )

    def slice(self, *, pot_id: str, filter_: ClaimQueryFilter) -> GraphSlice:
        rows = self.claim_query.find_claims(filter_)
        node_keys = sorted(
            {key for row in rows for key in (row.subject_key, row.object_key)}
        )
        return GraphSlice(
            pot_id=pot_id,
            nodes=self._nodes(pot_id, node_keys),
            edges=tuple(_edge(row) for row in rows),
        )

    def _current_rows(self, pot_id: str) -> list[ClaimRow]:
        return list(
            self.claim_query.find_claims(
                ClaimQueryFilter(pot_id=pot_id, limit=_SCAN_LIMIT)
            )
        )

    def _nodes(self, pot_id: str, entity_keys: Iterable[str]) -> tuple[GraphNode, ...]:
        keys = tuple(dict.fromkeys(entity_keys))
        if not keys:
            return ()
        labels_by_key = self.claim_query.entity_labels(
            pot_id=pot_id, entity_keys=keys
        )
        bulk_properties = getattr(self.claim_query, "entity_properties_many", None)
        if callable(bulk_properties):
            properties_by_key = bulk_properties(
                pot_id=pot_id, entity_keys=keys
            )
        else:
            properties_by_key = {
                key: self.claim_query.entity_properties(
                    pot_id=pot_id, entity_key=key
                )
                for key in keys
            }
        return tuple(
            GraphNode(
                key=key,
                labels=tuple(labels_by_key.get(key, ())),
                properties=normalize_entity_properties(
                    properties_by_key.get(key, {}), entity_key=key
                ),
            )
            for key in keys
        )


def _edge(row: ClaimRow) -> GraphEdge:
    properties = {
        **dict(row.properties),
        "claim_key": row.claim_key,
        "subgraph": row.subgraph,
        "truth": row.truth,
        "confidence": row.confidence,
        "description": row.description,
        "environment": row.environment,
        "fact": row.fact,
        "source_system": row.source_system,
        "source_ref": row.source_ref,
        "source_refs": list(row.source_refs),
        "valid_at": _dt_iso(row.valid_at),
        "valid_until": _dt_iso(row.valid_until),
        "observed_at": _dt_iso(row.observed_at),
        "mutation_id": row.mutation_id,
    }
    return GraphEdge(
        predicate=row.predicate,
        from_key=row.subject_key,
        to_key=row.object_key,
        properties={
            key: value for key, value in properties.items() if value is not None
        },
    )


def _dt_iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


__all__ = ["ClaimQueryInspection"]

"""FalkorDB :class:`GraphInspectionPort` — structural traversal projection.

Backs ``potpie graph inspect`` and the local graph explorer UI. A rebuildable
view over the canonical ``(:Entity {group_id})-[:RELATES_TO {name}]->(:Entity)``
claim edges — *not* the agent read path (that is the readers over
:class:`ClaimQueryPort`). Reuses the writer's graph handle + result-row
normalization so it runs unchanged on both Lite (``redislite``) and server
FalkorDB.

Traversal is done as a bounded breadth-first expansion in Python rather than a
variable-length Cypher pattern: FalkorDB requires literal hop bounds and the
local pot graphs are small, so a handful of 1-hop edge queries is both portable
and easy to cap.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from potpie_context_engine.adapters.outbound.graph.falkordb_writer import (
    _records_from_result,
    build_falkordb_graph,
)
from potpie_context_core.ports.claim_query import ClaimQueryFilter
from potpie_context_core.ports.graph.inspection import GraphEdge, GraphNode, GraphSlice
from potpie_context_engine.domain.ports.embedder import EmbedderPort
from potpie_context_engine.domain.ports.settings import ContextEngineSettingsPort

# Hard caps so a runaway pot can never OOM the explorer / the daemon process.
_MAX_NODES = 2000
_MAX_EDGES = 4000
_MAX_DEPTH = 4

_NODES_CYPHER = """
MATCH (e:Entity {group_id: $gid})
RETURN e.entity_key AS key, labels(e) AS labels, properties(e) AS props
LIMIT $limit
"""

_EDGES_CYPHER = """
MATCH (a:Entity {group_id: $gid})-[r:RELATES_TO {group_id: $gid}]->(b:Entity {group_id: $gid})
WHERE ($include_invalid OR r.invalid_at IS NULL)
  AND ($preds IS NULL OR r.name IN $preds)
RETURN a.entity_key AS source, b.entity_key AS target, r.name AS predicate, properties(r) AS props
LIMIT $limit
"""

# Edges incident (either direction) to the current BFS frontier.
_INCIDENT_CYPHER = """
MATCH (a:Entity {group_id: $gid})-[r:RELATES_TO {group_id: $gid}]->(b:Entity {group_id: $gid})
WHERE (a.entity_key IN $frontier OR b.entity_key IN $frontier)
  AND ($include_invalid OR r.invalid_at IS NULL)
RETURN a.entity_key AS source, b.entity_key AS target, r.name AS predicate, properties(r) AS props
"""

_HYDRATE_CYPHER = """
MATCH (e:Entity {group_id: $gid})
WHERE e.entity_key IN $keys
RETURN e.entity_key AS key, labels(e) AS labels, properties(e) AS props
"""


def _is_embedding(key: str, value: Any) -> bool:
    """Heuristic: drop large float vectors / embedding props from explorer payloads."""
    if "embedding" in key.lower() or "vector" in key.lower():
        return True
    if (
        isinstance(value, (list, tuple))
        and len(value) > 32
        and all(isinstance(x, (int, float)) for x in value)
    ):
        return True
    return False


def _clean_props(props: Any) -> dict[str, Any]:
    if not isinstance(props, Mapping):
        return {}
    return {
        k: v
        for k, v in props.items()
        if k not in ("group_id",) and not _is_embedding(str(k), v)
    }


class FalkorDBInspection:
    """:class:`GraphInspectionPort` over canonical ``:RELATES_TO`` edges in FalkorDB."""

    def __init__(
        self,
        settings: ContextEngineSettingsPort,
        *,
        graph: Any | None = None,
        graph_provider: Callable[[], Any] | None = None,
        embedder: EmbedderPort | None = None,
    ) -> None:
        self._settings = settings
        self._graph = graph  # injectable for unit tests
        self._graph_provider = graph_provider  # shared handle from the container
        # Accepted for construction parity with the other FalkorDB capability
        # adapters (the backend hands the same embedder to all of them).
        # Structural inspection reads embeddings already written by the writer,
        # so it never embeds at read time — kept for that uniform wiring.
        self._embedder = embedder

    def _get_graph(self) -> Any:
        if self._graph is None:
            self._graph = (
                self._graph_provider()
                if self._graph_provider is not None
                else build_falkordb_graph(self._settings)
            )
        return self._graph

    def _query(self, cypher: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        return _records_from_result(
            self._get_graph().query(cypher, params=dict(params))
        )

    def _hydrate_nodes(self, pot_id: str, keys: list[str]) -> tuple[GraphNode, ...]:
        if not keys:
            return ()
        recs = self._query(_HYDRATE_CYPHER, {"gid": pot_id, "keys": keys})
        return tuple(
            GraphNode(
                key=rec["key"],
                labels=tuple(lbl for lbl in (rec.get("labels") or [])),
                properties=_clean_props(rec.get("props")),
            )
            for rec in recs
        )

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
        depth = max(1, min(int(depth), _MAX_DEPTH))
        max_edges = (
            min(max(0, int(limit)), _MAX_EDGES) if limit is not None else _MAX_EDGES
        )
        predicate_set = {p.upper() for p in predicates if p}
        walk_out = direction in ("out", "both")
        walk_in = direction in ("in", "both")
        visited: set[str] = {entity_key}
        frontier: set[str] = {entity_key}
        edges: dict[tuple[str, str, str], GraphEdge] = {}
        truncated = False
        for _ in range(depth):
            if not frontier:
                break
            recs = self._query(
                _INCIDENT_CYPHER,
                {"gid": pot_id, "frontier": list(frontier), "include_invalid": False},
            )
            new: set[str] = set()
            for rec in recs:
                src, tgt, pred = rec["source"], rec["target"], rec["predicate"]
                if predicate_set and str(pred).upper() not in predicate_set:
                    continue
                follows_out = walk_out and src in frontier
                follows_in = walk_in and tgt in frontier
                if not (follows_out or follows_in):
                    continue
                edges[(src, pred, tgt)] = GraphEdge(
                    predicate=pred,
                    from_key=src,
                    to_key=tgt,
                    properties=_clean_props(rec.get("props")),
                )
                if follows_out and tgt not in visited:
                    new.add(tgt)
                if follows_in and src not in visited:
                    new.add(src)
                if len(edges) >= max_edges:
                    truncated = True
                    break
            if truncated:
                break
            visited |= new
            frontier = new
        nodes = self._hydrate_nodes(pot_id, list(visited)[:_MAX_NODES])
        return GraphSlice(
            pot_id=pot_id,
            nodes=nodes,
            edges=tuple(edges.values()),
            truncated=truncated or len(visited) > _MAX_NODES,
        )

    def slice(self, *, pot_id: str, filter_: ClaimQueryFilter) -> GraphSlice:
        preds = list(filter_.predicate_in) or None
        node_recs = self._query(_NODES_CYPHER, {"gid": pot_id, "limit": _MAX_NODES})
        edge_recs = self._query(
            _EDGES_CYPHER,
            {
                "gid": pot_id,
                "preds": preds,
                "include_invalid": bool(filter_.include_invalidated),
                "limit": _MAX_EDGES,
            },
        )
        nodes = tuple(
            GraphNode(
                key=rec["key"],
                labels=tuple(lbl for lbl in (rec.get("labels") or [])),
                properties=_clean_props(rec.get("props")),
            )
            for rec in node_recs
        )
        edges = tuple(
            GraphEdge(
                predicate=rec["predicate"],
                from_key=rec["source"],
                to_key=rec["target"],
                properties=_clean_props(rec.get("props")),
            )
            for rec in edge_recs
        )
        return GraphSlice(
            pot_id=pot_id,
            nodes=nodes,
            edges=edges,
            truncated=len(node_recs) >= _MAX_NODES or len(edge_recs) >= _MAX_EDGES,
        )

    def labels(
        self, *, pot_id: str, entity_keys: Iterable[str]
    ) -> Mapping[str, tuple[str, ...]]:
        keys = list(entity_keys)
        if not keys:
            return {}
        recs = self._query(_HYDRATE_CYPHER, {"gid": pot_id, "keys": keys})
        return {
            rec["key"]: tuple(lbl for lbl in (rec.get("labels") or [])) for rec in recs
        }

    def path(
        self, *, pot_id: str, from_key: str, to_key: str, max_depth: int = 4
    ) -> GraphSlice:
        max_depth = max(1, min(int(max_depth), _MAX_DEPTH))
        cypher = (
            "MATCH (a:Entity {group_id: $gid}), (b:Entity {group_id: $gid}), "
            f"p = shortestPath((a)-[:RELATES_TO*1..{max_depth}]-(b)) "
            "WHERE a.entity_key = $from AND b.entity_key = $to "
            "RETURN [n IN nodes(p) | n.entity_key] AS keys, "
            "[r IN relationships(p) | [startNode(r).entity_key, r.name, endNode(r).entity_key]] AS rels "
            "LIMIT 1"
        )
        recs = self._query(cypher, {"gid": pot_id, "from": from_key, "to": to_key})
        if not recs:
            return GraphSlice(pot_id=pot_id, nodes=(), edges=())
        rec = recs[0]
        edges = tuple(
            GraphEdge(predicate=r[1], from_key=r[0], to_key=r[2])
            for r in (rec.get("rels") or [])
        )
        nodes = self._hydrate_nodes(pot_id, list(rec.get("keys") or []))
        return GraphSlice(pot_id=pot_id, nodes=nodes, edges=edges)


__all__ = ["FalkorDBInspection"]

"""LadybugDB graph inspection — Claim-as-node BFS matching GraphInspectionPort."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from potpie_context_engine.adapters.outbound.graph.ladybug_writer import (
    _records_from_result,
    build_ladybug_db,
)
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter, ClaimQueryPort
from potpie_context_engine.core.ports.graph.inspection import (
    GraphEdge,
    GraphNode,
    GraphSlice,
)
from potpie_context_engine.domain.ports.embedder import EmbedderPort
from potpie_context_engine.domain.ports.settings import ContextEngineSettingsPort

_MAX_NODES = 2000
_MAX_EDGES = 4000
_MAX_DEPTH = 4


def _is_embedding(key: str, value: Any) -> bool:
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


class LadybugInspection:
    def __init__(
        self,
        settings: ContextEngineSettingsPort,
        *,
        conn: Any | None = None,
        conn_provider: Callable[[], Any] | None = None,
        embedder: EmbedderPort | None = None,
        claim_query: ClaimQueryPort | None = None,
    ) -> None:
        self._settings = settings
        self._conn = conn
        self._conn_provider = conn_provider
        self._embedder = embedder
        self._claim_query = claim_query

    def _get_conn(self) -> Any:
        if self._conn_provider is not None:
            return self._conn_provider()
        if self._conn is None:
            self._conn = build_ladybug_db(self._settings)
        return self._conn

    def _query(self, cypher: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        return _records_from_result(self._get_conn().execute(cypher, dict(params)))

    def _hydrate_nodes(self, pot_id: str, keys: list[str]) -> tuple[GraphNode, ...]:
        if not keys:
            return ()
        recs = self._query(
            """
            MATCH (e:Entity {group_id: $gid})
            WHERE e.entity_key IN $keys
            RETURN e.entity_key AS key, e.extra_labels AS labs,
                   e.name AS name, e.summary AS summary
            """,
            {"gid": pot_id, "keys": keys},
        )
        return tuple(
            GraphNode(
                key=str(rec["key"]),
                labels=("Entity",) + tuple(str(x) for x in (rec.get("labs") or [])),
                properties=_clean_props(
                    {"name": rec.get("name"), "summary": rec.get("summary")}
                ),
            )
            for rec in recs
            if rec.get("key")
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
                """
                MATCH (c:Claim {group_id: $gid})
                WHERE (c.subject_key IN $frontier OR c.object_key IN $frontier)
                  AND c.invalid_at IS NULL
                RETURN c.subject_key AS source, c.object_key AS target,
                       c.name AS predicate, c.claim_key AS claim_key, c.fact AS fact
                """,
                {"gid": pot_id, "frontier": list(frontier)},
            )
            new: set[str] = set()
            for rec in recs:
                src, tgt, pred = (
                    str(rec["source"]),
                    str(rec["target"]),
                    str(rec["predicate"]),
                )
                if predicate_set and pred.upper() not in predicate_set:
                    continue
                follows_out = walk_out and src in frontier
                follows_in = walk_in and tgt in frontier
                if not (follows_out or follows_in):
                    continue
                edges[(src, pred, tgt)] = GraphEdge(
                    predicate=pred,
                    from_key=src,
                    to_key=tgt,
                    properties=_clean_props(
                        {"claim_key": rec.get("claim_key"), "fact": rec.get("fact")}
                    ),
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

    def path(
        self, *, pot_id: str, from_key: str, to_key: str, max_depth: int = 4
    ) -> GraphSlice:
        max_depth = max(1, min(int(max_depth), _MAX_DEPTH))
        if from_key == to_key:
            return GraphSlice(
                pot_id=pot_id, nodes=self._hydrate_nodes(pot_id, [from_key])
            )
        parent: dict[str, tuple[str, GraphEdge] | None] = {from_key: None}
        frontier = {from_key}
        for _ in range(max_depth):
            if not frontier:
                break
            recs = self._query(
                """
                MATCH (c:Claim {group_id: $gid})
                WHERE (c.subject_key IN $frontier OR c.object_key IN $frontier)
                  AND c.invalid_at IS NULL
                RETURN c.subject_key AS source, c.object_key AS target,
                       c.name AS predicate, c.claim_key AS claim_key
                """,
                {"gid": pot_id, "frontier": list(frontier)},
            )
            nxt: set[str] = set()
            found = False
            for rec in recs:
                src, tgt = str(rec["source"]), str(rec["target"])
                edge = GraphEdge(
                    predicate=str(rec["predicate"]),
                    from_key=src,
                    to_key=tgt,
                    properties=_clean_props({"claim_key": rec.get("claim_key")}),
                )
                for a, b in ((src, tgt), (tgt, src)):
                    if a in frontier and b not in parent:
                        parent[b] = (a, edge)
                        nxt.add(b)
                        if b == to_key:
                            found = True
                            break
                if found:
                    break
            if found:
                keys: list[str] = []
                edges: list[GraphEdge] = []
                cur: str | None = to_key
                while cur is not None:
                    keys.append(cur)
                    step = parent.get(cur)
                    if step is None:
                        break
                    prev, edge = step
                    edges.append(edge)
                    cur = prev
                keys.reverse()
                edges.reverse()
                return GraphSlice(
                    pot_id=pot_id,
                    nodes=self._hydrate_nodes(pot_id, keys),
                    edges=tuple(edges),
                )
            frontier = nxt
        return GraphSlice(pot_id=pot_id, nodes=(), edges=())

    def labels(
        self, *, pot_id: str, entity_keys: Iterable[str]
    ) -> Mapping[str, tuple[str, ...]]:
        keys = list(entity_keys)
        if not keys:
            return {}
        if self._claim_query is not None:
            return self._claim_query.entity_labels(pot_id=pot_id, entity_keys=keys)
        recs = self._query(
            """
            MATCH (e:Entity {group_id: $gid})
            WHERE e.entity_key IN $keys
            RETURN e.entity_key AS key, e.extra_labels AS labs
            """,
            {"gid": pot_id, "keys": keys},
        )
        return {
            str(r["key"]): ("Entity",) + tuple(str(x) for x in (r.get("labs") or []))
            for r in recs
            if r.get("key")
        }

    def slice(self, *, pot_id: str, filter_: ClaimQueryFilter) -> GraphSlice:
        preds = list(filter_.predicate_in) or None
        node_recs = self._query(
            """
            MATCH (e:Entity {group_id: $gid})
            RETURN e.entity_key AS key, e.extra_labels AS labs,
                   e.name AS name, e.summary AS summary
            LIMIT $limit
            """,
            {"gid": pot_id, "limit": _MAX_NODES + 1},
        )
        edge_recs = self._query(
            """
            MATCH (c:Claim {group_id: $gid})
            WHERE ($include_invalid OR c.invalid_at IS NULL)
              AND (NOT $has_preds OR c.name IN $preds)
            RETURN c.subject_key AS source, c.object_key AS target,
                   c.name AS predicate, c.claim_key AS claim_key, c.fact AS fact
            LIMIT $limit
            """,
            {
                "gid": pot_id,
                "preds": preds or [],
                "has_preds": bool(preds),
                "include_invalid": bool(filter_.include_invalidated),
                "limit": _MAX_EDGES + 1,
            },
        )
        nodes = tuple(
            GraphNode(
                key=str(rec["key"]),
                labels=("Entity",) + tuple(str(x) for x in (rec.get("labs") or [])),
                properties=_clean_props(
                    {"name": rec.get("name"), "summary": rec.get("summary")}
                ),
            )
            for rec in node_recs
            if rec.get("key")
        )
        edges = tuple(
            GraphEdge(
                predicate=str(rec["predicate"]),
                from_key=str(rec["source"]),
                to_key=str(rec["target"]),
                properties=_clean_props(
                    {"claim_key": rec.get("claim_key"), "fact": rec.get("fact")}
                ),
            )
            for rec in edge_recs
            if rec.get("source") and rec.get("target")
        )
        return GraphSlice(
            pot_id=pot_id,
            nodes=nodes[:_MAX_NODES],
            edges=edges[:_MAX_EDGES],
            truncated=len(node_recs) > _MAX_NODES or len(edge_recs) > _MAX_EDGES,
        )


__all__ = ["LadybugInspection"]

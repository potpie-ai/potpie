"""FalkorDB :class:`ClaimQueryPort` — read surface for the P9 readers.

Parity with :class:`Neo4jClaimQueryStore`: same canonical ``:RELATES_TO``
claim shape, same filters, same ``ClaimRow`` output. The Phase-0 spike proved
the entire ``_FIND_CLAIMS_CYPHER`` (IN lists, IS NULL guards, temporal
predicates, ``labels()`` filters) runs unchanged on FalkorDB, so this adapter
**reuses** the Neo4j query constants and row-parsing helpers and only swaps the
driver call + record normalization (FalkorDB returns ``result_set`` rows, not
Neo4j ``Record`` maps).

``fact_query`` uses FalkorDB's native relationship vector index when an embedder
is wired. If the vector index/procedure is unavailable, the adapter falls back
to the labeled lexical scorer so local/dev profiles still return useful rows.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from adapters.outbound.graph.falkordb_writer import (
    _records_from_result,
    build_falkordb_graph,
)
from adapters.outbound.graph.canonical_claim_query import (
    ENTITY_LABELS_CYPHER,
    FIND_CLAIMS_CYPHER,
    iso,
    row_from_record,
    stamp_scored_rows,
    stamp_similarity,
)
from domain.ports.claim_query import ClaimQueryFilter, ClaimRow
from domain.ports.embedder import EmbedderPort
from domain.ports.settings import ContextEngineSettingsPort

_VECTOR_CLAIMS_CYPHER = """
CALL db.idx.vector.queryRelationships(
    'RELATES_TO',
    'fact_embedding',
    $k,
    vecf32($embedding)
) YIELD relationship AS r, score
WITH r, score
MATCH (a:Entity {group_id: $gid})-[rel:RELATES_TO]->(b:Entity {group_id: $gid})
WHERE id(rel) = id(r)
  AND rel.group_id = $gid
  AND ($preds IS NULL OR rel.name IN $preds)
  AND ($subjects IS NULL OR rel.subject_key IN $subjects)
  AND ($objects IS NULL OR rel.object_key IN $objects)
  AND ($claim_keys IS NULL OR rel.claim_key IN $claim_keys)
  AND ($subgraphs IS NULL OR rel.subgraph IN $subgraphs)
  AND ($mutation_ids IS NULL OR rel.mutation_id IN $mutation_ids)
  AND ($source_refs IS NULL OR rel.source_ref IN $source_refs OR any(ref IN coalesce(rel.source_refs, []) WHERE ref IN $source_refs))
  AND ($sources IS NULL OR rel.source_system IN $sources)
  AND ($include_invalid OR rel.invalid_at IS NULL)
  AND ($as_of IS NULL OR rel.valid_at IS NULL OR rel.valid_at <= $as_of)
  AND ($va_after IS NULL OR (rel.valid_at IS NOT NULL AND rel.valid_at >= $va_after))
  AND ($va_before IS NULL OR rel.valid_at IS NULL OR rel.valid_at <= $va_before)
  AND ($subject_label IS NULL OR $subject_label IN labels(a))
  AND ($object_label IS NULL OR $object_label IN labels(b))
RETURN rel{.*} AS props, score
ORDER BY score ASC
LIMIT $limit
"""

_ENTITY_PROPERTIES_CYPHER = """
MATCH (e:Entity {group_id: $gid})
WHERE e.entity_key = $key
RETURN properties(e) AS props
LIMIT 1
"""


def _distance_to_similarity(distance: float) -> float:
    """FalkorDB vector queries return distance; readers expect similarity."""
    return max(0.0, min(1.0, 1.0 - distance))


class FalkorDBClaimQueryStore:
    """:class:`ClaimQueryPort` over canonical ``:RELATES_TO`` edges in FalkorDB."""

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
        self._embedder = embedder

    @property
    def match_mode(self) -> str:
        return "vector" if self._embedder is not None else "lexical"

    def _get_graph(self) -> Any:
        if self._graph is None:
            self._graph = (
                self._graph_provider()
                if self._graph_provider is not None
                else build_falkordb_graph(self._settings)
            )
        return self._graph

    def close(self) -> None:
        self._graph = None

    def find_claims(self, filter_: ClaimQueryFilter) -> list[ClaimRow]:
        params = {
            "gid": filter_.pot_id,
            "preds": list(filter_.predicate_in) or None,
            "subjects": list(filter_.subject_key_in) or None,
            "objects": list(filter_.object_key_in) or None,
            "claim_keys": list(filter_.claim_key_in) or None,
            "subgraphs": list(filter_.subgraph_in) or None,
            "mutation_ids": list(filter_.mutation_id_in) or None,
            "source_refs": list(filter_.source_ref_in) or None,
            "sources": list(filter_.source_system_in) or None,
            "include_invalid": bool(filter_.include_invalidated),
            "as_of": iso(filter_.as_of),
            "va_after": iso(filter_.valid_at_after),
            "va_before": iso(filter_.valid_at_before),
            "subject_label": filter_.subject_label,
            "object_label": filter_.object_label,
        }
        if filter_.fact_query and self._embedder is not None:
            rows = self._find_claims_vector(filter_, params)
            if rows:
                return rows

        rows = self._find_claims_lexical(params)

        if filter_.fact_query:
            rows = stamp_similarity(rows, filter_.fact_query)

        if filter_.limit is not None and filter_.limit >= 0:
            rows = rows[: filter_.limit]
        return rows

    def _find_claims_lexical(self, params: Mapping[str, object]) -> list[ClaimRow]:
        result = self._get_graph().query(FIND_CLAIMS_CYPHER, params=dict(params))
        return [row_from_record(rec) for rec in _records_from_result(result)]

    def _find_claims_vector(
        self, filter_: ClaimQueryFilter, params: Mapping[str, object]
    ) -> list[ClaimRow]:
        assert filter_.fact_query is not None
        assert self._embedder is not None
        limit = filter_.limit if filter_.limit is not None and filter_.limit > 0 else 10
        vector_params = {
            **dict(params),
            "embedding": [float(x) for x in self._embedder.embed(filter_.fact_query)],
            "k": max(limit * 5, 50),
            "limit": limit,
        }
        try:
            result = self._get_graph().query(
                _VECTOR_CLAIMS_CYPHER, params=vector_params
            )
        except Exception:
            return []
        scored = [
            (
                _distance_to_similarity(float(rec.get("score", 1.0))),
                row_from_record(rec),
            )
            for rec in _records_from_result(result)
        ]
        return stamp_scored_rows(scored)

    def entity_labels(
        self, *, pot_id: str, entity_keys: Iterable[str]
    ) -> Mapping[str, tuple[str, ...]]:
        keys = list(entity_keys)
        if not keys:
            return {}
        result = self._get_graph().query(
            ENTITY_LABELS_CYPHER, params={"gid": pot_id, "keys": keys}
        )
        return {
            rec["key"]: tuple(lbl for lbl in (rec["labels"] or []))
            for rec in _records_from_result(result)
        }

    def entity_properties(self, *, pot_id: str, entity_key: str) -> dict[str, Any]:
        result = self._get_graph().query(
            _ENTITY_PROPERTIES_CYPHER,
            params={"gid": pot_id, "key": entity_key},
        )
        records = _records_from_result(result)
        if not records:
            return {}
        props = records[0].get("props")
        return dict(props) if isinstance(props, Mapping) else {}


__all__ = ["FalkorDBClaimQueryStore", "_VECTOR_CLAIMS_CYPHER"]

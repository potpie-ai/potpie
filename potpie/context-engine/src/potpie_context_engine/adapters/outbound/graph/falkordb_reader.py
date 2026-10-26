"""FalkorDB implementation of :class:`ClaimQueryPort`.

Parity with :class:`Neo4jClaimQueryStore`: same canonical ``:RELATES_TO``
claim shape, filters and ``ClaimRow`` output. The adapter shares its scan query,
row decoding and hydrated predicates with the other native readers. Temporal
comparisons use parsed instants before ranking and limiting, so legacy ISO
offsets never rely on Cypher string ordering.

``fact_query`` uses FalkorDB's native relationship vector index when an embedder
is wired. If the vector index/procedure is unavailable, the adapter falls back
to the labeled lexical scorer so local/dev profiles still return useful rows.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter, ClaimRow

from potpie_context_engine.adapters.outbound.graph.canonical_claim_query import (
    FIND_CLAIMS_CYPHER,
    claim_matches_post_filter,
    claim_query_params,
    row_from_record,
    stamp_scored_rows,
    stamp_similarity,
    vector_candidate_k,
    vector_filter_is_selective,
)
from potpie_context_engine.adapters.outbound.graph.falkordb_writer import (
    _records_from_result,
    build_falkordb_graph,
)

from potpie_context_engine.domain.ports.embedder import EmbedderPort
from potpie_context_engine.domain.ports.settings import ContextEngineSettingsPort

# Endpoint-free on purpose, twice over: re-matching each KNN hit through a
# ``(a)-[rel]->(b) WHERE id(rel) = id(r)`` pattern costs one full edge scan per
# hit (O(k * E)), and falkordb_lite additionally drops endpoint-resolving plans
# for yielded relationships after an RDB reload. Every filter below reads edge
# properties only; label filters use one batched entity lookup after hydration.
_VECTOR_CLAIMS_CYPHER = """
CALL db.idx.vector.queryRelationships(
    'RELATES_TO',
    'fact_embedding',
    $k,
    vecf32($embedding)
) YIELD relationship AS r, score
WHERE r.group_id = $gid
  AND ($include_retired OR coalesce(r.retired,false)=false)
  AND ($preds IS NULL OR r.name IN $preds)
  AND ($subjects IS NULL OR r.subject_key IN $subjects)
  AND ($objects IS NULL OR r.object_key IN $objects)
  AND ($claim_keys IS NULL OR r.claim_key IN $claim_keys)
  AND ($record_ids IS NULL OR coalesce(r.record_id,r.uuid) IN $record_ids)
  AND ($subgraphs IS NULL OR r.subgraph IN $subgraphs)
  AND ($excluded_subgraphs IS NULL OR NOT (r.subgraph IN $excluded_subgraphs))
  AND ($mutation_ids IS NULL OR r.mutation_id IN $mutation_ids)
  AND ($source_refs IS NULL OR r.source_ref IN $source_refs OR any(ref IN [] + coalesce(r.source_refs, []) WHERE ref IN $source_refs))
  AND ($sources IS NULL OR r.source_system IN $sources)
RETURN r{.*, fact_embedding: NULL} AS props, score
ORDER BY score ASC
"""

_ENTITY_PROPERTIES_CYPHER = """
MATCH (e:Entity {group_id: $gid})
WHERE e.entity_key = $key AND coalesce(e.retired,false)=false
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
        if filter_.limit == 0:
            return []
        query_time = filter_.as_of or datetime.now(timezone.utc)
        params = claim_query_params(filter_, query_time=query_time)
        # FalkorDBLite 4.18.3 corrupts memory when concatenating claim identity
        # fields inside ANY. Check text after hydration instead.
        params["exact_text"] = None
        # Stored ISO offsets do not sort by instant. Apply temporal filters
        # after hydration, before ranking and limiting.
        params.update(include_invalid=True, va_after=None, va_before=None)
        if filter_.fact_query and self._embedder is not None:
            rows = self._find_claims_vector(filter_, params, query_time=query_time)
            if rows:
                return rows

        rows = [
            row
            for row in self._find_claims_lexical(params)
            if claim_matches_post_filter(row, filter_, query_time=query_time)
        ]

        if filter_.fact_query:
            rows = stamp_similarity(rows, filter_.fact_query)

        if filter_.limit is not None and filter_.limit >= 0:
            rows = rows[: filter_.limit]
        return rows

    def _find_claims_lexical(self, params: Mapping[str, object]) -> list[ClaimRow]:
        result = self._get_graph().query(FIND_CLAIMS_CYPHER, params=dict(params))
        return [row_from_record(rec) for rec in _records_from_result(result)]

    def _find_claims_vector(
        self,
        filter_: ClaimQueryFilter,
        params: Mapping[str, object],
        *,
        query_time: datetime,
    ) -> list[ClaimRow]:
        assert filter_.fact_query is not None
        assert self._embedder is not None
        limit = filter_.limit if filter_.limit is not None and filter_.limit > 0 else 10
        k = vector_candidate_k(limit, selective=vector_filter_is_selective(filter_))
        label_filtered = bool(
            filter_.subject_label
            or filter_.object_label
            or filter_.endpoint_label
            or not filter_.include_retired
        )
        try:
            vector_params = {
                **dict(params),
                "embedding": [
                    float(x) for x in self._embedder.embed(filter_.fact_query)
                ],
                "k": k,
            }
            result = self._get_graph().query(
                _VECTOR_CLAIMS_CYPHER, params=vector_params
            )
        except Exception:
            return []
        scored = []
        for rec in _records_from_result(result):
            row = row_from_record(rec)
            if claim_matches_post_filter(row, filter_, query_time=query_time):
                scored.append(
                    (_distance_to_similarity(float(rec.get("score", 1.0))), row)
                )
        if label_filtered:
            scored = self._filter_by_entity_labels(filter_, scored)
        return stamp_scored_rows(scored[:limit])

    def _filter_by_entity_labels(
        self,
        filter_: ClaimQueryFilter,
        scored: list[tuple[float, ClaimRow]],
    ) -> list[tuple[float, ClaimRow]]:
        keys = {row.subject_key for _, row in scored if row.subject_key}
        keys |= {row.object_key for _, row in scored if row.object_key}
        labels = self._entity_labels(
            pot_id=filter_.pot_id,
            entity_keys=keys,
            include_retired=filter_.include_retired,
        )

        def keep(row: ClaimRow) -> bool:
            if not filter_.include_retired and (
                row.subject_key not in labels or row.object_key not in labels
            ):
                return False
            if filter_.subject_label is not None and filter_.subject_label not in (
                labels.get(row.subject_key) or ()
            ):
                return False
            if filter_.object_label is not None and filter_.object_label not in (
                labels.get(row.object_key) or ()
            ):
                return False
            if filter_.endpoint_label is not None and not any(
                filter_.endpoint_label in (labels.get(key) or ())
                for key in (row.subject_key, row.object_key)
            ):
                return False
            return True

        return [(score, row) for score, row in scored if keep(row)]

    def entity_labels(
        self, *, pot_id: str, entity_keys: Iterable[str]
    ) -> Mapping[str, tuple[str, ...]]:
        return self._entity_labels(pot_id=pot_id, entity_keys=entity_keys)

    def _entity_labels(
        self,
        *,
        pot_id: str,
        entity_keys: Iterable[str],
        include_retired: bool = False,
    ) -> Mapping[str, tuple[str, ...]]:
        keys = list(entity_keys)
        if not keys:
            return {}
        result = self._get_graph().query(
            "MATCH (e:Entity {group_id:$gid}) WHERE e.entity_key IN $keys "
            "AND ($include_retired OR coalesce(e.retired,false)=false) "
            "RETURN e.entity_key AS key,labels(e) AS labels",
            params={"gid": pot_id, "keys": keys, "include_retired": include_retired},
        )
        return {
            rec["key"]: tuple(lbl for lbl in (rec["labels"] or []))
            for rec in _records_from_result(result)
        }

    def entity_properties_many(
        self, *, pot_id: str, entity_keys: Iterable[str]
    ) -> dict[str, dict[str, Any]]:
        keys = list(entity_keys)
        if not keys:
            return {}
        query = "MATCH (e:Entity {group_id: $gid}) WHERE e.entity_key IN $keys AND coalesce(e.retired,false)=false RETURN e.entity_key AS key, properties(e) AS props"
        records = _records_from_result(
            self._get_graph().query(query, params={"gid": pot_id, "keys": keys})
        )
        return {rec["key"]: _public_entity_properties(rec["props"]) for rec in records}

    def entity_properties(self, *, pot_id: str, entity_key: str) -> dict[str, Any]:
        result = self._get_graph().query(
            _ENTITY_PROPERTIES_CYPHER,
            params={"gid": pot_id, "key": entity_key},
        )
        records = _records_from_result(result)
        if not records:
            return {}
        props = records[0].get("props")
        return _public_entity_properties(props)


def _public_entity_properties(props: Any) -> dict[str, Any]:
    if not isinstance(props, Mapping):
        return {}
    return {
        key: value
        for key, value in props.items()
        if key != "__potpie_snapshot_properties_v2"
    }


__all__ = ["_VECTOR_CLAIMS_CYPHER", "FalkorDBClaimQueryStore"]

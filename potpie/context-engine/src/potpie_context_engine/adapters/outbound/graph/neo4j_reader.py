"""Neo4j implementation of :class:`ClaimQueryPort`.

Queries the canonical ``:RELATES_TO`` claim edges and returns :class:`ClaimRow`s.
The adapter creates a synchronous Neo4j driver from settings on first use and
reuses it until ``close``.

``fact_query`` uses Neo4j's native relationship vector index when an embedder is
wired. If the vector procedure is unavailable, the adapter falls back to the
labeled lexical scorer so old deployments degrade instead of dropping results.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter, ClaimRow

from potpie_context_engine.adapters.outbound.graph.canonical_claim_query import (
    ENTITY_LABELS_CYPHER as _ENTITY_LABELS_CYPHER,
)
from potpie_context_engine.adapters.outbound.graph.canonical_claim_query import (
    FIND_CLAIMS_CYPHER as _FIND_CLAIMS_CYPHER,
)
from potpie_context_engine.adapters.outbound.graph.canonical_claim_query import (
    embedding_score as _embedding_score,
)
from potpie_context_engine.adapters.outbound.graph.canonical_claim_query import (
    iso as _iso,
)
from potpie_context_engine.adapters.outbound.graph.canonical_claim_query import (
    row_from_record as _row_from_record,
)
from potpie_context_engine.adapters.outbound.graph.canonical_claim_query import (
    claim_matches_post_filter,
    claim_query_params,
    stamp_scored_rows,
    stamp_similarity,
    vector_candidate_k,
    vector_filter_is_selective,
)

from potpie_context_engine.domain.ports.embedder import EmbedderPort
from potpie_context_engine.domain.ports.settings import ContextEngineSettingsPort

_VECTOR_INDEX_NAME = "claim_fact_embeddings"

_VECTOR_CLAIMS_CYPHER = """
CALL db.index.vector.queryRelationships($index_name, $k, $embedding)
YIELD relationship AS r, score
MATCH (a:Entity {group_id: $gid})-[r:RELATES_TO]->(b:Entity {group_id: $gid})
WHERE r.group_id = $gid
  AND ($include_retired OR (coalesce(r.retired,false)=false AND coalesce(a.retired,false)=false AND coalesce(b.retired,false)=false))
  AND ($preds IS NULL OR r.name IN $preds)
  AND ($subjects IS NULL OR r.subject_key IN $subjects)
  AND ($objects IS NULL OR r.object_key IN $objects)
  AND ($claim_keys IS NULL OR r.claim_key IN $claim_keys)
  AND ($record_ids IS NULL OR coalesce(r.record_id, r.uuid) IN $record_ids)
  AND ($subgraphs IS NULL OR r.subgraph IN $subgraphs)
  AND ($excluded_subgraphs IS NULL OR NOT (r.subgraph IN $excluded_subgraphs))
  AND ($mutation_ids IS NULL OR r.mutation_id IN $mutation_ids)
  AND ($source_refs IS NULL OR r.source_ref IN $source_refs OR any(ref IN [] + coalesce(r.source_refs, []) WHERE ref IN $source_refs))
  AND ($sources IS NULL OR r.source_system IN $sources)
  AND ($subject_label IS NULL OR $subject_label IN labels(a))
  AND ($object_label IS NULL OR $object_label IN labels(b))
  AND ($endpoint_label IS NULL OR $endpoint_label IN labels(a) OR $endpoint_label IN labels(b))
RETURN r{.*, fact_embedding: NULL} AS props, score
ORDER BY score DESC
"""

_ENTITY_PROPERTIES_CYPHER = """
MATCH (e:Entity {group_id: $gid})
WHERE e.entity_key = $key AND coalesce(e.retired,false)=false
RETURN properties(e) AS props
LIMIT 1
"""


class Neo4jClaimQueryStore:
    """:class:`ClaimQueryPort` over canonical ``:RELATES_TO`` edges."""

    def __init__(
        self,
        settings: ContextEngineSettingsPort,
        *,
        driver: Any | None = None,
        embedder: EmbedderPort | None = None,
    ) -> None:
        self._settings = settings
        self._driver = driver
        self._embedder = embedder

    @property
    def match_mode(self) -> str:
        return "vector" if self._embedder is not None else "lexical"

    # -- driver lifecycle --------------------------------------------------
    def _get_driver(self) -> Any:
        if self._driver is None:
            from neo4j import GraphDatabase

            uri = self._settings.neo4j_uri()
            user = self._settings.neo4j_user()
            password = self._settings.neo4j_password()
            if not uri or user is None or password is None:
                raise RuntimeError("neo4j_unavailable")
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    # -- ClaimQueryPort ----------------------------------------------------
    def find_claims(self, filter_: ClaimQueryFilter) -> list[ClaimRow]:
        # Return before the vector path substitutes its default candidate limit.
        if filter_.limit == 0:
            return []
        query_time = filter_.as_of or datetime.now(timezone.utc)
        params = claim_query_params(filter_, query_time=query_time)
        if filter_.fact_query and self._embedder is not None:
            rows = self._find_claims_vector(filter_, params, query_time=query_time)
            if rows:
                return rows

        rows = self._find_claims_lexical(filter_, params, query_time=query_time)

        if filter_.limit is not None and filter_.limit >= 0:
            rows = rows[: filter_.limit]
        return rows

    def _find_claims_lexical(
        self,
        filter_: ClaimQueryFilter,
        params: Mapping[str, object],
        *,
        query_time: datetime,
    ) -> list[ClaimRow]:
        driver = self._get_driver()
        with driver.session() as session:
            records = list(session.run(_FIND_CLAIMS_CYPHER, **params))
        rows = [
            row
            for rec in records
            if claim_matches_post_filter(
                row := _row_from_record(rec), filter_, query_time=query_time
            )
        ]
        if filter_.fact_query:
            rows = stamp_similarity(rows, filter_.fact_query)
        return rows

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
        try:
            vector_params = {
                **dict(params),
                "index_name": _VECTOR_INDEX_NAME,
                "embedding": [
                    float(x) for x in self._embedder.embed(filter_.fact_query)
                ],
                "k": vector_candidate_k(
                    limit, selective=vector_filter_is_selective(filter_)
                ),
            }
            driver = self._get_driver()
            with driver.session() as session:
                records = list(session.run(_VECTOR_CLAIMS_CYPHER, **vector_params))
        except Exception:
            return []
        rows = [
            (float(rec.get("score", 0.0)), row)
            for rec in records
            if claim_matches_post_filter(
                row := _row_from_record(rec), filter_, query_time=query_time
            )
        ]
        return stamp_scored_rows(rows[:limit])

    def entity_labels(
        self, *, pot_id: str, entity_keys: Iterable[str]
    ) -> Mapping[str, tuple[str, ...]]:
        keys = [k for k in entity_keys]
        if not keys:
            return {}
        driver = self._get_driver()
        with driver.session() as session:
            records = list(session.run(_ENTITY_LABELS_CYPHER, gid=pot_id, keys=keys))
        return {
            rec["key"]: tuple(lbl for lbl in (rec["labels"] or [])) for rec in records
        }

    def entity_properties_many(
        self, *, pot_id: str, entity_keys: Iterable[str]
    ) -> dict[str, dict[str, Any]]:
        keys = list(entity_keys)
        if not keys:
            return {}
        query = "MATCH (e:Entity {group_id: $gid}) WHERE e.entity_key IN $keys RETURN e.entity_key AS key, properties(e) AS props"
        with self._get_driver().session() as session:
            records = list(session.run(query, gid=pot_id, keys=keys))
        return {rec["key"]: _public_entity_properties(rec["props"]) for rec in records}

    def entity_properties(self, *, pot_id: str, entity_key: str) -> dict[str, Any]:
        driver = self._get_driver()
        with driver.session() as session:
            records = list(
                session.run(
                    _ENTITY_PROPERTIES_CYPHER,
                    gid=pot_id,
                    key=entity_key,
                )
            )
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


__all__ = [
    "_ENTITY_LABELS_CYPHER",
    "_ENTITY_PROPERTIES_CYPHER",
    "_FIND_CLAIMS_CYPHER",
    "_VECTOR_CLAIMS_CYPHER",
    "Neo4jClaimQueryStore",
    "_embedding_score",
    "_iso",
    "_row_from_record",
]

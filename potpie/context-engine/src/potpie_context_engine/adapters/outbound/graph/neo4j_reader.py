"""Neo4j :class:`ClaimQueryPort` — the production read surface for P9 readers.

Runs Cypher over the canonical Position-B ``:RELATES_TO`` claim edges (the
shape ``canonical_writer`` emits) and returns :class:`ClaimRow`s. This is the
keystone that makes the P9 read trunk (readers → ranking → envelope) live;
the in-memory store remains the test/offline double with identical filter
semantics.

The driver is the plain *synchronous* Neo4j driver (``ClaimQueryPort`` is a
sync surface; the readers call it inline). It is built lazily from settings
and cached for reuse.

``fact_query`` uses Neo4j's native relationship vector index when an embedder is
wired. If the vector procedure is unavailable, the adapter falls back to the
labeled lexical scorer so old deployments degrade instead of dropping results.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from potpie_context_engine.adapters.outbound.graph.canonical_claim_query import (
    ENTITY_LABELS_CYPHER as _ENTITY_LABELS_CYPHER,
    FIND_CLAIMS_CYPHER as _FIND_CLAIMS_CYPHER,
    embedding_score as _embedding_score,
    iso as _iso,
    row_from_record as _row_from_record,
    stamp_scored_rows,
    stamp_similarity,
)
from potpie_context_engine.domain.ports.claim_query import ClaimQueryFilter, ClaimRow
from potpie_context_engine.domain.ports.embedder import EmbedderPort
from potpie_context_engine.domain.ports.settings import ContextEngineSettingsPort

_VECTOR_INDEX_NAME = "claim_fact_embeddings"

_VECTOR_CLAIMS_CYPHER = """
CALL db.index.vector.queryRelationships($index_name, $k, $embedding)
YIELD relationship AS r, score
MATCH (a:Entity {group_id: $gid})-[r:RELATES_TO]->(b:Entity {group_id: $gid})
WHERE r.group_id = $gid
  AND ($preds IS NULL OR r.name IN $preds)
  AND ($subjects IS NULL OR r.subject_key IN $subjects)
  AND ($objects IS NULL OR r.object_key IN $objects)
  AND ($claim_keys IS NULL OR r.claim_key IN $claim_keys)
  AND ($subgraphs IS NULL OR r.subgraph IN $subgraphs)
  AND ($mutation_ids IS NULL OR r.mutation_id IN $mutation_ids)
  AND ($source_refs IS NULL OR r.source_ref IN $source_refs OR any(ref IN coalesce(r.source_refs, []) WHERE ref IN $source_refs))
  AND ($sources IS NULL OR r.source_system IN $sources)
  AND ($include_invalid OR r.invalid_at IS NULL)
  AND ($as_of IS NULL OR r.valid_at IS NULL OR r.valid_at <= $as_of)
  AND ($va_after IS NULL OR (r.valid_at IS NOT NULL AND r.valid_at >= $va_after))
  AND ($va_before IS NULL OR r.valid_at IS NULL OR r.valid_at <= $va_before)
  AND ($subject_label IS NULL OR $subject_label IN labels(a))
  AND ($object_label IS NULL OR $object_label IN labels(b))
RETURN r{.*} AS props, score
ORDER BY score DESC
LIMIT $limit
"""

_ENTITY_PROPERTIES_CYPHER = """
MATCH (e:Entity {group_id: $gid})
WHERE e.entity_key = $key
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
        # An explicit zero-row request must win before the vector path (which
        # coerces limit=0 → 10) or the lexical slice can return anything.
        if filter_.limit == 0:
            return []
        # Short-circuit the one hopeless filter the port documents.
        if (
            not filter_.predicate_in
            and not filter_.include_invalidated
            and not (
                filter_.subject_key_in or filter_.object_key_in or filter_.fact_query
            )
        ):
            # No predicate, no keys, no semantic anchor → unbounded; let it run
            # only when an explicit anchor exists to avoid full-partition scans.
            pass

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
            "as_of": _iso(filter_.as_of),
            "va_after": _iso(filter_.valid_at_after),
            "va_before": _iso(filter_.valid_at_before),
            "subject_label": filter_.subject_label,
            "object_label": filter_.object_label,
        }
        if filter_.fact_query and self._embedder is not None:
            rows = self._find_claims_vector(filter_, params)
            if rows:
                return rows

        rows = self._find_claims_lexical(filter_, params)

        if filter_.limit is not None and filter_.limit >= 0:
            rows = rows[: filter_.limit]
        return rows

    def _find_claims_lexical(
        self, filter_: ClaimQueryFilter, params: Mapping[str, object]
    ) -> list[ClaimRow]:
        driver = self._get_driver()
        with driver.session() as session:
            records = list(session.run(_FIND_CLAIMS_CYPHER, **params))
        rows = [_row_from_record(rec) for rec in records]
        if filter_.fact_query:
            rows = stamp_similarity(rows, filter_.fact_query)
        return rows

    def _find_claims_vector(
        self, filter_: ClaimQueryFilter, params: Mapping[str, object]
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
                "k": max(limit * 5, 50),
                "limit": limit,
            }
            driver = self._get_driver()
            with driver.session() as session:
                records = list(session.run(_VECTOR_CLAIMS_CYPHER, **vector_params))
        except Exception:
            return []
        rows = [
            (float(rec.get("score", 0.0)), _row_from_record(rec)) for rec in records
        ]
        return stamp_scored_rows(rows)

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
        return dict(props) if isinstance(props, Mapping) else {}


__all__ = [
    "Neo4jClaimQueryStore",
    "_ENTITY_LABELS_CYPHER",
    "_ENTITY_PROPERTIES_CYPHER",
    "_FIND_CLAIMS_CYPHER",
    "_VECTOR_CLAIMS_CYPHER",
    "_embedding_score",
    "_iso",
    "_row_from_record",
]

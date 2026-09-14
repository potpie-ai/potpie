"""LadybugDB :class:`ClaimQueryPort` — Claim-as-node reads + HNSW vector search."""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Mapping

from potpie_context_engine.adapters.outbound.graph.canonical_claim_query import (
    row_from_record,
    stamp_scored_rows,
    stamp_similarity,
)
from potpie_context_engine.adapters.outbound.graph.ladybug_vector import (
    LadybugVectorConfig,
    has_graph_prune_filters,
    prepare_embedding,
    query_vector_index,
)
from potpie_context_engine.adapters.outbound.graph.ladybug_writer import (
    _VECTOR_INDEX,
    _records_from_result,
    build_ladybug_db,
)
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter, ClaimRow
from potpie_context_engine.domain.ports.embedder import EmbedderPort
from potpie_context_engine.domain.ports.settings import ContextEngineSettingsPort
from potpie_context_engine.domain.retrieval_card import cosine_similarity

logger = logging.getLogger(__name__)

_FIND_CLAIMS = """
MATCH (c:Claim)
WHERE c.group_id = $gid
  AND (NOT $has_preds OR c.name IN $preds)
  AND (NOT $has_subjects OR c.subject_key IN $subjects)
  AND (NOT $has_objects OR c.object_key IN $objects)
  AND (NOT $has_claim_keys OR c.claim_key IN $claim_keys)
  AND (NOT $has_subgraphs OR c.subgraph IN $subgraphs)
  AND (NOT $has_mutation_ids OR c.mutation_id IN $mutation_ids)
  AND (NOT $has_source_refs OR c.source_ref IN $source_refs)
  AND (NOT $has_sources OR c.source_system IN $sources)
  AND ($include_invalid OR c.invalid_at IS NULL)
RETURN c.claim_key AS claim_key, c.group_id AS group_id, c.name AS name,
       c.subject_key AS subject_key, c.object_key AS object_key,
       c.fact AS fact, c.truth AS truth, c.confidence AS confidence,
       c.source_ref AS source_ref, c.source_system AS source_system,
       c.subgraph AS subgraph, c.mutation_id AS mutation_id,
       c.valid_at AS valid_at, c.invalid_at AS invalid_at,
       c.created_at AS created_at, c.expired_at AS expired_at,
       c.embedding_model AS embedding_model, c.embedding_dim AS embedding_dim,
       c.fact_embedding AS fact_embedding
"""


def _distance_to_similarity(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - float(distance)))


def _row_dict_to_props(rec: Mapping[str, Any]) -> dict[str, Any]:
    props = dict(rec)
    props.pop("distance", None)
    return props


def _claim_row_from_rec(rec: Mapping[str, Any]) -> ClaimRow:
    props = _row_dict_to_props(rec)
    return row_from_record({"props": props})


class LadybugClaimQueryStore:
    """ClaimQueryPort over Ladybug ``Claim`` nodes."""

    def __init__(
        self,
        settings: ContextEngineSettingsPort,
        *,
        conn: Any | None = None,
        conn_provider: Callable[[], Any] | None = None,
        embedder: EmbedderPort | None = None,
    ) -> None:
        self._settings = settings
        self._conn = conn
        self._conn_provider = conn_provider
        self._embedder = embedder
        self._vector_config = LadybugVectorConfig.from_settings(settings)
        self._vector_ok: bool | None = None

    @property
    def match_mode(self) -> str:
        return "vector" if self._embedder is not None else "lexical"

    def _get_conn(self) -> Any:
        if self._conn_provider is not None:
            return self._conn_provider()
        if self._conn is None:
            self._conn = build_ladybug_db(self._settings)
        return self._conn

    def close(self) -> None:
        self._conn = None

    def find_claims(self, filter_: ClaimQueryFilter) -> list[ClaimRow]:
        params = self._filter_params(filter_)
        if filter_.fact_query and self._embedder is not None:
            rows = self._find_claims_vector(filter_, params)
            if rows:
                return rows

        rows = self._find_claims_lexical(params)
        if filter_.subject_label or filter_.object_label:
            rows = self._filter_rows_by_labels(filter_, rows)
        if filter_.fact_query:
            rows = stamp_similarity(rows, filter_.fact_query)
        if filter_.limit is not None and filter_.limit >= 0:
            rows = rows[: filter_.limit]
        return rows

    def _filter_params(self, filter_: ClaimQueryFilter) -> dict[str, object]:
        preds = list(filter_.predicate_in)
        subjects = list(filter_.subject_key_in)
        objects = list(filter_.object_key_in)
        claim_keys = list(filter_.claim_key_in)
        subgraphs = list(filter_.subgraph_in)
        mutation_ids = list(filter_.mutation_id_in)
        source_refs = list(filter_.source_ref_in)
        sources = list(filter_.source_system_in)
        return {
            "gid": filter_.pot_id,
            "preds": preds,
            "has_preds": bool(preds),
            "subjects": subjects,
            "has_subjects": bool(subjects),
            "objects": objects,
            "has_objects": bool(objects),
            "claim_keys": claim_keys,
            "has_claim_keys": bool(claim_keys),
            "subgraphs": subgraphs,
            "has_subgraphs": bool(subgraphs),
            "mutation_ids": mutation_ids,
            "has_mutation_ids": bool(mutation_ids),
            "source_refs": source_refs,
            "has_source_refs": bool(source_refs),
            "sources": sources,
            "has_sources": bool(sources),
            "include_invalid": bool(filter_.include_invalidated),
        }

    def _find_claims_lexical(self, params: Mapping[str, object]) -> list[ClaimRow]:
        conn = self._get_conn()
        logger.info(
            "ladybug claim query started",
            extra={"query_kind": "claim_scan", "connection_id": id(conn)},
        )
        result = conn.execute(_FIND_CLAIMS, dict(params))
        records = _records_from_result(result)
        logger.info(
            "ladybug claim query complete",
            extra={
                "query_kind": "claim_scan",
                "connection_id": id(conn),
                "row_count": len(records),
            },
        )
        return [_claim_row_from_rec(rec) for rec in records]

    def _prepare_query_vector(self, text: str) -> list[float]:
        assert self._embedder is not None
        dim = int(getattr(self._embedder, "dimensions", 0) or 0)
        raw = [float(x) for x in self._embedder.embed(text)]
        if dim <= 0:
            return raw
        return prepare_embedding(raw, dim=dim, config=self._vector_config)

    def _find_claims_vector(
        self, filter_: ClaimQueryFilter, params: Mapping[str, object]
    ) -> list[ClaimRow]:
        assert filter_.fact_query is not None
        assert self._embedder is not None
        limit = filter_.limit if filter_.limit is not None and filter_.limit > 0 else 10
        logger.info(
            "ladybug semantic query started",
            extra={
                "query_kind": "semantic",
                "has_graph_prune_filters": has_graph_prune_filters(params),
                "limit": limit,
            },
        )
        query_vec = self._prepare_query_vector(filter_.fact_query)

        # Hybrid: graph filters first, then cosine re-rank (higher precision).
        if has_graph_prune_filters(params):
            rows = self._graph_pruned_vector(filter_, params, query_vec, limit)
            if rows:
                logger.info(
                    "ladybug semantic query complete",
                    extra={"query_kind": "cosine_fallback", "row_count": len(rows)},
                )
                return rows

        if self._vector_ok is not False:
            try:
                rows = self._native_ann_vector(filter_, params, query_vec, limit)
                if rows:
                    self._vector_ok = True
                    logger.info(
                        "ladybug semantic query complete",
                        extra={"query_kind": "native_ann", "row_count": len(rows)},
                    )
                    return rows
            except Exception:
                logger.debug("Ladybug vector query failed; cosine fallback", exc_info=True)
                self._vector_ok = False

        rows = self._cosine_fallback(filter_, params, query_vec, limit)
        logger.info(
            "ladybug semantic query complete",
            extra={"query_kind": "cosine_fallback", "row_count": len(rows)},
        )
        return rows

    def _graph_pruned_vector(
        self,
        filter_: ClaimQueryFilter,
        params: Mapping[str, object],
        query_vec: list[float],
        limit: int,
    ) -> list[ClaimRow]:
        rows = self._find_claims_lexical(params)
        scored: list[tuple[float, ClaimRow]] = []
        for row in rows:
            emb = row.fact_embedding
            if not emb:
                continue
            scored.append((cosine_similarity(query_vec, list(emb)), row))
        scored.sort(key=lambda t: t[0], reverse=True)
        if filter_.subject_label or filter_.object_label:
            scored = [(s, r) for s, r in scored if self._label_ok(filter_, r)]
        return stamp_scored_rows(scored[:limit])

    def _native_ann_vector(
        self,
        filter_: ClaimQueryFilter,
        params: Mapping[str, object],
        query_vec: list[float],
        limit: int,
    ) -> list[ClaimRow]:
        conn = self._get_conn()
        k = self._vector_config.ann_k(limit)
        result = query_vector_index(
            conn,
            embedding=query_vec,
            k=k,
            config=self._vector_config,
            index_name=_VECTOR_INDEX,
        )
        scored: list[tuple[float, ClaimRow]] = []
        for rec in _records_from_result(result):
            if not self._passes_filters(rec, params):
                continue
            row = _claim_row_from_rec(rec)
            score = _distance_to_similarity(float(rec.get("distance", 1.0)))
            scored.append((score, row))
        if filter_.subject_label or filter_.object_label:
            scored = [(s, r) for s, r in scored if self._label_ok(filter_, r)]
        return stamp_scored_rows(scored[:limit])

    def _cosine_fallback(
        self,
        filter_: ClaimQueryFilter,
        params: Mapping[str, object],
        query_vec: list[float],
        limit: int,
    ) -> list[ClaimRow]:
        rows = self._find_claims_lexical(params)
        scored: list[tuple[float, ClaimRow]] = []
        for row in rows:
            emb = row.fact_embedding
            if not emb:
                continue
            scored.append((cosine_similarity(query_vec, list(emb)), row))
        scored.sort(key=lambda t: t[0], reverse=True)
        if filter_.subject_label or filter_.object_label:
            scored = [(s, r) for s, r in scored if self._label_ok(filter_, r)]
        return stamp_scored_rows(scored[:limit])

    def _passes_filters(
        self, rec: Mapping[str, Any], params: Mapping[str, object]
    ) -> bool:
        if rec.get("group_id") != params.get("gid"):
            return False
        if not params.get("include_invalid") and rec.get("invalid_at") is not None:
            return False
        if params.get("has_preds") and rec.get("name") not in params.get("preds", []):
            return False
        if params.get("has_subjects") and rec.get("subject_key") not in params.get(
            "subjects", []
        ):
            return False
        if params.get("has_objects") and rec.get("object_key") not in params.get(
            "objects", []
        ):
            return False
        if params.get("has_claim_keys") and rec.get("claim_key") not in params.get(
            "claim_keys", []
        ):
            return False
        return True

    def _filter_rows_by_labels(
        self, filter_: ClaimQueryFilter, rows: list[ClaimRow]
    ) -> list[ClaimRow]:
        return [r for r in rows if self._label_ok(filter_, r)]

    def _label_ok(self, filter_: ClaimQueryFilter, row: ClaimRow) -> bool:
        labels = self.entity_labels(
            pot_id=filter_.pot_id,
            entity_keys={
                k
                for k in (row.subject_key, row.object_key)
                if k
            },
        )
        if filter_.subject_label is not None and filter_.subject_label not in (
            labels.get(row.subject_key) or ()
        ):
            return False
        if filter_.object_label is not None and filter_.object_label not in (
            labels.get(row.object_key) or ()
        ):
            return False
        return True

    def entity_labels(
        self, *, pot_id: str, entity_keys: Iterable[str]
    ) -> Mapping[str, tuple[str, ...]]:
        keys = list(entity_keys)
        if not keys:
            return {}
        result = self._get_conn().execute(
            """
            MATCH (e:Entity)
            WHERE e.group_id = $gid AND e.entity_key IN $keys
            RETURN e.entity_key AS key, e.extra_labels AS labs
            """,
            {"gid": pot_id, "keys": keys},
        )
        out: dict[str, tuple[str, ...]] = {}
        for rec in _records_from_result(result):
            key = str(rec.get("key") or "")
            labs = rec.get("labs") or []
            base = ("Entity",) + tuple(str(x) for x in labs)
            out[key] = base
        return out

    def entity_properties(self, *, pot_id: str, entity_key: str) -> dict[str, Any]:
        result = self._get_conn().execute(
            """
            MATCH (e:Entity {entity_key: $key})
            WHERE e.group_id = $gid
            RETURN e.entity_key AS entity_key, e.group_id AS group_id,
                   e.name AS name, e.summary AS summary,
                   e.extra_labels AS extra_labels
            LIMIT 1
            """,
            {"gid": pot_id, "key": entity_key},
        )
        rows = _records_from_result(result)
        return dict(rows[0]) if rows else {}


__all__ = ["LadybugClaimQueryStore"]

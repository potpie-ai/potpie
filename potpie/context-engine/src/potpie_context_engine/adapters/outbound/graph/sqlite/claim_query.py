"""Canonical claim reads and strict semantic ranking for SQLite."""

from __future__ import annotations

import dataclasses
import json
import math
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from potpie_context_core.graph_contract import evidence_strength_for_truth
from potpie_context_core.ports.claim_query import ClaimQueryFilter, ClaimRow
from potpie_context_engine.adapters.outbound.graph.sqlite.connection import (
    SQLiteConnectionFactory,
)
from potpie_context_engine.adapters.outbound.graph.sqlite.schema import (
    EMBEDDING_DIM,
    ensure_schema,
)
from potpie_context_engine.domain.ports.embedder import EmbedderPort

_CLAIM_RESULT_COLUMN_NAMES = (
    "claim_id",
    "pot_id",
    "predicate",
    "subject_key",
    "object_key",
    "claim_key",
    "subgraph",
    "source_system",
    "source_ref",
    "fact",
    "description",
    "retrieval_card",
    "properties_json",
    "source_refs_json",
    "evidence_json",
    "truth",
    "confidence",
    "environment",
    "valid_at",
    "valid_until",
    "invalid_at",
    "observed_at",
    "mutation_id",
    "graph_contract_version",
    "ontology_version",
)
_CLAIM_RESULT_COLUMNS = ", ".join(
    f"c.{column}" for column in _CLAIM_RESULT_COLUMN_NAMES
)


def encode_vector(vector: Sequence[float]) -> bytes:
    """Encode one finite 384-dimensional vector as little-endian float32."""

    values = tuple(float(value) for value in vector)
    if len(values) != EMBEDDING_DIM:
        raise ValueError(
            f"sqlite MiniLM embeddings must have {EMBEDDING_DIM} dimensions, "
            f"received {len(values)}"
        )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("sqlite MiniLM embeddings must contain only finite values")
    return struct.pack(f"<{EMBEDDING_DIM}f", *values)


def decode_vector(blob: bytes | bytearray | memoryview) -> tuple[float, ...]:
    raw = bytes(blob)
    expected = EMBEDDING_DIM * 4
    if len(raw) != expected:
        raise ValueError(
            f"stored sqlite embedding is {len(raw)} bytes; expected {expected}"
        )
    return tuple(struct.unpack(f"<{EMBEDDING_DIM}f", raw))


@dataclass(slots=True)
class SQLiteClaimQuery:
    """``ClaimQueryPort`` backed by canonical SQLite tables and sqlite-vec."""

    connections: SQLiteConnectionFactory
    embedder: EmbedderPort

    @property
    def match_mode(self) -> str:
        return "vector"

    def find_claims(self, filter_: ClaimQueryFilter) -> list[ClaimRow]:
        if filter_.limit == 0:
            return []
        with self.connections.connect() as connection:
            ensure_schema(connection)
            if filter_.fact_query:
                query_vector = encode_vector(self.embedder.embed(filter_.fact_query))
                if _can_use_knn(filter_):
                    return self._find_semantic_knn(
                        connection, filter_, query_vector
                    )
                return self._find_semantic_exact(
                    connection, filter_, query_vector
                )
            return self._find_canonical(connection, filter_)

    def entity_labels(
        self, *, pot_id: str, entity_keys: Iterable[str]
    ) -> Mapping[str, tuple[str, ...]]:
        keys = tuple(dict.fromkeys(str(key) for key in entity_keys if str(key)))
        if not keys:
            return {}
        with self.connections.connect() as connection:
            ensure_schema(connection)
            rows = connection.execute(
                """
                SELECT entity_key, label
                FROM entity_labels
                WHERE pot_id = ?
                  AND entity_key IN (
                      SELECT CAST(value AS TEXT) FROM json_each(?)
                  )
                ORDER BY entity_key, label
                """,
                (pot_id, json.dumps(keys)),
            )
            result: dict[str, list[str]] = {}
            for row in rows:
                result.setdefault(str(row["entity_key"]), []).append(str(row["label"]))
        return {key: tuple(labels) for key, labels in result.items()}

    def entity_properties(self, *, pot_id: str, entity_key: str) -> Mapping[str, Any]:
        with self.connections.connect() as connection:
            ensure_schema(connection)
            row = connection.execute(
                """
                SELECT properties_json
                FROM entities
                WHERE pot_id = ? AND entity_key = ?
                """,
                (pot_id, entity_key),
            ).fetchone()
        return _json_mapping(row["properties_json"]) if row is not None else {}

    def entity_properties_many(
        self, *, pot_id: str, entity_keys: Iterable[str]
    ) -> Mapping[str, Mapping[str, Any]]:
        """Backend-private bulk path used by claim-query inspection."""

        keys = tuple(dict.fromkeys(str(key) for key in entity_keys if str(key)))
        if not keys:
            return {}
        with self.connections.connect() as connection:
            ensure_schema(connection)
            rows = connection.execute(
                """
                SELECT entity_key, properties_json
                FROM entities
                WHERE pot_id = ?
                  AND entity_key IN (
                      SELECT CAST(value AS TEXT) FROM json_each(?)
                  )
                """,
                (pot_id, json.dumps(keys)),
            )
            return {
                str(row["entity_key"]): _json_mapping(row["properties_json"])
                for row in rows
            }

    def _find_canonical(
        self,
        connection: sqlite3.Connection,
        filter_: ClaimQueryFilter,
    ) -> list[ClaimRow]:
        where, params = _canonical_predicate(filter_)
        sql = f"""
            SELECT {_CLAIM_RESULT_COLUMNS}
            FROM claims AS c
            WHERE {where}
            ORDER BY c.claim_id
        """
        if filter_.limit is not None and filter_.limit >= 0:
            sql += " LIMIT ?"
            params.append(int(filter_.limit))
        return [_row_from_sql(row) for row in connection.execute(sql, params)]

    def _find_semantic_exact(
        self,
        connection: sqlite3.Connection,
        filter_: ClaimQueryFilter,
        query_vector: bytes,
    ) -> list[ClaimRow]:
        where, params = _canonical_predicate(filter_)
        sql = f"""
            SELECT {_CLAIM_RESULT_COLUMNS},
                   vec_distance_cosine(c.embedding, ?) AS distance
            FROM claims AS c
            WHERE {where}
            ORDER BY distance, c.claim_id
        """
        query_params: list[Any] = [query_vector, *params]
        if filter_.limit is not None and filter_.limit >= 0:
            sql += " LIMIT ?"
            query_params.append(int(filter_.limit))
        return [
            _stamp_distance(_row_from_sql(row), float(row["distance"]))
            for row in connection.execute(sql, query_params)
        ]

    def _find_semantic_knn(
        self,
        connection: sqlite3.Connection,
        filter_: ClaimQueryFilter,
        query_vector: bytes,
    ) -> list[ClaimRow]:
        assert filter_.limit is not None and filter_.limit > 0
        clauses = ["embedding MATCH ?", "k = ?", "pot_id = ?"]
        params: list[Any] = [query_vector, int(filter_.limit), filter_.pot_id]
        for column, values in (
            ("predicate", filter_.predicate_in),
            ("subject_key", filter_.subject_key_in),
            ("object_key", filter_.object_key_in),
            ("subgraph", filter_.subgraph_in),
            ("source_system", filter_.source_system_in),
            ("mutation_id", filter_.mutation_id_in),
        ):
            if values:
                clauses.append(f"{column} = ?")
                params.append(values[0])
        matches = sorted(
            connection.execute(
                f"""
                SELECT claim_id, distance
                FROM claim_vectors
                WHERE {' AND '.join(clauses)}
                ORDER BY distance
                """,
                params,
            ),
            key=lambda row: (float(row["distance"]), int(row["claim_id"])),
        )
        if not matches:
            return []
        ids = [int(row["claim_id"]) for row in matches]
        distances = {int(row["claim_id"]): float(row["distance"]) for row in matches}
        hydrated = {
            int(row["claim_id"]): _row_from_sql(row)
            for row in connection.execute(
                f"""
                SELECT {_CLAIM_RESULT_COLUMNS}
                FROM claims AS c
                WHERE c.claim_id IN (
                    SELECT CAST(value AS INTEGER) FROM json_each(?)
                )
                """,
                (json.dumps(ids),),
            )
        }
        return [
            _stamp_distance(hydrated[claim_id], distances[claim_id])
            for claim_id in ids
            if claim_id in hydrated
        ]


def _can_use_knn(filter_: ClaimQueryFilter) -> bool:
    """Whether vec0 metadata can express the complete current-state filter."""

    if filter_.limit is None or filter_.limit <= 0:
        return False
    if (
        filter_.include_invalidated
        or filter_.as_of is not None
        or filter_.valid_at_after is not None
        or filter_.valid_at_before is not None
        or filter_.claim_key_in
        or filter_.source_ref_in
        or filter_.subject_label is not None
        or filter_.object_label is not None
    ):
        return False
    return all(
        len(values) <= 1
        for values in (
            filter_.predicate_in,
            filter_.subject_key_in,
            filter_.object_key_in,
            filter_.subgraph_in,
            filter_.source_system_in,
            filter_.mutation_id_in,
        )
    )


def _canonical_predicate(filter_: ClaimQueryFilter) -> tuple[str, list[Any]]:
    clauses = ["c.pot_id = ?"]
    params: list[Any] = [filter_.pot_id]
    for column, values in (
        ("predicate", filter_.predicate_in),
        ("subject_key", filter_.subject_key_in),
        ("object_key", filter_.object_key_in),
        ("claim_key", filter_.claim_key_in),
        ("subgraph", filter_.subgraph_in),
        ("mutation_id", filter_.mutation_id_in),
        ("source_system", filter_.source_system_in),
    ):
        if values:
            clauses.append(
                f"""
                c.{column} IN (
                    SELECT CAST(value AS TEXT) FROM json_each(?)
                )
                """
            )
            params.append(json.dumps(values))

    if filter_.source_ref_in:
        wanted = tuple(
            dict.fromkeys(
                str(value).strip().lower()
                for value in filter_.source_ref_in
                if str(value).strip()
            )
        )
        if wanted:
            wanted_json = json.dumps(wanted)
            clauses.append(
                """
                (
                    lower(c.source_ref) IN (
                        SELECT lower(CAST(value AS TEXT)) FROM json_each(?)
                    )
                    OR EXISTS (
                        SELECT 1 FROM json_each(c.source_refs_json) AS refs
                        WHERE lower(CAST(refs.value AS TEXT)) IN (
                            SELECT lower(CAST(value AS TEXT)) FROM json_each(?)
                        )
                    )
                    OR EXISTS (
                        SELECT 1 FROM json_each(c.evidence_json) AS evidence
                        WHERE lower(
                            coalesce(
                                json_extract(evidence.value, '$.source_ref'),
                                json_extract(evidence.value, '$.ref')
                            )
                        ) IN (
                            SELECT lower(CAST(value AS TEXT)) FROM json_each(?)
                        )
                    )
                )
                """
            )
            params.extend((wanted_json, wanted_json, wanted_json))

    if not filter_.include_invalidated:
        clauses.append("c.invalid_at IS NULL")
    if filter_.valid_at_after is not None:
        clauses.append("c.valid_at IS NOT NULL AND c.valid_at >= ?")
        params.append(_iso(filter_.valid_at_after))
    if filter_.valid_at_before is not None:
        clauses.append("(c.valid_at IS NULL OR c.valid_at <= ?)")
        params.append(_iso(filter_.valid_at_before))
    if filter_.as_of is not None:
        clauses.append("(c.valid_at IS NULL OR c.valid_at <= ?)")
        params.append(_iso(filter_.as_of))
    if filter_.subject_label:
        clauses.append(
            """
            EXISTS (
                SELECT 1 FROM entity_labels AS subject_labels
                WHERE subject_labels.pot_id = c.pot_id
                  AND subject_labels.entity_key = c.subject_key
                  AND subject_labels.label = ?
            )
            """
        )
        params.append(filter_.subject_label)
    if filter_.object_label:
        clauses.append(
            """
            EXISTS (
                SELECT 1 FROM entity_labels AS object_labels
                WHERE object_labels.pot_id = c.pot_id
                  AND object_labels.entity_key = c.object_key
                  AND object_labels.label = ?
            )
            """
        )
        params.append(filter_.object_label)
    return " AND ".join(f"({clause.strip()})" for clause in clauses), params


def _row_from_sql(row: sqlite3.Row) -> ClaimRow:
    truth = _optional_str(row["truth"])
    return ClaimRow(
        pot_id=str(row["pot_id"]),
        predicate=str(row["predicate"]),
        subject_key=str(row["subject_key"]),
        object_key=str(row["object_key"]),
        valid_at=_parse_dt(row["valid_at"]),
        invalid_at=_parse_dt(row["invalid_at"]),
        evidence_strength=evidence_strength_for_truth(truth),
        source_system=_optional_str(row["source_system"]),
        source_ref=_optional_str(row["source_ref"]),
        fact=_optional_str(row["fact"]),
        properties=_json_mapping(row["properties_json"]),
        # Embeddings stay canonical in SQLite and are intentionally omitted
        # from ordinary hydrated rows, matching the other durable backends.
        fact_embedding=None,
        claim_key=_optional_str(row["claim_key"]),
        subgraph=_optional_str(row["subgraph"]),
        truth=truth,
        confidence=float(row["confidence"]) if row["confidence"] is not None else None,
        description=_optional_str(row["description"]),
        environment=_optional_str(row["environment"]),
        observed_at=_parse_dt(row["observed_at"]),
        valid_until=_parse_dt(row["valid_until"]),
        mutation_id=_optional_str(row["mutation_id"]),
        source_refs=_json_str_tuple(row["source_refs_json"]),
        evidence=_json_evidence(row["evidence_json"]),
        graph_contract_version=_optional_str(row["graph_contract_version"]),
        ontology_version=_optional_str(row["ontology_version"]),
    )


def _stamp_distance(row: ClaimRow, distance: float) -> ClaimRow:
    properties = dict(row.properties)
    properties["semantic_similarity"] = max(0.0, min(1.0, 1.0 - distance))
    return dataclasses.replace(row, properties=properties)


def _json_value(raw: Any, fallback: Any) -> Any:
    if not isinstance(raw, str):
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def _json_mapping(raw: Any) -> dict[str, Any]:
    value = _json_value(raw, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _json_str_tuple(raw: Any) -> tuple[str, ...]:
    value = _json_value(raw, [])
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if item is not None and str(item))


def _json_evidence(raw: Any) -> tuple[Mapping[str, Any], ...]:
    value = _json_value(raw, [])
    if not isinstance(value, list):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, Mapping))


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


__all__ = ["SQLiteClaimQuery", "decode_vector", "encode_vector"]

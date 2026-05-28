"""Neo4j :class:`ClaimQueryPort` — the production read surface for P9 readers.

Runs Cypher over the canonical Position-B ``:RELATES_TO`` claim edges (the
shape ``canonical_writer`` emits) and returns :class:`ClaimRow`s. This is the
keystone that makes the P9 read trunk (readers → ranking → envelope) live;
the in-memory store remains the test/offline double with identical filter
semantics.

The driver is the plain *synchronous* Neo4j driver (``ClaimQueryPort`` is a
sync surface; the readers call it inline). It is built lazily from settings
and cached for reuse.

``fact_query`` (semantic) currently uses the same token-overlap scoring as the
in-memory store rather than the native relationship vector index — the
embed-on-write/embed-query path isn't wired yet. When an embedder lands, swap
the Python scoring here for ``db.index.vector.queryRelationships`` over
``r.fact_embedding`` (the index name is ``claim_fact_embeddings``); the row
shape and the ``properties['semantic_similarity']`` stamp stay identical.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Iterable, Mapping

from domain.ports.claim_query import ClaimQueryFilter, ClaimRow
from domain.ports.settings import ContextEngineSettingsPort

# Edge properties that are part of the canonical identity/bitemporal frame and
# are surfaced as first-class ClaimRow fields; everything else on the edge is
# passed through in ``ClaimRow.properties`` for the readers (code_scope,
# policy_kind, environment, corroboration_count, …).
_RESERVED_EDGE_KEYS = frozenset(
    {
        "group_id",
        "name",
        "subject_key",
        "object_key",
        "valid_at",
        "invalid_at",
        "evidence_strength",
        "source_system",
        "source_ref",
        "fact",
        "fact_embedding",
    }
)


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    # neo4j.time.DateTime → native
    to_native = getattr(value, "to_native", None)
    if callable(to_native):
        try:
            native = to_native()
        except Exception:  # noqa: BLE001
            return None
        return native if isinstance(native, datetime) else None
    return None


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _row_from_record(rec: Mapping[str, Any]) -> ClaimRow:
    """Build a :class:`ClaimRow` from one Cypher record (``r{.*}`` props map)."""
    props = dict(rec.get("props") or {})
    extras = {k: v for k, v in props.items() if k not in _RESERVED_EDGE_KEYS}
    return ClaimRow(
        pot_id=str(props.get("group_id") or rec.get("gid") or ""),
        predicate=str(props.get("name") or ""),
        subject_key=str(props.get("subject_key") or ""),
        object_key=str(props.get("object_key") or ""),
        valid_at=_parse_dt(props.get("valid_at")),
        invalid_at=_parse_dt(props.get("invalid_at")),
        evidence_strength=str(props.get("evidence_strength") or "stated"),
        source_system=props.get("source_system"),
        source_ref=props.get("source_ref"),
        fact=props.get("fact"),
        properties=extras,
    )


def _embedding_score(fact: str | None, query: str) -> float:
    """Token-overlap stand-in for native vector similarity (mirrors in-memory)."""
    if not fact:
        return 0.0
    a = set(query.lower().split())
    b = set(fact.lower().split())
    if not a or not b:
        return 0.0
    base = len(a & b) / len(a | b)
    if a & b:
        base = math.sqrt(base)
    return max(0.0, min(1.0, base))


_FIND_CLAIMS_CYPHER = """
MATCH (a:Entity {group_id: $gid})-[r:RELATES_TO {group_id: $gid}]->(b:Entity {group_id: $gid})
WHERE ($preds IS NULL OR r.name IN $preds)
  AND ($subjects IS NULL OR r.subject_key IN $subjects)
  AND ($objects IS NULL OR r.object_key IN $objects)
  AND ($sources IS NULL OR r.source_system IN $sources)
  AND ($include_invalid OR r.invalid_at IS NULL)
  AND ($as_of IS NULL OR r.valid_at IS NULL OR r.valid_at <= $as_of)
  AND ($va_after IS NULL OR (r.valid_at IS NOT NULL AND r.valid_at >= $va_after))
  AND ($va_before IS NULL OR r.valid_at IS NULL OR r.valid_at <= $va_before)
  AND ($subject_label IS NULL OR $subject_label IN labels(a))
  AND ($object_label IS NULL OR $object_label IN labels(b))
RETURN r{.*} AS props
"""

_ENTITY_LABELS_CYPHER = """
MATCH (e:Entity {group_id: $gid})
WHERE e.entity_key IN $keys
RETURN e.entity_key AS key, labels(e) AS labels
"""


class Neo4jClaimQueryStore:
    """:class:`ClaimQueryPort` over canonical ``:RELATES_TO`` edges."""

    def __init__(self, settings: ContextEngineSettingsPort, *, driver: Any | None = None) -> None:
        self._settings = settings
        self._driver = driver

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
        # Short-circuit the one hopeless filter the port documents.
        if not filter_.predicate_in and not filter_.include_invalidated and not (
            filter_.subject_key_in or filter_.object_key_in or filter_.fact_query
        ):
            # No predicate, no keys, no semantic anchor → unbounded; let it run
            # only when an explicit anchor exists to avoid full-partition scans.
            pass

        params = {
            "gid": filter_.pot_id,
            "preds": list(filter_.predicate_in) or None,
            "subjects": list(filter_.subject_key_in) or None,
            "objects": list(filter_.object_key_in) or None,
            "sources": list(filter_.source_system_in) or None,
            "include_invalid": bool(filter_.include_invalidated),
            "as_of": _iso(filter_.as_of),
            "va_after": _iso(filter_.valid_at_after),
            "va_before": _iso(filter_.valid_at_before),
            "subject_label": filter_.subject_label,
            "object_label": filter_.object_label,
        }
        driver = self._get_driver()
        with driver.session() as session:
            records = list(session.run(_FIND_CLAIMS_CYPHER, **params))
        rows = [_row_from_record(rec) for rec in records]

        if filter_.fact_query:
            scored = sorted(
                (
                    (_embedding_score(row.fact, filter_.fact_query), row)
                    for row in rows
                ),
                key=lambda pair: pair[0],
                reverse=True,
            )
            rows = []
            for score, row in scored:
                props = dict(row.properties)
                props["semantic_similarity"] = score
                rows.append(
                    ClaimRow(
                        pot_id=row.pot_id,
                        predicate=row.predicate,
                        subject_key=row.subject_key,
                        object_key=row.object_key,
                        valid_at=row.valid_at,
                        invalid_at=row.invalid_at,
                        evidence_strength=row.evidence_strength,
                        source_system=row.source_system,
                        source_ref=row.source_ref,
                        fact=row.fact,
                        properties=props,
                        fact_embedding=row.fact_embedding,
                    )
                )

        if filter_.limit is not None and filter_.limit >= 0:
            rows = rows[: filter_.limit]
        return rows

    def entity_labels(
        self, *, pot_id: str, entity_keys: Iterable[str]
    ) -> Mapping[str, tuple[str, ...]]:
        keys = [k for k in entity_keys]
        if not keys:
            return {}
        driver = self._get_driver()
        with driver.session() as session:
            records = list(
                session.run(_ENTITY_LABELS_CYPHER, gid=pot_id, keys=keys)
            )
        return {
            rec["key"]: tuple(lbl for lbl in (rec["labels"] or []))
            for rec in records
        }


__all__ = ["Neo4jClaimQueryStore"]

"""FalkorDB :class:`ClaimQueryPort` — read surface for the P9 readers.

Parity with :class:`Neo4jClaimQueryStore`: same canonical ``:RELATES_TO``
claim shape, same filters, same ``ClaimRow`` output. This adapter **reuses**
the Neo4j query constants and row-parsing helpers and only swaps the driver
call + record normalization (FalkorDB returns ``result_set`` rows, not Neo4j
``Record`` maps). Claim queries never reference the edge endpoints — entity
label filters are applied in Python — because embedded FalkorDB silently
drops rows from endpoint-resolving plans after a persistence reload (see
``FIND_CLAIMS_CYPHER``).

``fact_query`` merges two passes: the lexical query defines which claims are
visible (an unembedded claim must never disappear from reads), and FalkorDB's
native relationship vector index — when an embedder is wired — overlays real
similarity scores on top. A failing vector index degrades ranking, loudly,
never visibility.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Mapping

from adapters.outbound.graph.falkordb_writer import (
    _records_from_result,
    build_falkordb_graph,
)
from adapters.outbound.graph.canonical_claim_query import (
    ENTITY_LABELS_CYPHER,
    FIND_CLAIMS_CYPHER,
    filter_rows_by_labels,
    iso,
    merge_vector_scored_rows,
    row_from_record,
)
from domain.ports.claim_query import ClaimQueryFilter, ClaimRow
from domain.ports.embedder import EmbedderPort
from domain.ports.settings import ContextEngineSettingsPort

logger = logging.getLogger(__name__)

# No MATCH after the procedure call: binding endpoint nodes puts the edge scan
# under a bound node scan — the embedded-FalkorDB plan shape that returns zero
# rows when the bound node is internal id 0 (see FIND_CLAIMS_CYPHER). Every
# filter here is an edge property; entity-label filters are re-applied by the
# caller in Python. ``score`` is a cosine *distance* on FalkorDB.
_VECTOR_CLAIMS_CYPHER = """
CALL db.idx.vector.queryRelationships(
    'RELATES_TO',
    'fact_embedding',
    $k,
    vecf32($embedding)
) YIELD relationship AS r, score
WITH r, score
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
RETURN r{.*} AS props, score
ORDER BY score ASC
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
        self._vector_query_warned = False

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
        # An explicit zero-row request must win before the vector path (which
        # coerces limit=0 → 10) or the lexical slice can return anything.
        if filter_.limit == 0:
            return []
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
        }
        # Lexical rows define membership: a claim must stay readable even when
        # its embedding is missing or stale. Vector results only overlay
        # ranking scores (plus defense-in-depth extras) on top.
        rows = filter_rows_by_labels(
            self._find_claims_lexical(params), filter_, self.entity_labels
        )

        if filter_.fact_query:
            vector_scored = (
                self._vector_scored(filter_, params)
                if self._embedder is not None
                else []
            )
            rows = merge_vector_scored_rows(
                rows,
                vector_scored,
                filter_.fact_query,
                admit_extra=self._label_filter_check(filter_),
            )

        if filter_.limit is not None and filter_.limit >= 0:
            rows = rows[: filter_.limit]
        return rows

    def _find_claims_lexical(self, params: Mapping[str, object]) -> list[ClaimRow]:
        result = self._get_graph().query(FIND_CLAIMS_CYPHER, params=dict(params))
        return [row_from_record(rec) for rec in _records_from_result(result)]

    def _vector_scored(
        self, filter_: ClaimQueryFilter, params: Mapping[str, object]
    ) -> list[tuple[float, ClaimRow]]:
        assert filter_.fact_query is not None
        assert self._embedder is not None
        limit = filter_.limit if filter_.limit is not None and filter_.limit > 0 else 10
        try:
            vector_params = {
                **dict(params),
                "embedding": [
                    float(x) for x in self._embedder.embed(filter_.fact_query)
                ],
                "k": max(limit * 5, 50),
            }
            result = self._get_graph().query(
                _VECTOR_CLAIMS_CYPHER, params=vector_params
            )
        except Exception as exc:  # noqa: BLE001
            # Degrading to lexical scores must not be silent: a broken vector
            # index (missing, or built for a different embedder dimension)
            # quietly flattens semantic ranking otherwise.
            level = logging.DEBUG if self._vector_query_warned else logging.WARNING
            self._vector_query_warned = True
            logger.log(
                level,
                "falkordb vector query failed (%s); using lexical scores — "
                "if the embedder changed, run 'potpie graph repair semantic_index'",
                exc,
            )
            return []
        return [
            (
                _distance_to_similarity(float(rec.get("score", 1.0))),
                row_from_record(rec),
            )
            for rec in _records_from_result(result)
        ]

    def _label_filter_check(
        self, filter_: ClaimQueryFilter
    ) -> Callable[[ClaimRow], bool] | None:
        """Re-apply entity-label filters to vector-only rows in Python."""
        if not (filter_.subject_label or filter_.object_label):
            return None

        def check(row: ClaimRow) -> bool:
            labels = self.entity_labels(
                pot_id=filter_.pot_id,
                entity_keys=[row.subject_key, row.object_key],
            )
            if filter_.subject_label and filter_.subject_label not in labels.get(
                row.subject_key, ()
            ):
                return False
            if filter_.object_label and filter_.object_label not in labels.get(
                row.object_key, ()
            ):
                return False
            return True

        return check

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

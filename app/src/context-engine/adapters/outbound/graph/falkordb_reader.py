"""FalkorDB :class:`ClaimQueryPort` — read surface for the P9 readers.

Parity with :class:`Neo4jClaimQueryStore`: same canonical ``:RELATES_TO``
claim shape, same filters, same ``ClaimRow`` output. The Phase-0 spike proved
the entire ``_FIND_CLAIMS_CYPHER`` (IN lists, IS NULL guards, temporal
predicates, ``labels()`` filters) runs unchanged on FalkorDB, so this adapter
**reuses** the Neo4j query constants and row-parsing helpers and only swaps the
driver call + record normalization (FalkorDB returns ``result_set`` rows, not
Neo4j ``Record`` maps).

``fact_query`` keeps the Python token-overlap scoring (the native relationship
vector index is not on the hot path; see history-local-graph-db.md).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from adapters.outbound.graph.falkordb_writer import (
    _records_from_result,
    build_falkordb_graph,
)
from adapters.outbound.graph.neo4j_reader import (
    _ENTITY_LABELS_CYPHER,
    _FIND_CLAIMS_CYPHER,
    _embedding_score,
    _iso,
    _row_from_record,
)
from domain.ports.claim_query import ClaimQueryFilter, ClaimRow
from domain.ports.settings import ContextEngineSettingsPort


class FalkorDBClaimQueryStore:
    """:class:`ClaimQueryPort` over canonical ``:RELATES_TO`` edges in FalkorDB."""

    def __init__(
        self,
        settings: ContextEngineSettingsPort,
        *,
        graph: Any | None = None,
        graph_provider: Callable[[], Any] | None = None,
    ) -> None:
        self._settings = settings
        self._graph = graph  # injectable for unit tests
        self._graph_provider = graph_provider  # shared handle from the container

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
            "sources": list(filter_.source_system_in) or None,
            "include_invalid": bool(filter_.include_invalidated),
            "as_of": _iso(filter_.as_of),
            "va_after": _iso(filter_.valid_at_after),
            "va_before": _iso(filter_.valid_at_before),
            "subject_label": filter_.subject_label,
            "object_label": filter_.object_label,
        }
        result = self._get_graph().query(_FIND_CLAIMS_CYPHER, params=params)
        rows = [_row_from_record(rec) for rec in _records_from_result(result)]

        if filter_.fact_query:
            scored = sorted(
                ((_embedding_score(row.fact, filter_.fact_query), row) for row in rows),
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
        keys = list(entity_keys)
        if not keys:
            return {}
        result = self._get_graph().query(
            _ENTITY_LABELS_CYPHER, params={"gid": pot_id, "keys": keys}
        )
        return {
            rec["key"]: tuple(lbl for lbl in (rec["labels"] or []))
            for rec in _records_from_result(result)
        }


__all__ = ["FalkorDBClaimQueryStore"]

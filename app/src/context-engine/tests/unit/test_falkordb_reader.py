"""Unit tests for the FalkorDB ClaimQueryPort adapter.

No live FalkorDB: an injected fake graph returns canned ``result_set`` rows so
we exercise param-building, row parsing (reusing ``_row_from_record``), the
fact_query token-overlap stamping + ordering, limit truncation, and
``entity_labels`` — mirroring ``test_neo4j_claim_query`` for parity.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from adapters.outbound.graph.backends.falkor.reader import FalkorDBClaimQueryStore
from domain.ports.claim_query import ClaimQueryFilter

pytestmark = pytest.mark.unit


class _FakeResult:
    def __init__(self, header, result_set):
        self.header = header
        self.result_set = result_set


class _FakeGraph:
    def __init__(self, header, result_set):
        self._header = header
        self._result_set = result_set
        self.captured: list[tuple[str, dict]] = []

    def query(self, cypher, params=None):
        self.captured.append((cypher, params or {}))
        return _FakeResult(self._header, self._result_set)


def _props_graph(*prop_dicts):
    return _FakeGraph(header=[[1, "props"]], result_set=[[p] for p in prop_dicts])


def test_find_claims_builds_params_and_parses_rows() -> None:
    graph = _props_graph(
        {
            "group_id": "p1",
            "name": "DEPENDS_ON",
            "subject_key": "service:web",
            "object_key": "service:auth",
            "valid_at": "2026-02-01T00:00:00+00:00",
            "evidence_strength": "attested",
        }
    )
    store = FalkorDBClaimQueryStore(settings=object(), graph=graph)  # type: ignore[arg-type]
    rows = store.find_claims(
        ClaimQueryFilter(
            pot_id="p1",
            predicate_in=("DEPENDS_ON",),
            as_of=datetime(2026, 3, 1, tzinfo=timezone.utc),
            limit=10,
        )
    )
    assert len(rows) == 1
    assert rows[0].predicate == "DEPENDS_ON"
    assert rows[0].subject_key == "service:web"
    _, params = graph.captured[0]
    assert params["gid"] == "p1"
    assert params["preds"] == ["DEPENDS_ON"]
    assert params["include_invalid"] is False
    assert params["as_of"] == "2026-03-01T00:00:00+00:00"


def test_fact_query_stamps_similarity_and_orders() -> None:
    graph = _props_graph(
        {"group_id": "p1", "name": "X", "subject_key": "a", "object_key": "b",
         "fact": "connection pool exhausted in checkout"},
        {"group_id": "p1", "name": "X", "subject_key": "c", "object_key": "d",
         "fact": "unrelated note about logging"},
    )
    store = FalkorDBClaimQueryStore(settings=object(), graph=graph)  # type: ignore[arg-type]
    rows = store.find_claims(ClaimQueryFilter(pot_id="p1", fact_query="connection pool exhausted"))
    assert rows[0].subject_key == "a"
    assert rows[0].properties["semantic_similarity"] > rows[1].properties["semantic_similarity"]


def test_limit_truncates() -> None:
    graph = _props_graph(
        *[{"group_id": "p1", "name": "X", "subject_key": str(i), "object_key": "o"} for i in range(5)]
    )
    store = FalkorDBClaimQueryStore(settings=object(), graph=graph)  # type: ignore[arg-type]
    rows = store.find_claims(ClaimQueryFilter(pot_id="p1", predicate_in=("X",), limit=2))
    assert len(rows) == 2


def test_entity_labels_maps_keys() -> None:
    graph = _FakeGraph(
        header=[[1, "key"], [1, "labels"]],
        result_set=[
            ["service:web", ["Entity", "Service"]],
            ["team:platform", ["Entity", "Team"]],
        ],
    )
    store = FalkorDBClaimQueryStore(settings=object(), graph=graph)  # type: ignore[arg-type]
    out = store.entity_labels(pot_id="p1", entity_keys=["service:web", "team:platform"])
    assert out["service:web"] == ("Entity", "Service")
    assert out["team:platform"] == ("Entity", "Team")


def test_entity_labels_empty_keys_short_circuits() -> None:
    graph = _FakeGraph(header=[], result_set=[])
    store = FalkorDBClaimQueryStore(settings=object(), graph=graph)  # type: ignore[arg-type]
    assert store.entity_labels(pot_id="p1", entity_keys=[]) == {}
    assert graph.captured == []

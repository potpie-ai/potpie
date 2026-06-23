"""Unit tests for the FalkorDB graph-inspection adapter.

A fake graph handle returns canned ``result_set`` rows (the FalkorDB row shape
``_records_from_result`` normalizes), so these run without redislite.
"""

from __future__ import annotations

from typing import Any

from context_engine.adapters.outbound.graph.falkordb_inspection import FalkorDBInspection
from context_engine.domain.ports.claim_query import ClaimQueryFilter

POT = "pot_test"


class _FakeResult:
    def __init__(self, names: list[str], rows: list[list[Any]]) -> None:
        self.header = [[1, n] for n in names]
        self.result_set = rows


class _FakeGraph:
    """Routes the inspection cyphers to canned results by content."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> _FakeResult:
        self.queries.append(cypher)
        if "$frontier" in cypher:  # neighborhood incident edges
            return _FakeResult(
                ["source", "target", "predicate"],
                [["person:y", "repo:x", "PERFORMED"]],
            )
        if "$keys" in cypher:  # hydrate nodes
            return _FakeResult(
                ["key", "labels", "props"],
                [
                    ["repo:x", ["Entity", "Repository"], {"name": "x"}],
                    ["person:y", ["Entity", "Person"], {"name": "y"}],
                ],
            )
        if "a.entity_key AS source" in cypher:  # slice edges
            return _FakeResult(
                ["source", "target", "predicate"],
                [["person:y", "repo:x", "PERFORMED"]],
            )
        # slice nodes
        return _FakeResult(
            ["key", "labels", "props"],
            [
                [
                    "repo:x",
                    ["Entity", "Repository"],
                    {"name": "x", "fact_embedding": [0.1] * 64, "group_id": POT},
                ],
                ["person:y", ["Entity", "Person"], {"name": "y"}],
            ],
        )


def test_slice_returns_nodes_and_edges_and_strips_embeddings() -> None:
    insp = FalkorDBInspection(settings=object(), graph=_FakeGraph())
    sl = insp.slice(pot_id=POT, filter_=ClaimQueryFilter(pot_id=POT))
    assert sl.pot_id == POT
    assert {n.key for n in sl.nodes} == {"repo:x", "person:y"}
    repo = next(n for n in sl.nodes if n.key == "repo:x")
    assert ("Entity", "Repository") == repo.labels
    # embedding + group_id are dropped from the explorer payload
    assert "fact_embedding" not in repo.properties
    assert "group_id" not in repo.properties
    assert repo.properties["name"] == "x"
    assert len(sl.edges) == 1
    e = sl.edges[0]
    assert (e.from_key, e.predicate, e.to_key) == ("person:y", "PERFORMED", "repo:x")


def test_neighborhood_bfs_collects_incident_edges_and_hydrates() -> None:
    insp = FalkorDBInspection(settings=object(), graph=_FakeGraph())
    sl = insp.neighborhood(pot_id=POT, entity_key="repo:x", depth=1)
    assert {n.key for n in sl.nodes} == {"repo:x", "person:y"}
    assert len(sl.edges) == 1
    assert sl.edges[0].predicate == "PERFORMED"


def test_neighborhood_clamps_depth() -> None:
    g = _FakeGraph()
    insp = FalkorDBInspection(settings=object(), graph=g)
    # depth far over the cap must not loop unbounded (one frontier round here
    # since the fake always returns the same edge → BFS converges immediately).
    insp.neighborhood(pot_id=POT, entity_key="repo:x", depth=99)
    incident_rounds = sum(1 for q in g.queries if "$frontier" in q)
    assert 1 <= incident_rounds <= 4

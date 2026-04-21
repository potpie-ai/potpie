"""Causal one-hop search expansion (Level 1 hybrid retrieval)."""

from __future__ import annotations

import pytest

from application.use_cases.query_context import merge_causal_expanded_search_rows


class _StubStructural:
    def __init__(self, expanded: list[dict]) -> None:
        self._expanded = expanded
        self.calls: list[tuple[str, list[str]]] = []

    def expand_causal_neighbours(
        self,
        pot_id: str,
        node_uuids: list[str],
        *,
        depth: int = 1,
    ) -> list[dict]:
        _ = depth
        self.calls.append((pot_id, list(node_uuids)))
        return list(self._expanded)


@pytest.fixture
def enable_causal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_CAUSAL_EXPAND", "1")


def test_merge_adds_expanded_neighbor_with_decay(
    monkeypatch: pytest.MonkeyPatch, enable_causal: None
) -> None:
    stub = _StubStructural(
        [
            {
                "neighbor_uuid": "scaling-pain",
                "name": "MongoDB scaling pain",
                "summary": "Write contention and slow aggregations",
                "edge_uuid": "edge-c1",
                "edge_name": "CAUSED",
                "seed_uuid": "migration-decision",
            }
        ]
    )
    rows = [
        {
            "uuid": "edge-top",
            "name": "RELATES_TO",
            "summary": "Post-migration cluster decommission",
            "score": 0.9,
            "source_node_uuid": "decommissioned-cluster",
            "target_node_uuid": "migration-decision",
        }
    ]
    out = merge_causal_expanded_search_rows(rows, stub, "pot-1", limit=5)
    uuids = [r["uuid"] for r in out]
    assert "scaling-pain" in uuids
    extra = next(r for r in out if r["uuid"] == "scaling-pain")
    assert extra.get("causal_via", {}).get("relation") == "CAUSED"
    assert float(extra["score"]) == pytest.approx(0.9 * 0.6)


def test_merge_skips_when_neighbor_already_in_semantic_endpoints(
    monkeypatch: pytest.MonkeyPatch, enable_causal: None
) -> None:
    stub = _StubStructural(
        [
            {
                "neighbor_uuid": "already-known",
                "name": "X",
                "summary": "",
                "edge_uuid": "e2",
                "edge_name": "CAUSED",
                "seed_uuid": "seed-1",
            }
        ]
    )
    rows = [
        {
            "uuid": "e0",
            "name": "E",
            "summary": "",
            "score": 1.0,
            "source_node_uuid": "already-known",
            "target_node_uuid": "other",
        }
    ]
    out = merge_causal_expanded_search_rows(rows, stub, "pot-1", limit=5)
    assert len(out) == 1


def test_merge_disabled_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_CAUSAL_EXPAND", "0")
    stub = _StubStructural([{"neighbor_uuid": "n1", "edge_uuid": "x"}])
    rows = [{"uuid": "a", "source_node_uuid": "s", "target_node_uuid": "t", "score": 1.0}]
    out = merge_causal_expanded_search_rows(rows, stub, "pot-1", limit=5)
    assert len(out) == 1
    assert not stub.calls

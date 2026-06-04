"""Semantic→structural fallback for decisions and change_history legs.

Covers the bridge added to close the 'unscoped goal=answer returns empty' gap
observed on 2026-04-22 (docs/context-graph/implementation-next-steps.md #2).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from adapters.outbound.graphiti.query_helpers import (
    get_change_history,
    get_decisions,
)

pytestmark = pytest.mark.unit


class _StubStructural:
    """Records get_decisions / get_change_history calls; returns canned rows per call."""

    def __init__(
        self,
        *,
        decision_rows_by_call: list[list[dict[str, Any]]] | None = None,
        change_rows_by_call: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        self._decision_rows_by_call = list(decision_rows_by_call or [])
        self._change_rows_by_call = list(change_rows_by_call or [])
        self.decision_calls: list[dict[str, Any]] = []
        self.change_calls: list[dict[str, Any]] = []

    def get_decisions(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.decision_calls.append(kwargs)
        if not self._decision_rows_by_call:
            return []
        return list(self._decision_rows_by_call.pop(0))

    def get_change_history(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.change_calls.append(kwargs)
        if not self._change_rows_by_call:
            return []
        return list(self._change_rows_by_call.pop(0))


class _StubEpisodic:
    """Minimal episodic search port: returns edge objects with source/target uuids."""

    enabled = True

    def __init__(self, seeds: list[tuple[str, str]]) -> None:
        self._edges = [
            SimpleNamespace(
                uuid=f"edge-{i}",
                name="RELATES_TO",
                summary=None,
                fact=None,
                source_node_uuid=src,
                target_node_uuid=tgt,
                attributes={"_context_similarity_score": 1.0 - i * 0.1},
            )
            for i, (src, tgt) in enumerate(seeds)
        ]
        self.calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
        return list(self._edges)


# --- get_decisions ---------------------------------------------------------------


def test_scoped_request_does_not_trigger_fallback() -> None:
    structural = _StubStructural(
        decision_rows_by_call=[
            [{"decision_made": "X", "rationale": "y", "pr_number": 1}]
        ]
    )
    episodic = _StubEpisodic([("s1", "t1")])

    rows = get_decisions(
        structural,
        "pot-1",
        file_path="app/x.py",
        episodic=episodic,
        query="why did we choose Neo4j",
    )

    assert len(rows) == 1
    assert episodic.calls == []
    assert len(structural.decision_calls) == 1
    assert structural.decision_calls[0]["file_path"] == "app/x.py"


def test_unscoped_empty_structural_triggers_semantic_fallback() -> None:
    structural = _StubStructural(
        decision_rows_by_call=[
            [],  # first call: empty
            [{"decision_made": "Adopt Neo4j", "rationale": "temporal edges", "pr_number": 7}],
        ]
    )
    episodic = _StubEpisodic([("seed-a", "seed-b"), ("seed-c", "seed-a")])

    rows = get_decisions(
        structural,
        "pot-1",
        episodic=episodic,
        query="why did we choose Neo4j over Postgres?",
    )

    assert len(rows) == 1
    assert rows[0]["source_method"] == "semantic_fallback"
    assert set(rows[0]["seed_node_uuids"]) == {"seed-a", "seed-b", "seed-c"}
    # Two structural calls: the primary (no node_uuids) + seeded (with node_uuids).
    assert len(structural.decision_calls) == 2
    assert structural.decision_calls[0].get("node_uuids") is None
    assert structural.decision_calls[1]["node_uuids"] == ["seed-a", "seed-b", "seed-c"]
    assert len(episodic.calls) == 1


def test_fallback_skipped_without_query() -> None:
    structural = _StubStructural(decision_rows_by_call=[[]])
    episodic = _StubEpisodic([("seed-a", "seed-b")])

    rows = get_decisions(structural, "pot-1", episodic=episodic, query=None)

    assert rows == []
    assert episodic.calls == []
    assert len(structural.decision_calls) == 1


def test_fallback_skipped_without_episodic() -> None:
    structural = _StubStructural(decision_rows_by_call=[[]])

    rows = get_decisions(structural, "pot-1", query="anything")

    assert rows == []
    assert len(structural.decision_calls) == 1


def test_fallback_skipped_when_no_semantic_hits() -> None:
    structural = _StubStructural(decision_rows_by_call=[[]])
    episodic = _StubEpisodic([])  # no hits

    rows = get_decisions(structural, "pot-1", episodic=episodic, query="something")

    assert rows == []
    # Primary called once; no seeded retry because seeds were empty.
    assert len(structural.decision_calls) == 1
    assert len(episodic.calls) == 1


def test_seeded_retry_empty_returns_empty_not_marked() -> None:
    structural = _StubStructural(decision_rows_by_call=[[], []])
    episodic = _StubEpisodic([("seed-a", "seed-b")])

    rows = get_decisions(structural, "pot-1", episodic=episodic, query="x")

    assert rows == []
    assert len(structural.decision_calls) == 2


# --- get_change_history ----------------------------------------------------------


def test_change_history_unscoped_triggers_fallback() -> None:
    structural = _StubStructural(
        change_rows_by_call=[
            [],
            [{"pr_number": 42, "title": "Switch to Hatchet", "decisions": []}],
        ]
    )
    episodic = _StubEpisodic([("prA", "featB")])

    rows = get_change_history(
        structural,
        "pot-1",
        episodic=episodic,
        query="what's the latest background worker work?",
    )

    assert len(rows) == 1
    assert rows[0]["source_method"] == "semantic_fallback"
    assert structural.change_calls[0].get("node_uuids") is None
    assert structural.change_calls[1]["node_uuids"] == ["prA", "featB"]


def test_change_history_scoped_bypasses_fallback() -> None:
    structural = _StubStructural(change_rows_by_call=[[]])
    episodic = _StubEpisodic([("x", "y")])

    rows = get_change_history(
        structural,
        "pot-1",
        pr_number=99,
        episodic=episodic,
        query="anything",
    )

    assert rows == []
    assert episodic.calls == []
    assert len(structural.change_calls) == 1

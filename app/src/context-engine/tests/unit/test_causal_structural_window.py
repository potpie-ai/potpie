"""Temporal bounds for ``walk_causal_chain_backward`` (04-causal-multihop)."""

from __future__ import annotations

from datetime import datetime, timezone

from adapters.outbound.neo4j.structural import Neo4jStructuralAdapter


def test_as_of_window_bounds_invalid_returns_none() -> None:
    lo, hi = Neo4jStructuralAdapter._as_of_window_bounds(None, 180)
    assert lo is None and hi is None
    lo, hi = Neo4jStructuralAdapter._as_of_window_bounds("not-a-date", 30)
    assert lo is None and hi is None


def test_as_of_window_bounds_centered_on_as_of() -> None:
    lo, hi = Neo4jStructuralAdapter._as_of_window_bounds("2025-06-15T12:00:00+00:00", 10)
    assert lo is not None and hi is not None
    assert (hi - lo).days == 20


def test_causal_step_in_window_filters_by_edge_time() -> None:
    mid = datetime(2025, 3, 1, tzinfo=timezone.utc)
    lo = datetime(2025, 1, 1, tzinfo=timezone.utc)
    hi = datetime(2025, 12, 31, tzinfo=timezone.utc)
    row = {"valid_at": mid, "uuid": "u1"}
    assert Neo4jStructuralAdapter._causal_step_in_as_of_window(
        row, window_lo=lo, window_hi=hi
    )
    assert not Neo4jStructuralAdapter._causal_step_in_as_of_window(
        row,
        window_lo=datetime(2025, 6, 1, tzinfo=timezone.utc),
        window_hi=datetime(2025, 8, 1, tzinfo=timezone.utc),
    )


def test_causal_step_undated_allowed_when_window_set() -> None:
    row = {"valid_at": None, "pred_valid_at": None, "uuid": "u1"}
    lo = datetime(2025, 1, 1, tzinfo=timezone.utc)
    hi = datetime(2025, 12, 31, tzinfo=timezone.utc)
    assert Neo4jStructuralAdapter._causal_step_in_as_of_window(
        row, window_lo=lo, window_hi=hi
    )

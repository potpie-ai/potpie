"""Tests for the ContextGraphQuery planner and multi-leg executor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from adapters.outbound.graphiti.context_graph import GraphitiContextGraphAdapter
from application.services.graph_query_planner import GraphQueryPlanner
from domain.graph_query import (
    ContextGraphGoal,
    ContextGraphQuery,
    ContextGraphScope,
    ContextGraphStrategy,
)
from domain.graph_query_plan import LegStrategy, MergePolicy


def _planner() -> GraphQueryPlanner:
    return GraphQueryPlanner()


def test_planner_maps_single_family_include_to_single_leg() -> None:
    plan = _planner().plan(
        ContextGraphQuery(
            pot_id="p1",
            query="auth",
            goal=ContextGraphGoal.RETRIEVE,
            strategy=ContextGraphStrategy.SEMANTIC,
            include=["semantic_search"],
        )
    )
    assert plan.merge_policy == MergePolicy.SINGLE
    assert [l.family for l in plan.legs] == ["semantic_search"]
    assert plan.legs[0].strategy == LegStrategy.SEMANTIC


def test_planner_semantic_search_without_query_falls_back() -> None:
    plan = _planner().plan(
        ContextGraphQuery(
            pot_id="p1",
            goal=ContextGraphGoal.RETRIEVE,
            strategy=ContextGraphStrategy.SEMANTIC,
            include=["semantic_search"],
        )
    )
    assert plan.legs == ()
    reasons = {fb["reason"] for fb in plan.planner_fallbacks}
    assert "missing_query" in reasons
    assert "no_matching_family" in reasons


def test_planner_owners_without_file_path_falls_back() -> None:
    plan = _planner().plan(
        ContextGraphQuery(
            pot_id="p1",
            goal=ContextGraphGoal.AGGREGATE,
            include=["owners"],
        )
    )
    assert plan.legs == ()
    assert any(
        fb["family"] == "owners" and fb["reason"] == "missing_scope"
        for fb in plan.planner_fallbacks
    )


def test_planner_unknown_include_token_becomes_fallback() -> None:
    plan = _planner().plan(
        ContextGraphQuery(
            pot_id="p1",
            goal=ContextGraphGoal.TIMELINE,
            include=["change_history", "bogus_family"],
        )
    )
    assert [l.family for l in plan.legs] == ["change_history"]
    assert any(
        fb["family"] == "bogus_family" and fb["reason"] == "unsupported_include"
        for fb in plan.planner_fallbacks
    )


def test_planner_bulk_multi_family_produces_multi_merge() -> None:
    plan = _planner().plan(
        ContextGraphQuery(
            pot_id="p1",
            query="auth",
            goal=ContextGraphGoal.RETRIEVE,
            strategy=ContextGraphStrategy.HYBRID,
            include=["semantic_search", "change_history", "decisions"],
            scope=ContextGraphScope(file_path="app/auth.py"),
        )
    )
    assert plan.merge_policy == MergePolicy.MULTI
    assert {l.family for l in plan.legs} == {
        "semantic_search",
        "change_history",
        "decisions",
    }


def test_adapter_multi_family_execution_merges_results() -> None:
    episodic = MagicMock()
    episodic.enabled = True
    structural = MagicMock()
    adapter = GraphitiContextGraphAdapter(episodic=episodic, structural=structural)

    with patch(
        "adapters.outbound.graphiti.context_graph.search_pot_context",
        return_value=[{"uuid": "s1"}],
    ), patch(
        "adapters.outbound.graphiti.context_graph.get_change_history",
        return_value=[{"pr_number": 9}],
    ), patch(
        "adapters.outbound.graphiti.context_graph.get_decisions",
        return_value=[{"id": "d1"}],
    ):
        out = adapter.query(
            ContextGraphQuery(
                pot_id="p1",
                query="auth",
                goal=ContextGraphGoal.RETRIEVE,
                strategy=ContextGraphStrategy.HYBRID,
                include=["semantic_search", "change_history", "decisions"],
            )
        )

    assert out.kind == "multi"
    assert out.error is None
    assert set(out.result.keys()) == {"semantic_search", "change_history", "decisions"}
    legs = out.meta["legs"]
    assert {l["family"] for l in legs} == {
        "semantic_search",
        "change_history",
        "decisions",
    }
    assert out.meta.get("merge") == "multi"


def test_adapter_executor_error_becomes_fallback_not_raise() -> None:
    episodic = MagicMock()
    episodic.enabled = True
    structural = MagicMock()
    adapter = GraphitiContextGraphAdapter(episodic=episodic, structural=structural)

    def boom(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("kaboom")

    with patch(
        "adapters.outbound.graphiti.context_graph.search_pot_context",
        side_effect=boom,
    ), patch(
        "adapters.outbound.graphiti.context_graph.get_decisions",
        return_value=[{"id": "d1"}],
    ):
        out = adapter.query(
            ContextGraphQuery(
                pot_id="p1",
                query="auth",
                strategy=ContextGraphStrategy.HYBRID,
                include=["semantic_search", "decisions"],
            )
        )

    assert out.kind == "multi"
    assert "decisions" in out.result
    assert "semantic_search" not in out.result
    fallbacks = out.meta.get("fallbacks") or []
    assert any(
        fb["family"] == "semantic_search" and fb["reason"] == "executor_error"
        for fb in fallbacks
    )


def test_adapter_unsupported_request_returns_structured_error() -> None:
    adapter = GraphitiContextGraphAdapter(
        episodic=MagicMock(),
        structural=MagicMock(),
    )
    out = adapter.query(
        ContextGraphQuery(
            pot_id="p1",
            goal=ContextGraphGoal.RETRIEVE,
            strategy=ContextGraphStrategy.EXACT,
            include=["nonsense"],
        )
    )
    assert out.error == "unsupported_context_graph_query"
    assert out.meta.get("fallbacks")


def test_adapter_pr_diff_leg_is_flagged_compat() -> None:
    adapter = GraphitiContextGraphAdapter(
        episodic=MagicMock(),
        structural=MagicMock(),
    )
    with patch(
        "adapters.outbound.graphiti.context_graph.get_pr_diff",
        return_value=[{"file_path": "a.py"}],
    ):
        out = adapter.query(
            ContextGraphQuery(
                pot_id="p1",
                include=["pr_diff"],
                scope=ContextGraphScope(pr_number=5),
            )
        )
    assert out.kind == "pr_diff"
    assert out.meta.get("compat") is True
    assert out.meta["legs"][0]["compat"] is True

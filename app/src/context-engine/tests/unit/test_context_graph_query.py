from __future__ import annotations

from unittest.mock import MagicMock, patch

from adapters.outbound.graphiti.context_graph import GraphitiContextGraphAdapter
from bootstrap.container import ContextEngineContainer
from domain.graph_query import (
    ContextGraphGoal,
    ContextGraphQuery,
    ContextGraphScope,
    ContextGraphStrategy,
)


def test_context_graph_adapter_semantic_query_delegates_to_existing_search() -> None:
    episodic = MagicMock()
    episodic.enabled = True
    structural = MagicMock()
    adapter = GraphitiContextGraphAdapter(episodic=episodic, structural=structural)

    with patch(
        "adapters.outbound.graphiti.context_graph.search_pot_context",
        return_value=[{"uuid": "u1"}],
    ) as search:
        out = adapter.query(
            ContextGraphQuery(
                pot_id="p1",
                query="auth",
                goal=ContextGraphGoal.RETRIEVE,
                strategy=ContextGraphStrategy.SEMANTIC,
                include=["semantic_search"],
                node_labels=["Decision"],
            )
        )

    search.assert_called_once()
    assert out.kind == "semantic_search"
    assert out.result == [{"uuid": "u1"}]


def test_context_graph_adapter_timeline_query_delegates_change_history() -> None:
    episodic = MagicMock()
    structural = MagicMock()
    adapter = GraphitiContextGraphAdapter(episodic=episodic, structural=structural)

    with patch(
        "adapters.outbound.graphiti.context_graph.get_change_history",
        return_value=[{"pr_number": 1}],
    ) as history:
        out = adapter.query(
            ContextGraphQuery(
                pot_id="p1",
                goal=ContextGraphGoal.TIMELINE,
                strategy=ContextGraphStrategy.TRAVERSAL,
                scope=ContextGraphScope(file_path="app.py", function_name="handle"),
            )
        )

    history.assert_called_once()
    assert out.kind == "change_history"
    assert out.result == [{"pr_number": 1}]


def test_container_can_hold_unified_context_graph() -> None:
    graph = MagicMock()
    c = ContextEngineContainer(
        settings=MagicMock(),
        episodic=MagicMock(),
        structural=MagicMock(),
        pots=MagicMock(),
        source_for_repo=lambda _repo: MagicMock(),
        context_graph=graph,
    )

    assert c.context_graph is graph

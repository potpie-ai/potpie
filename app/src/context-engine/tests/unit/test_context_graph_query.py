from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from adapters.outbound.graphiti.context_graph import GraphitiContextGraphAdapter
from bootstrap.container import ContextEngineContainer
from domain.graph_query import (
    ContextGraphGoal,
    ContextGraphQuery,
    ContextGraphScope,
    ContextGraphStrategy,
    preset_change_history,
    preset_file_owners,
    preset_pr_review_context,
    preset_semantic_search,
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


def test_context_graph_presets_compile_legacy_query_shapes() -> None:
    search = preset_semantic_search(
        pot_id="p1",
        query="auth",
        repo_name="o/r",
        node_labels=["Decision"],
        source_description="github/pr/1",
        limit=7,
    )
    assert search.goal == ContextGraphGoal.RETRIEVE
    assert search.strategy == ContextGraphStrategy.SEMANTIC
    assert search.include == ["semantic_search"]
    assert search.scope.repo_name == "o/r"
    assert search.node_labels == ["Decision"]
    assert search.source_descriptions == ["github/pr/1"]
    assert search.limit == 7

    history = preset_change_history(
        pot_id="p1",
        file_path="app.py",
        function_name="handle",
        pr_number=4,
    )
    assert history.goal == ContextGraphGoal.TIMELINE
    assert history.include == ["change_history"]
    assert history.scope.file_path == "app.py"
    assert history.scope.function_name == "handle"
    assert history.scope.pr_number == 4

    owners = preset_file_owners(pot_id="p1", file_path="app.py")
    assert owners.goal == ContextGraphGoal.AGGREGATE
    assert owners.include == ["owners"]
    assert owners.scope.file_path == "app.py"

    review = preset_pr_review_context(pot_id="p1", pr_number=8, repo_name="o/r")
    assert review.include == ["pr_review_context"]
    assert review.scope.pr_number == 8
    assert review.scope.repo_name == "o/r"


@pytest.mark.asyncio
async def test_context_graph_adapter_sync_answer_query_rejects_running_loop() -> None:
    adapter = GraphitiContextGraphAdapter(
        episodic=MagicMock(),
        structural=MagicMock(),
        resolution_service=MagicMock(),
    )

    with pytest.raises(RuntimeError, match="use query_async"):
        adapter.query(
            ContextGraphQuery(
                pot_id="p1",
                query="summarize auth",
                goal=ContextGraphGoal.ANSWER,
            )
        )


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

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from adapters.outbound.graph.context_graph_service import ContextGraphService
from adapters.outbound.graph.in_memory_reader import InMemoryClaimQueryStore
from application.services.read_orchestrator import ReadOrchestrator
from bootstrap.container import ContextEngineContainer
from domain.graph_query import (
    ContextGraphGoal,
    ContextGraphQuery,
    ContextGraphStrategy,
    preset_context_search,
    preset_reader_lookup,
)


def _adapter(**kw) -> ContextGraphService:
    episodic = kw.pop("episodic", MagicMock())
    if not hasattr(episodic, "enabled") or episodic.enabled is None:
        episodic.enabled = True
    kw.setdefault(
        "orchestrator", ReadOrchestrator(claim_query=InMemoryClaimQueryStore())
    )
    return ContextGraphService(graph_writer=episodic, **kw)


def test_context_graph_presets_compile_reader_query_shapes() -> None:
    # Generic search: routes by intent, so it carries no fixed include.
    search = preset_context_search(
        pot_id="p1",
        query="auth",
        repo_name="o/r",
        node_labels=["Decision"],
        limit=7,
    )
    assert search.goal == ContextGraphGoal.RETRIEVE
    assert search.strategy == ContextGraphStrategy.AUTO
    assert search.include == []
    assert search.scope.repo_name == "o/r"
    assert search.node_labels == ["Decision"]
    assert search.limit == 7

    # Targeted reader lookup: pins a single backed include + scope.
    timeline = preset_reader_lookup(
        pot_id="p1",
        include="timeline",
        file_path="app.py",
        function_name="handle",
        pr_number=4,
    )
    assert timeline.goal == ContextGraphGoal.RETRIEVE
    assert timeline.include == ["timeline"]
    assert timeline.scope.file_path == "app.py"
    assert timeline.scope.function_name == "handle"
    assert timeline.scope.pr_number == 4

    prefs = preset_reader_lookup(
        pot_id="p1", include="coding_preferences", file_path="app.py"
    )
    assert prefs.include == ["coding_preferences"]
    assert prefs.scope.file_path == "app.py"


@pytest.mark.asyncio
async def test_context_graph_adapter_sync_answer_query_rejects_running_loop() -> None:
    adapter = _adapter()

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
        graph_writer=MagicMock(),
        pots=MagicMock(),
        context_graph=graph,
    )

    assert c.context_graph is graph

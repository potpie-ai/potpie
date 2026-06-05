from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from adapters.outbound.graph.context_graph_service import ContextGraphService
from bootstrap.ingestion_server import IngestionServerContainer
from domain.graph_query import (
    ContextGraphGoal,
    ContextGraphQuery,
    ContextGraphStrategy,
    preset_context_search,
    preset_reader_lookup,
)


def _adapter(**kw) -> ContextGraphService:
    backend = kw.pop("backend", None) or InMemoryGraphBackend()
    return ContextGraphService(backend=backend)


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


def test_context_graph_query_returns_envelope_for_any_goal() -> None:
    # One read contract: query() always routes through the orchestrator to a
    # ranked-evidence envelope. There is no agentic/answer path to special-case
    # or reject — the agent synthesises from the returned evidence.
    adapter = _adapter()

    result = adapter.query(
        ContextGraphQuery(
            pot_id="p1",
            query="summarize auth",
            goal=ContextGraphGoal.RETRIEVE,
        )
    )

    assert result.kind == "resolve"
    assert result.meta["path"] == "resolve"
    assert isinstance(result.result, dict) and "items" in result.result


@pytest.mark.asyncio
async def test_context_graph_query_async_returns_envelope() -> None:
    adapter = _adapter()

    result = await adapter.query_async(
        ContextGraphQuery(pot_id="p1", query="auth", goal=ContextGraphGoal.RETRIEVE)
    )

    assert result.kind == "resolve"
    assert isinstance(result.result, dict) and "items" in result.result


def test_container_can_hold_unified_context_graph() -> None:
    graph = MagicMock()
    c = IngestionServerContainer(
        settings=MagicMock(),
        graph_writer=MagicMock(),
        pots=MagicMock(),
        context_graph=graph,
    )

    assert c.context_graph is graph

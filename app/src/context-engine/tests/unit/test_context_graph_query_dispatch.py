"""CGT-8: read dispatch and async contract for ``GraphitiContextGraphAdapter``.

Pins the documented event-loop safety contract and agentic goal routing:
- ``query()`` must not run ``ANSWER`` / ``INVESTIGATE`` inside a live loop;
- ``query_async()`` is the async entry for those goals;
- non-agentic reads stay on the sync reader path;
- reconciliation ``context_search`` returns the normalized ``ContextGraphResult``
  envelope (``kind``, ``goal``, ``strategy``, ``result``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters.outbound.graphiti.context_graph import GraphitiContextGraphAdapter
from adapters.outbound.readers import SemanticSearchReader
from adapters.outbound.reconciliation.context_graph_tools import (
    ContextGraphReconciliationTools,
)
from application.services.context_reader_registry import ContextReaderRegistry
from domain.graph_query import (
    ContextGraphGoal,
    ContextGraphQuery,
    ContextGraphResult,
    ContextGraphScope,
    ContextGraphStrategy,
)
from domain.intelligence_models import (
    ContextResolutionRequest,
    CoverageReport,
    IntelligenceBundle,
    ResolutionMeta,
)
from domain.ports.query_agent import QueryAgentResult
from domain.context_events import ContextEvent
from domain.reconciliation import ReconciliationRequest

pytestmark = pytest.mark.unit

_AGENTIC_LOOP_MSG = "use query_async"


def _adapter(**kwargs: object) -> GraphitiContextGraphAdapter:
    episodic = kwargs.pop("episodic", MagicMock())
    if not hasattr(episodic, "enabled"):
        episodic.enabled = True
    structural = kwargs.pop("structural", MagicMock())
    readers = kwargs.pop("readers", ContextReaderRegistry())
    return GraphitiContextGraphAdapter(
        episodic=episodic,
        structural=structural,
        readers=readers,
        **kwargs,
    )


def _answer_query() -> ContextGraphQuery:
    return ContextGraphQuery(
        pot_id="pot-1",
        query="summarize auth",
        goal=ContextGraphGoal.ANSWER,
        strategy=ContextGraphStrategy.AUTO,
    )


def _investigate_query() -> ContextGraphQuery:
    return ContextGraphQuery(
        pot_id="pot-1",
        query="who owns auth?",
        goal=ContextGraphGoal.INVESTIGATE,
        strategy=ContextGraphStrategy.AUTO,
        scope=ContextGraphScope(repo_name="acme/widgets"),
    )


class TestSyncQueryEventLoopSafety:
    @pytest.mark.asyncio
    async def test_rejects_answer_while_event_loop_running(self) -> None:
        adapter = _adapter(resolution_service=MagicMock())
        with pytest.raises(RuntimeError, match=_AGENTIC_LOOP_MSG):
            adapter.query(_answer_query())

    @pytest.mark.asyncio
    async def test_rejects_investigate_while_event_loop_running(self) -> None:
        adapter = _adapter(resolution_service=MagicMock(), query_agent=MagicMock())
        with pytest.raises(RuntimeError, match=_AGENTIC_LOOP_MSG):
            adapter.query(_investigate_query())


class TestSyncQueryRouting:
    def test_answer_without_loop_runs_sync_bridge_to_async_path(self) -> None:
        """No running loop → ``query()`` may bridge via ``asyncio.run``."""
        bundle = IntelligenceBundle(
            request=ContextResolutionRequest(pot_id="pot-1", query="summarize auth"),
            coverage=CoverageReport(status="complete"),
            meta=ResolutionMeta(provider="test"),
        )

        class _ResolutionService:
            async def resolve(self, _request: object) -> IntelligenceBundle:
                return bundle

        out = _adapter(resolution_service=_ResolutionService()).query(_answer_query())

        assert out.kind == "resolve_context"
        assert out.meta.get("path") == "answer"

    def test_retrieve_goal_does_not_call_asyncio_run(self) -> None:
        episodic = MagicMock()
        episodic.enabled = True
        structural = MagicMock()
        readers = ContextReaderRegistry()
        readers.register(SemanticSearchReader(episodic=episodic, structural=structural))
        adapter = _adapter(
            episodic=episodic, structural=structural, readers=readers
        )

        with patch(
            "adapters.outbound.graphiti.context_graph.asyncio.run",
        ) as run, patch(
            "adapters.outbound.readers.semantic_search.search_pot_context",
            return_value=[{"uuid": "u1"}],
        ):
            out = adapter.query(
                ContextGraphQuery(
                    pot_id="pot-1",
                    query="auth",
                    goal=ContextGraphGoal.RETRIEVE,
                    strategy=ContextGraphStrategy.SEMANTIC,
                    include=["semantic_search"],
                )
            )

        run.assert_not_called()
        assert out.kind == "semantic_search"


class TestQueryAsyncAgenticDispatch:
    @pytest.mark.asyncio
    async def test_answer_dispatches_to_resolve_context_path(self) -> None:
        bundle = IntelligenceBundle(
            request=ContextResolutionRequest(pot_id="pot-1", query="why?"),
            coverage=CoverageReport(status="complete"),
            meta=ResolutionMeta(provider="test"),
        )

        class _ResolutionService:
            async def resolve(self, _request: object) -> IntelligenceBundle:
                return bundle

        adapter = _adapter(resolution_service=_ResolutionService())
        out = await adapter.query_async(_answer_query())

        assert out.kind == "resolve_context"
        assert out.goal == "answer"
        assert out.meta.get("path") == "answer"
        assert out.result is not None
        assert "answer" in out.result

    @pytest.mark.asyncio
    async def test_investigate_dispatches_to_query_agent_path(self) -> None:
        agent = MagicMock()
        agent.investigate = AsyncMock(
            return_value=QueryAgentResult(
                answer="Auth is owned by platform.",
                evidence=[{"tool": "context_search", "hit": 1}],
                source_refs=["gh:pr:1"],
                iterations=2,
                confidence=0.85,
            )
        )
        adapter = _adapter(
            resolution_service=MagicMock(),
            query_agent=agent,
        )
        out = await adapter.query_async(_investigate_query())

        assert out.kind == "query_agent"
        assert out.meta.get("path") == "investigate"
        assert out.result["answer"]["summary"] == "Auth is owned by platform."
        assert out.result["agent"]["iterations"] == 2
        agent.investigate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_investigate_without_agent_falls_back_to_answer_path(self) -> None:
        bundle = IntelligenceBundle(
            request=ContextResolutionRequest(pot_id="pot-1", query="who?"),
            coverage=CoverageReport(status="empty"),
            meta=ResolutionMeta(provider="test"),
        )

        class _ResolutionService:
            async def resolve(self, _request: object) -> IntelligenceBundle:
                return bundle

        adapter = _adapter(
            resolution_service=_ResolutionService(),
            query_agent=None,
        )
        out = await adapter.query_async(_investigate_query())

        assert out.kind == "resolve_context"
        assert out.meta.get("path") == "investigate_fallback"
        assert out.meta.get("fallback") == "query_agent_not_configured"

    @pytest.mark.asyncio
    async def test_retrieve_goal_uses_reader_registry_on_async_path(self) -> None:
        episodic = MagicMock()
        episodic.enabled = True
        structural = MagicMock()
        readers = ContextReaderRegistry()
        readers.register(SemanticSearchReader(episodic=episodic, structural=structural))
        adapter = _adapter(
            episodic=episodic, structural=structural, readers=readers
        )

        with patch(
            "adapters.outbound.readers.semantic_search.search_pot_context",
            return_value=[{"uuid": "async-u1"}],
        ):
            out = await adapter.query_async(
                ContextGraphQuery(
                    pot_id="pot-1",
                    query="auth",
                    goal=ContextGraphGoal.RETRIEVE,
                    strategy=ContextGraphStrategy.SEMANTIC,
                    include=["semantic_search"],
                )
            )

        assert out.kind == "semantic_search"
        assert out.result == [{"uuid": "async-u1"}]


class TestAgentReadToolEnvelope:
    """``context_search`` tool surface must return the normalized result dict."""

    def test_context_search_success_includes_stable_envelope_keys(self) -> None:
        graph = MagicMock()
        graph.enabled = True
        graph.query.return_value = ContextGraphResult(
            kind="semantic_search",
            goal="retrieve",
            strategy="semantic",
            result=[{"uuid": "u1", "name": "Auth decision"}],
            meta={"limit": 8},
        )
        tools = ContextGraphReconciliationTools(graph)
        event = ContextEvent(
            event_id="e1",
            pot_id="pot-1",
            provider="github",
            provider_host="github.com",
            repo_name="acme/widgets",
            source_system="github",
            event_type="pull_request",
            action="opened",
            source_id="pr-1",
            occurred_at=datetime(2026, 5, 22, tzinfo=timezone.utc),
        )
        req = ReconciliationRequest(
            event=event, pot_id="pot-1", repo_name="acme/widgets"
        )
        out = tools.execute_read_tool(
            req, "context_search", {"query": "auth module", "limit": 5}
        )

        assert set(out.keys()) >= {"kind", "goal", "strategy", "result"}
        assert out["kind"] == "semantic_search"
        assert out["result"] == [{"uuid": "u1", "name": "Auth decision"}]
        call = graph.query.call_args[0][0]
        assert call.pot_id == "pot-1"
        assert call.scope.repo_name == "acme/widgets"
        assert call.query == "auth module"

"""Behaviour contract for the ``apply_graph_mutations`` reconciliation tool.

These tests pin security/tenancy invariants on the tool boundary — not the
full pydantic-deep agent loop. A regression must change observable tool
responses or whether ``apply_plan_async`` is reached with the wrong pot.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from adapters.outbound.reconciliation.pydantic_deep_agent import (
    PydanticDeepReconciliationAgent,
    _BatchRunState,
)
from domain.context_events import ContextEvent
from domain.reconciliation import MutationSummary, ReconciliationResult

pytestmark = pytest.mark.unit


def _event(eid: str = "e1") -> ContextEvent:
    return ContextEvent(
        event_id=eid,
        source_system="github",
        event_type="pull_request",
        action="merged",
        pot_id="pot-1",
        provider="github",
        provider_host="github.com",
        repo_name="acme/widgets",
        source_id=f"src-{eid}",
        occurred_at=datetime(2026, 5, 6, tzinfo=timezone.utc),
    )


def _valid_plan() -> dict:
    return {
        "summary": "Record merge",
        "episodes": [
            {
                "name": "PR merged",
                "episode_body": "Merged PR #1",
                "source_description": "github",
            }
        ],
        "entity_upserts": [],
        "edge_upserts": [],
        "edge_deletes": [],
        "invalidations": [],
        "evidence": [],
    }


def _ok_result() -> ReconciliationResult:
    return ReconciliationResult(
        ok=True,
        episode_uuids=["ep-uuid-1"],
        mutation_summary=MutationSummary(
            episodes_written=1,
            entity_upserts_applied=2,
            edge_upserts_applied=3,
            edge_deletes_applied=0,
            invalidations_applied=0,
        ),
        downgrades=[],
    )


def _build_apply_tool(
    *,
    events: list[ContextEvent] | None = None,
    graph: MagicMock | None = None,
) -> tuple[object, MagicMock, _BatchRunState]:
    pytest.importorskip("pydantic_deep")
    evs = events or [_event()]
    by_id = {ev.event_id: ev for ev in evs}
    g = graph if graph is not None else MagicMock()
    if graph is None:
        g.apply_plan_async = AsyncMock(return_value=_ok_result())
    agent = PydanticDeepReconciliationAgent()
    agent.set_context_graph(g)
    state = _BatchRunState(
        pot_id="pot-1",
        repo_name="acme/widgets",
        events_by_id=by_id,
        context_graph=g,
    )
    tools = agent._build_mutation_tools(state)
    return tools[0].function, g, state


def _run(coro):
    return asyncio.run(coro)


class TestApplyGraphMutationsContract:
    def test_unknown_event_id_does_not_call_graph_apply(self) -> None:
        apply_fn, graph, _state = _build_apply_tool()
        out = _run(apply_fn(_valid_plan(), "missing-id", "s"))
        assert out["ok"] is False
        assert "unknown_event_id" in out["error"]
        assert "e1" in out["known_event_ids"]
        graph.apply_plan_async.assert_not_called()

    def test_invalid_plan_does_not_call_graph_apply(self) -> None:
        apply_fn, graph, _state = _build_apply_tool()
        out = _run(apply_fn({"not_a_plan": True}, "e1", "s"))
        assert out["ok"] is False
        assert out["error"].startswith("invalid_plan:")
        graph.apply_plan_async.assert_not_called()

    def test_plan_conversion_failed_does_not_call_graph_apply(self) -> None:
        apply_fn, graph, _state = _build_apply_tool()
        bad = _valid_plan()
        bad["invalidations"] = [{"reason": "orphan invalidation"}]
        out = _run(apply_fn(bad, "e1", "s"))
        assert out["ok"] is False
        assert out["error"].startswith("plan_conversion_failed:")
        graph.apply_plan_async.assert_not_called()

    def test_apply_failed_surfaces_graph_error(self) -> None:
        graph = MagicMock()
        graph.apply_plan_async = AsyncMock(side_effect=RuntimeError("neo4j down"))
        apply_fn, graph, _state = _build_apply_tool(graph=graph)
        out = _run(apply_fn(_valid_plan(), "e1", "s"))
        assert out["ok"] is False
        assert "apply_failed:" in out["error"]
        assert "neo4j down" in out["error"]
        graph.apply_plan_async.assert_awaited_once()

    def test_success_passes_expected_pot_and_provenance(self) -> None:
        apply_fn, graph, _state = _build_apply_tool()
        out = _run(apply_fn(_valid_plan(), "e1", "ignored summary"))
        assert out["ok"] is True
        assert out["mutation_counts"]["episodes_written"] == 1
        graph.apply_plan_async.assert_awaited_once()
        _plan, kwargs = graph.apply_plan_async.await_args
        assert kwargs["expected_pot_id"] == "pot-1"
        prov = kwargs["provenance_context"]
        assert prov.created_by_agent == "pydantic-deep"
        assert prov.source_ref == "src-e1"

    def test_apply_call_cap_blocks_runaway_mutations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CONTEXT_ENGINE_MAX_APPLY_CALLS_PER_BATCH", "1")
        apply_fn, graph, _state = _build_apply_tool()
        first = _run(apply_fn(_valid_plan(), "e1", "s"))
        second = _run(apply_fn(_valid_plan(), "e1", "s"))
        assert first["ok"] is True
        assert second == {
            "ok": False,
            "error": "apply_call_cap_exceeded",
            "detail": (
                "apply_graph_mutations was called more than 1 times in this "
                "batch; stop and call finish_batch."
            ),
        }
        graph.apply_plan_async.assert_awaited_once()

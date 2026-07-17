"""Tests for the PydanticDeepReconciliationAgent batch interface (no LLM call)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from types import ModuleType

import anyio
import pytest

from potpie_context_engine.adapters.outbound.reconciliation.pydantic_deep_agent import (
    PydanticDeepReconciliationAgent,
)
from potpie_context_engine.bootstrap import sentry_metrics_runtime
from potpie_context_core.domain.context_events import ContextEvent
from potpie_context_engine.domain.reconciliation_batch import BatchAgentContext

pytestmark = pytest.mark.unit


def _event(eid: str) -> ContextEvent:
    return ContextEvent(
        event_id=eid,
        source_system="test",
        event_type="x",
        action="y",
        pot_id="pot-1",
        provider="github",
        provider_host="github.com",
        repo_name="o/r",
        source_id=f"src-{eid}",
        occurred_at=datetime(2026, 5, 6, tzinfo=timezone.utc),
    )


def test_capability_metadata_marks_batch_tools_v1() -> None:
    pytest.importorskip("pydantic_deep")
    agent = PydanticDeepReconciliationAgent()
    meta = agent.capability_metadata()
    assert meta["agent"] == "pydantic-deep"
    assert meta["toolset_version"] == "batch-tools-v1"
    assert meta["has_context_graph"] is False


def test_run_batch_without_context_graph_returns_error() -> None:
    pytest.importorskip("pydantic_deep")
    agent = PydanticDeepReconciliationAgent()
    ctx = BatchAgentContext(
        batch_id="b1",
        pot_id="pot-1",
        repo_name="o/r",
        events=[_event("e1")],
    )
    out = agent.run_batch(ctx)
    assert out.ok is False
    assert out.error == "context_graph_unavailable"


def test_compose_instructions_includes_playbooks_for_batch_kinds() -> None:
    pytest.importorskip("pydantic_deep")
    agent = PydanticDeepReconciliationAgent()

    bootstrap = _event("e-boot")
    bootstrap.source_system = "github"
    bootstrap.event_type = "repository"
    bootstrap.action = "added"

    pr = _event("e-pr")
    pr.source_system = "github"
    pr.event_type = "pull_request"
    pr.action = "merged"

    ctx = BatchAgentContext(
        batch_id="b1",
        pot_id="pot-1",
        repo_name="o/r",
        events=[bootstrap, pr],
    )
    text = agent._compose_instructions(ctx)
    # Both kinds present.
    assert "github / repository / added" in text
    assert "github / pull_request / merged" in text
    # Tool budget line uses the higher of the two playbook budgets.
    assert "TOOL BUDGET" in text
    # Base instructions still present.
    assert "context-graph ingestion agent" in text


def test_compose_instructions_dedupes_repeated_event_kinds() -> None:
    pytest.importorskip("pydantic_deep")
    agent = PydanticDeepReconciliationAgent()

    e1 = _event("e1")
    e2 = _event("e2")
    for ev in (e1, e2):
        ev.source_system = "github"
        ev.event_type = "pull_request"
        ev.action = "merged"

    ctx = BatchAgentContext(
        batch_id="b1",
        pot_id="pot-1",
        repo_name="o/r",
        events=[e1, e2],
    )
    text = agent._compose_instructions(ctx)
    # The kind header appears once, not twice.
    assert text.count("github / pull_request / merged") == 1


def test_run_batch_short_circuits_for_empty_event_list() -> None:
    pytest.importorskip("pydantic_deep")
    agent = PydanticDeepReconciliationAgent()
    # Wire a fake graph so the early return is the empty-events guard, not the
    # context-graph check.

    class _Graph:
        def apply_plan(self, *_args, **_kwargs):
            raise AssertionError("must not be called for empty batch")

    agent.set_context_graph(_Graph())
    ctx = BatchAgentContext(
        batch_id="b1",
        pot_id="pot-1",
        repo_name="o/r",
        events=[],
    )
    out = agent.run_batch(ctx)
    assert out.ok is True
    assert out.completed_event_ids == []


def test_run_batch_timeout_mirrors_sentry_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts: list[tuple[str, int, dict[str, str]]] = []
    monkeypatch.setattr(
        sentry_metrics_runtime,
        "count",
        lambda name, value=1, *, attributes=None, unit=None: counts.append(
            (name, value, dict(attributes or {}))
        ),
    )
    monkeypatch.setenv("CONTEXT_ENGINE_AGENT_RUN_TIMEOUT_SECS", "0.001")

    class _SlowAgent:
        def tool_plain(self, **_kwargs):
            def _decorator(fn):
                return fn

            return _decorator

        async def run(self, *_args, **_kwargs):
            await anyio.sleep(1)

    fake_deep = ModuleType("pydantic_deep")
    fake_deep.CheckpointMiddleware = lambda **_kwargs: object()
    fake_deep.create_deep_agent = lambda **_kwargs: _SlowAgent()
    fake_deep.create_default_deps = lambda: object()
    monkeypatch.setitem(sys.modules, "pydantic_deep", fake_deep)
    monkeypatch.setitem(
        sys.modules, "pydantic_ai.usage", ModuleType("pydantic_ai.usage")
    )

    agent = PydanticDeepReconciliationAgent()
    agent.set_context_graph(object())
    monkeypatch.setattr(agent, "_build_read_tools", lambda _ctx: [])
    monkeypatch.setattr(agent, "_build_mutation_tools", lambda _state: [])
    monkeypatch.setattr(agent, "_build_control_tools", lambda _state: [])
    ctx = BatchAgentContext(
        batch_id="batch-1",
        pot_id="pot-1",
        repo_name="o/r",
        events=[_event("e1")],
    )

    out = agent.run_batch(ctx)

    assert out.ok is False
    assert ("ce.agent.timeout_total", 1, {"result": "timeout"}) in counts

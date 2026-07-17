"""Integration tests for ``PydanticDeepReconciliationAgent.run_batch``.

These exercise the *real* pydantic-deep / pydantic-ai stack with a
``TestModel`` (no LLM call). They exist to catch upstream API drift early —
unit tests with mocks miss problems like:

- ``create_deep_agent`` rejecting kwargs that were renamed in a new release
  (``include_web`` → ``web_search``/``web_fetch``,
   ``include_general_purpose_subagent`` → ``include_builtin_subagents``).
- pydantic-ai's tool-schema generator rejecting tool functions whose first
  parameter is not annotated ``RunContext[...]`` when registered via
  ``agent.tool(...)`` (the pydantic-deep ``tools=...`` re-registration path).

If any of those break again, these tests fail at construction time rather
than waiting for a real Celery batch to surface the error.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from potpie_context_engine.adapters.outbound.reconciliation.pydantic_deep_agent import (
    PydanticDeepReconciliationAgent,
)
from potpie_context_core.domain.context_events import ContextEvent
from potpie_context_engine.domain.reconciliation_batch import BatchAgentContext

pytestmark = pytest.mark.integration


def _ev(eid: str) -> ContextEvent:
    return ContextEvent(
        event_id=eid,
        source_system="github",
        event_type="repository",
        action="added",
        pot_id="pot-int-1",
        provider="github",
        provider_host="github.com",
        repo_name="acme/widgets",
        source_id=f"src-{eid}",
        occurred_at=datetime(2026, 5, 7, tzinfo=timezone.utc),
        payload={"owner": "acme", "repo": "widgets"},
    )


def _ctx() -> BatchAgentContext:
    return BatchAgentContext(
        batch_id="b-int-1",
        pot_id="pot-int-1",
        repo_name="acme/widgets",
        events=[_ev("e1")],
    )


def _agent_with_test_model(*, call_tools: "str | list[str]" = "all"):
    """Build the reconciliation agent wired to a pydantic-ai TestModel."""
    pytest.importorskip("pydantic_deep")
    from pydantic_ai.models.test import TestModel

    agent = PydanticDeepReconciliationAgent(model=TestModel(call_tools=call_tools))
    fake_graph = MagicMock()
    fake_graph.apply_plan.return_value = MagicMock(
        ok=True,
        error=None,
        mutation_id="mut-1",
        downgrades=[],
        mutation_summary=MagicMock(
            entity_upserts_applied=0,
            edge_upserts_applied=0,
            edge_deletes_applied=0,
            invalidations_applied=0,
        ),
    )
    agent.set_context_graph(fake_graph)
    return agent, fake_graph


def test_run_batch_constructs_agent_and_completes_with_test_model() -> None:
    """create_deep_agent kwargs + tool schema generation must succeed.

    With ``call_tools=[]`` TestModel returns text immediately without invoking
    any tool — but pydantic-ai still generates JSON schemas for every
    registered tool during construction. If any signature is incompatible,
    construction raises before this line returns ``ok=True``.
    """
    agent, _ = _agent_with_test_model(call_tools=[])

    out = agent.run_batch(_ctx())

    assert out.ok is True, f"expected ok run, got error={out.error}"


def test_run_batch_invokes_finish_batch_tool_via_test_model() -> None:
    """Verify the finish_batch tool is reachable end-to-end.

    ``call_tools=['finish_batch']`` makes TestModel call exactly that tool with
    auto-generated arguments. After the run we expect the agent to have
    completed successfully — the integration confirms our ``tool_plain``
    registration path actually wires the function into pydantic-ai's toolset.
    """
    agent, _ = _agent_with_test_model(call_tools=["finish_batch"])

    out = agent.run_batch(_ctx())

    assert out.ok is True
    # ``completed_event_ids`` is sourced from ``state.completed_event_ids``,
    # which is only populated by ``mark_events_processed`` (the singular
    # ``mark_event_processed`` delegates to it). ``finish_batch`` alone
    # leaves it empty — that's expected here. We just need the run to not
    # crash, which it would if tool registration was broken.
    assert out.completed_event_ids == []


def test_run_batch_returns_error_when_no_context_graph() -> None:
    """Sanity guard: without a context graph, run_batch short-circuits."""
    pytest.importorskip("pydantic_deep")
    from pydantic_ai.models.test import TestModel

    agent = PydanticDeepReconciliationAgent(model=TestModel(call_tools=[]))
    # Deliberately no set_context_graph().
    out = agent.run_batch(_ctx())
    assert out.ok is False
    assert out.error == "context_graph_unavailable"


def test_run_batch_short_circuits_for_empty_event_list() -> None:
    """Empty batch should never construct the agent — fast no-op path."""
    agent, _ = _agent_with_test_model(call_tools=[])
    ctx = BatchAgentContext(
        batch_id="b-empty",
        pot_id="pot-int-1",
        repo_name="acme/widgets",
        events=[],
    )
    out = agent.run_batch(ctx)
    assert out.ok is True
    assert out.completed_event_ids == []

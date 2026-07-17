"""Tests for ``NoOpReconciliationAgent`` (phase-2 smoke agent)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from potpie_context_engine.adapters.outbound.reconciliation.noop_agent import NoOpReconciliationAgent
from potpie_context_core.context_events import ContextEvent
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


def test_noop_agent_marks_all_events_completed() -> None:
    agent = NoOpReconciliationAgent()
    ctx = BatchAgentContext(
        batch_id="batch-1",
        pot_id="pot-1",
        repo_name="o/r",
        events=[_event("e1"), _event("e2")],
    )
    out = agent.run_batch(ctx)
    assert out.ok is True
    assert out.error is None
    assert sorted(out.completed_event_ids) == ["e1", "e2"]


def test_noop_agent_handles_empty_batch() -> None:
    agent = NoOpReconciliationAgent()
    ctx = BatchAgentContext(
        batch_id="batch-empty",
        pot_id="pot-1",
        repo_name="o/r",
        events=[],
    )
    out = agent.run_batch(ctx)
    assert out.ok is True
    assert out.completed_event_ids == []


def test_noop_agent_capability_metadata() -> None:
    meta = NoOpReconciliationAgent().capability_metadata()
    assert meta["agent"] == "noop"

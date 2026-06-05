"""Tests for write methods on the unified context graph adapter."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from adapters.outbound.graph.cypher import upsert_entities_async
from adapters.outbound.graph.context_graph_service import ContextGraphService
from domain.context_events import EventRef
from domain.graph_mutations import EntityUpsert, ProvenanceRef
from domain.reconciliation import (
    MutationSummary,
    ReconciliationPlan,
    ReconciliationResult,
)


def test_context_graph_apply_plan_delegates_to_backend_mutation() -> None:
    expected = ReconciliationResult(
        ok=True,
        mutation_id="u1",
        mutation_summary=MutationSummary(),
        error=None,
    )
    backend = MagicMock()
    backend.mutation.apply_async = AsyncMock(return_value=expected)
    graph = ContextGraphService(backend=backend)

    plan = ReconciliationPlan(
        event_ref=EventRef(event_id="e1", source_system="t", pot_id="p1"),
        summary="s",
        entity_upserts=[],
        edge_upserts=[],
        edge_deletes=[],
        invalidations=[],
    )

    out = graph.apply_plan(plan, expected_pot_id="p1")

    backend.mutation.apply_async.assert_awaited_once()
    assert out.ok is True
    assert out.mutation_id == "u1"


def test_context_graph_reset_pot_delegates_to_backend() -> None:
    backend = MagicMock()
    backend.mutation.reset_pot.return_value = {"ok": True}
    graph = ContextGraphService(backend=backend)

    out = graph.reset_pot("pot1")

    assert out == {
        "pot_id": "pot1",
        "ok": True,
        "graph_writer": {"ok": True},
    }
    backend.mutation.reset_pot.assert_called_once_with("pot1")


def test_context_graph_reset_pot_stops_on_backend_failure() -> None:
    backend = MagicMock()
    backend.mutation.reset_pot.return_value = {"ok": False, "error": "bad"}
    graph = ContextGraphService(backend=backend)

    out = graph.reset_pot("pot1")

    assert out == {
        "pot_id": "pot1",
        "ok": False,
        "graph_writer": {"ok": False, "error": "bad"},
        "error": "bad",
    }


class _FakeResult:
    async def consume(self) -> None:
        return None


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return None

    async def run(self, query: str, **kwargs):
        self.calls.append((query, kwargs))
        return _FakeResult()


class _FakeDriver:
    def __init__(self) -> None:
        self.session_obj = _FakeSession()

    def session(self) -> _FakeSession:
        return self.session_obj


def test_canonical_entity_upsert_sets_non_null_string_defaults() -> None:
    driver = _FakeDriver()
    count = asyncio.run(
        upsert_entities_async(
            driver,
            "pot1",
            [
                EntityUpsert(
                    "component:cli",
                    ("Entity",),
                    {"name": 123, "summary": None},
                )
            ],
            ProvenanceRef(pot_id="pot1", source_event_id="event1"),
        )
    )

    assert count == 1
    props = driver.session_obj.calls[0][1]["props"]
    assert props["name"] == "123"
    assert props["summary"] == ""

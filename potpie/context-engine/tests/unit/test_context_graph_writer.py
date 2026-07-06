"""Tests for write methods on the unified context graph adapter."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from adapters.outbound.graph.apply_plan import apply_mutation_batch
from adapters.outbound.graph.context_graph_service import ContextGraphService
from adapters.outbound.graph.cypher import upsert_entities_async
from domain.context_events import EventRef
from domain.errors import CapabilityNotImplemented
from domain.graph_mutations import (
    EdgeUpsert,
    EntityUpsert,
    ProvenanceContext,
    ProvenanceRef,
)
from domain.reconciliation import (
    MutationBatch,
)


def test_context_graph_apply_plan_is_legacy_not_implemented() -> None:
    backend = MagicMock()
    graph = ContextGraphService(backend=backend)

    plan = MutationBatch(
        event_ref=EventRef(event_id="e1", source_system="t", pot_id="p1"),
        summary="s",
        entity_upserts=[],
        edge_upserts=[],
        edge_deletes=[],
        invalidations=[],
    )

    with pytest.raises(CapabilityNotImplemented) as exc:
        graph.apply_plan(plan, expected_pot_id="p1")
    assert exc.value.capability == "context_graph.apply_plan"
    backend.mutation.apply_async.assert_not_called()


def test_apply_mutation_batch_uses_provenance_context_without_event_ref() -> None:
    captured: dict[str, ProvenanceRef] = {}

    class _Writer:
        async def upsert_entities(
            self, pot_id: str, items: list[EntityUpsert], provenance: ProvenanceRef
        ) -> int:
            captured["entities"] = provenance
            return len(items)

        async def upsert_edges(
            self, pot_id: str, items: list[EdgeUpsert], provenance: ProvenanceRef
        ) -> int:
            captured["edges"] = provenance
            return len(items)

        async def delete_edges(
            self, pot_id: str, items: list[object], provenance: ProvenanceRef
        ) -> int:
            return len(items)

        async def invalidate(
            self, pot_id: str, items: list[object], provenance: ProvenanceRef
        ) -> int:
            return len(items)

    plan = MutationBatch(
        entity_upserts=[
            EntityUpsert("service:payments-api", ("Service",), {}),
            EntityUpsert("service:ledger-api", ("Service",), {}),
        ],
        edge_upserts=[
            EdgeUpsert("DEPENDS_ON", "service:payments-api", "service:ledger-api")
        ],
    )
    result = asyncio.run(
        apply_mutation_batch(
            _Writer(),
            plan,
            expected_pot_id="p1",
            provenance_context=ProvenanceContext(
                source_event_id="harness-run-1",
                source_system="harness",
                source_kind="graph-mutation",
                source_ref="repo:acme/api@abc",
                created_by_agent="codex",
            ),
        )
    )

    assert result.ok is True
    prov = captured["edges"]
    assert prov.source_event_id == "harness-run-1"
    assert prov.source_system == "harness"
    assert prov.source_kind == "graph-mutation"
    assert prov.source_ref == "repo:acme/api@abc"
    assert prov.created_by_agent == "codex"


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
    kwargs = driver.session_obj.calls[0][1]
    assert kwargs["a_name"] == "123"
    assert kwargs["a_summary"] == "123"
    assert kwargs["a_description"] == "123"


def test_canonical_entity_upsert_derives_compact_summary_from_description() -> None:
    driver = _FakeDriver()
    count = asyncio.run(
        upsert_entities_async(
            driver,
            "pot1",
            [
                EntityUpsert(
                    "service:payments-api",
                    ("Service",),
                    {
                        "name": "payments-api",
                        "description": "  Payments API handles refunds, settlement, and ledger posting.  ",
                    },
                )
            ],
            ProvenanceRef(pot_id="pot1", source_event_id="event1"),
        )
    )

    assert count == 1
    kwargs = driver.session_obj.calls[0][1]
    assert (
        kwargs["a_summary"]
        == "Payments API handles refunds, settlement, and ledger posting."
    )
    assert (
        kwargs["a_description"]
        == "Payments API handles refunds, settlement, and ledger posting."
    )


def test_bare_entity_reference_never_carries_key_derived_display_fields() -> None:
    """A bare re-reference (key + type only) must not overwrite authored
    summaries: the writer sends empty authored params and lets the CASE
    expressions keep the stored value (filling the key only on new nodes)."""
    driver = _FakeDriver()
    asyncio.run(
        upsert_entities_async(
            driver,
            "pot1",
            [EntityUpsert("repo:github.com/acme/shop", ("Repository",), {})],
            ProvenanceRef(pot_id="pot1", source_event_id="event1"),
        )
    )

    query, kwargs = driver.session_obj.calls[0]
    assert kwargs["a_name"] == ""
    assert kwargs["a_summary"] == ""
    assert kwargs["a_description"] == ""
    for field in ("name", "summary", "description"):
        assert field not in kwargs["props"]
    # The query must only fill display fields when the node has none.
    assert "CASE WHEN $a_summary <> ''" in query
    assert "coalesce(e.summary, '') = ''" in query

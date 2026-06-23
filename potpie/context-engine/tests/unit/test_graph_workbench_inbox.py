from __future__ import annotations

import json

import pytest

from context_engine.adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from context_engine.adapters.outbound.graph.inbox_stores.local_json import LocalJsonGraphInboxStore
from context_engine.application.services.graph_workbench import GraphWorkbenchService
from context_engine.domain.graph_inbox import GraphInboxItem
from context_engine.domain.ports.claim_query import ClaimQueryFilter

pytestmark = pytest.mark.unit

POT = "p"


class _UnusedPlanStore:
    def save(self, _record) -> None:
        raise AssertionError("plan store should not be used by inbox tests")

    def get(self, *, pot_id: str, plan_id: str):
        raise AssertionError("plan store should not be used by inbox tests")

    def list(self, **_kwargs):
        raise AssertionError("plan store should not be used by inbox tests")


def _service(tmp_path) -> tuple[GraphWorkbenchService, InMemoryGraphBackend]:
    backend = InMemoryGraphBackend()
    return (
        GraphWorkbenchService(
            backend=backend,
            plan_store=_UnusedPlanStore(),
            inbox_store=LocalJsonGraphInboxStore(home=tmp_path),
        ),
        backend,
    )


def test_inbox_add_persists_pending_work_without_writing_graph_facts(tmp_path) -> None:
    workbench, backend = _service(tmp_path)

    result = workbench.inbox_add(
        pot_id=POT,
        summary="Possible graph update",
        evidence=("github:pr:955",),
        source_refs=("github:pr:955",),
        suspected_subgraphs=("debugging",),
        created_by={"surface": "cli", "actor": "codex"},
    )

    assert result.ok is True
    assert result.item is not None
    assert result.item.status == "pending"
    reloaded = LocalJsonGraphInboxStore(home=tmp_path).get(
        pot_id=POT,
        item_id=result.item.item_id,
    )
    assert reloaded is not None
    assert reloaded.summary == "Possible graph update"
    assert backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT)) == []


def test_inbox_claim_and_mark_applied_records_plan_and_mutation(tmp_path) -> None:
    workbench, _backend = _service(tmp_path)
    added = workbench.inbox_add(pot_id=POT, summary="Investigate prior bug")
    assert added.item is not None

    claimed = workbench.inbox_claim(
        pot_id=POT,
        item_id=added.item.item_id,
        claimed_by="user:alice",
    )
    applied = workbench.inbox_mark_applied(
        pot_id=POT,
        item_id=added.item.item_id,
        closed_by="user:alice",
        linked_plan_id="mutation-plan:test",
        linked_mutation_id="mutation-1",
    )

    assert claimed.ok is True
    assert claimed.item is not None
    assert claimed.item.status == "claimed"
    assert claimed.item.claimed_by == "user:alice"
    assert applied.ok is True
    assert applied.item is not None
    assert applied.item.status == "applied"
    assert applied.item.linked_plan_id == "mutation-plan:test"
    assert applied.item.linked_mutation_id == "mutation-1"


def test_inbox_rejected_items_are_terminal(tmp_path) -> None:
    workbench, _backend = _service(tmp_path)
    added = workbench.inbox_add(pot_id=POT, summary="Weak evidence")
    assert added.item is not None

    rejected = workbench.inbox_mark_rejected(
        pot_id=POT,
        item_id=added.item.item_id,
        closed_by="user:alice",
        rejection_reason="not enough evidence",
    )
    claimed_again = workbench.inbox_claim(
        pot_id=POT,
        item_id=added.item.item_id,
        claimed_by="user:bob",
    )

    assert rejected.ok is True
    assert rejected.item is not None
    assert rejected.item.status == "rejected"
    assert rejected.item.rejection_reason == "not enough evidence"
    assert claimed_again.ok is False
    assert "cannot be changed" in (claimed_again.detail or "")


def test_inbox_close_requires_plan_mutation_or_reason(tmp_path) -> None:
    workbench, _backend = _service(tmp_path)
    added = workbench.inbox_add(pot_id=POT, summary="Needs decision")
    assert added.item is not None

    with pytest.raises(ValueError, match="--plan, --mutation, or --reason"):
        workbench.inbox_close(
            pot_id=POT,
            item_id=added.item.item_id,
            closed_by="user:alice",
        )

    closed = workbench.inbox_close(
        pot_id=POT,
        item_id=added.item.item_id,
        closed_by="user:alice",
        rejection_reason="superseded",
    )
    assert closed.ok is True
    assert closed.item is not None
    assert closed.item.status == "closed"
    assert closed.item.rejection_reason == "superseded"


def test_inbox_list_filters_status_subgraph_and_source(tmp_path) -> None:
    workbench, _backend = _service(tmp_path)
    first = workbench.inbox_add(
        pot_id=POT,
        summary="Debugging item",
        evidence=("github:pr:955",),
        suspected_subgraphs=("debugging",),
    )
    second = workbench.inbox_add(
        pot_id=POT,
        summary="Feature item",
        evidence=("github:issue:12",),
        suspected_subgraphs=("features",),
    )
    assert first.item is not None
    assert second.item is not None
    workbench.inbox_mark_rejected(
        pot_id=POT,
        item_id=second.item.item_id,
        closed_by="user:alice",
        rejection_reason="duplicate",
    )

    pending_debugging = workbench.inbox_list(
        pot_id=POT,
        status=("pending",),
        suspected_subgraph="debugging",
        source_ref="github:pr:955",
    )
    rejected = workbench.inbox_list(pot_id=POT, status=("rejected",))

    assert [item.item_id for item in pending_debugging.items] == [first.item.item_id]
    assert [item.item_id for item in rejected.items] == [second.item.item_id]


def test_local_json_inbox_store_round_trips_items(tmp_path) -> None:
    store = LocalJsonGraphInboxStore(home=tmp_path)
    item = GraphInboxItem(
        item_id="graph-inbox:test",
        pot_id=POT,
        status="pending",
        summary="Round trip",
        evidence=("github:pr:955",),
        suspected_subgraphs=("debugging",),
        created_by={"surface": "cli"},
    )

    store.save(item)

    reloaded = LocalJsonGraphInboxStore(home=tmp_path).get(
        pot_id=POT,
        item_id=item.item_id,
    )
    assert reloaded == item
    raw = json.loads((tmp_path / "graph_inbox.json").read_text(encoding="utf-8"))
    assert item.item_id in raw["items"][POT]

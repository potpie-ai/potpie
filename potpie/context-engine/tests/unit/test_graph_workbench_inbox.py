from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import threading

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.inbox_stores.local_json import (
    LocalJsonGraphInboxStore,
)
from potpie_context_core.workbench_service import (
    GraphWorkbenchService,
)
from potpie_context_core.graph_inbox import GraphInboxItem
from potpie_context_core.ports.claim_query import ClaimQueryFilter

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


def test_second_agent_cannot_claim_an_item_another_agent_holds(tmp_path) -> None:
    # Two agents polling the same worklist both used to "claim" the same item
    # and do the work twice: the claim was a read-then-save with no exclusion.
    workbench, _backend = _service(tmp_path)
    added = workbench.inbox_add(pot_id=POT, summary="Investigate prior bug")
    assert added.item is not None

    first = workbench.inbox_claim(
        pot_id=POT,
        item_id=added.item.item_id,
        claimed_by="agent:alice",
    )
    second = workbench.inbox_claim(
        pot_id=POT,
        item_id=added.item.item_id,
        claimed_by="agent:bob",
    )

    assert first.ok is True
    assert first.item is not None
    assert first.item.claim_expires_at is not None
    assert second.ok is False
    assert "already claimed by 'agent:alice'" in (second.detail or "")
    assert second.recommended_next_action
    stored = LocalJsonGraphInboxStore(home=tmp_path).get(
        pot_id=POT,
        item_id=added.item.item_id,
    )
    assert stored is not None
    assert stored.claimed_by == "agent:alice"


def test_claim_holder_can_refresh_its_own_lease(tmp_path) -> None:
    workbench, _backend = _service(tmp_path)
    added = workbench.inbox_add(pot_id=POT, summary="Long running item")
    assert added.item is not None

    first = workbench.inbox_claim(
        pot_id=POT,
        item_id=added.item.item_id,
        claimed_by="agent:alice",
        lease_seconds=60,
    )
    renewed = workbench.inbox_claim(
        pot_id=POT,
        item_id=added.item.item_id,
        claimed_by="agent:alice",
        lease_seconds=600,
    )

    assert first.item is not None and renewed.item is not None
    assert renewed.ok is True
    assert renewed.item.claim_expires_at > first.item.claim_expires_at


def test_an_expired_claim_can_be_taken_over(tmp_path) -> None:
    # A worker that dies mid-item must not strand it forever.
    workbench, _backend = _service(tmp_path)
    store = LocalJsonGraphInboxStore(home=tmp_path)
    added = workbench.inbox_add(pot_id=POT, summary="Abandoned item")
    assert added.item is not None
    workbench.inbox_claim(
        pot_id=POT,
        item_id=added.item.item_id,
        claimed_by="agent:alice",
        lease_seconds=60,
    )
    held = store.get(pot_id=POT, item_id=added.item.item_id)
    assert held is not None
    store.save(
        replace(
            held,
            claim_expires_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
    )

    taken = workbench.inbox_claim(
        pot_id=POT,
        item_id=added.item.item_id,
        claimed_by="agent:bob",
    )

    assert taken.ok is True
    assert taken.item is not None
    assert taken.item.claimed_by == "agent:bob"


def test_concurrent_claims_produce_exactly_one_winner(tmp_path) -> None:
    workbench, _backend = _service(tmp_path)
    added = workbench.inbox_add(pot_id=POT, summary="Contended item")
    assert added.item is not None
    barrier = threading.Barrier(4)

    def claim(actor: str):
        barrier.wait(timeout=5)
        return workbench.inbox_claim(
            pot_id=POT,
            item_id=added.item.item_id,
            claimed_by=actor,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(claim, [f"agent:{i}" for i in range(4)]))

    winners = [result for result in results if result.ok]
    assert len(winners) == 1
    stored = LocalJsonGraphInboxStore(home=tmp_path).get(
        pot_id=POT,
        item_id=added.item.item_id,
    )
    assert stored is not None
    assert stored.claimed_by == winners[0].item.claimed_by

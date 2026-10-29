"""Record retries recover receipt authority after durable acknowledgement loss."""

from dataclasses import replace

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
    LocalJsonGraphPlanStore,
)
from potpie_context_engine.api import build_graph_runtime
from potpie_context_engine.core.ports.agent_context import RecordRequest


@pytest.mark.parametrize("persisted", [False, True], ids=["before-commit", "lost-ack"])
def test_record_retry_recovers_original_preparation_and_execution(
    tmp_path, monkeypatch, persisted
):
    graph_home = tmp_path / "graph"
    plan_path = tmp_path / "plans.json"
    backend = EmbeddedGraphBackend(home=graph_home)
    plans = LocalJsonGraphPlanStore(plan_path)
    runtime = build_graph_runtime(backend, plans)
    request = RecordRequest(
        pot_id="p",
        record_type="decision",
        summary="The ledger owns balances.",
        details={"rationale": "Keep balance changes atomic."},
        idempotency_key="record-recovery",
    )
    original_persist = EmbeddedGraphBackend._persist

    def fail_once(self, inner):
        monkeypatch.setattr(EmbeddedGraphBackend, "_persist", original_persist)
        if persisted:
            original_persist(self, inner)
        raise OSError("lost persistence acknowledgement")

    monkeypatch.setattr(EmbeddedGraphBackend, "_persist", fail_once)
    failed = runtime.record(request)
    assert not failed.accepted
    prepared = plans.get(pot_id="p", plan_id=failed.metadata["plan_id"])
    assert prepared.prepared is not None

    # Reopen both stores so a cached result cannot hide a durability failure.
    reopened = EmbeddedGraphBackend(home=graph_home)
    reopened_plans = LocalJsonGraphPlanStore(plan_path)
    recovered_runtime = build_graph_runtime(reopened, reopened_plans)
    before = reopened.mutation.lookup_execution(
        prepared.lowered_batch,
        expected_pot_id="p",
        mutation_id=prepared.mutation_id,
    )
    assert before.state == ("completed" if persisted else "absent")
    assert reopened.mutation.current_version("p") == int(persisted)

    recovered = recovered_runtime.record(request)
    assert recovered.accepted
    assert recovered.status == ("duplicate" if persisted else "recorded")
    assert (recovered.mutations_applied == 0) is persisted
    assert recovered.metadata["plan_id"] == prepared.plan_id
    assert recovered.metadata["mutation_id"] == prepared.mutation_id
    assert reopened.mutation.current_version("p") == 1

    receipt = reopened.mutation.lookup_execution(
        prepared.lowered_batch,
        expected_pot_id="p",
        mutation_id=prepared.mutation_id,
    )
    if persisted:
        assert receipt == before
    assert recovered_runtime.record(request).status == "duplicate"
    changed = recovered_runtime.record(replace(request, summary="Changed decision."))
    assert not changed.accepted
    assert "different content" in changed.detail
    assert reopened.mutation.current_version("p") == 1
    assert (
        reopened.mutation.lookup_execution(
            prepared.lowered_batch,
            expected_pot_id="p",
            mutation_id=prepared.mutation_id,
        )
        == receipt
    )

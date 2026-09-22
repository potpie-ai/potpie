"""Record retry identities use persisted plans, not fresh unguarded mutations."""

from dataclasses import replace

import pytest
from potpie_context_core.api import build_graph_runtime
from potpie_context_core.ports.agent_context import RecordRequest
from potpie_context_core.ports.graph_service import GraphReadRequest

from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
    LocalJsonGraphPlanStore,
)
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.testing import build_test_graph_runtime


def request():
    return RecordRequest(
        pot_id="capture",
        record_type="fix",
        summary="Retry pool exhaustion",
        details={
            "root_cause": "Leaked sockets",
            "fix_steps": ["Close sockets in finally"],
        },
        scope={"service": "checkout"},
        source_refs=("fixture:incident:7",),
        idempotency_key="capture:incident:7",
    )


def test_explicit_retry_keeps_original_receipt_and_rejects_changed_content():
    runtime = build_test_graph_runtime()
    first = runtime.record(request())
    assert first.accepted and first.mutations_applied > 0
    replay = runtime.record(request())
    assert replay.accepted and replay.status == "duplicate"
    assert replay.mutations_applied == 0
    assert replay.metadata["mutation_id"] == first.metadata["mutation_id"]
    assert replay.metadata["plan_id"] == first.metadata["plan_id"]
    changed = runtime.record(
        replace(request(), details={"root_cause": "Different cause"})
    )
    assert not changed.accepted
    assert "different content" in changed.detail
    # A refused key-reuse attempt must not poison the original retry identity.
    assert runtime.record(request()).accepted


def test_retry_survives_runtime_and_backend_reconstruction(tmp_path):
    def open_runtime():
        return build_graph_runtime(
            EmbeddedGraphBackend(home=tmp_path / "graph"),
            LocalJsonGraphPlanStore(home=tmp_path / "plans"),
        )

    first = open_runtime().record(request())
    assert first.accepted
    recovered = open_runtime()
    replay = recovered.record(request())
    assert replay.accepted and replay.status == "duplicate"
    assert replay.metadata["mutation_id"] == first.metadata["mutation_id"]
    assert not recovered.record(replace(request(), summary="Changed summary")).accepted
    read = recovered.read(
        GraphReadRequest(
            pot_id="capture",
            subgraph="debugging",
            view="prior_occurrences",
            scope={"service": "checkout"},
            detail="full",
        )
    )
    assert "Leaked sockets" in str(read.to_dict())
    assert "Changed summary" not in str(read.to_dict())


def test_unwired_service_refuses_to_claim_retry_safety():
    runtime = build_test_graph_runtime()
    graph = DefaultGraphService(backend=runtime.backend)
    with pytest.raises(ValueError, match="build_graph_runtime"):
        graph.record(request())


async def test_async_reservation_fallback_ignores_rejected_key_reuse():
    from potpie_context_core.workbench_service import _request_fingerprint

    from potpie_context_engine.testing import InMemoryGraphPlanStore, build_test_backend

    class LegacyPlanStore:
        """A supported store from before atomic reservation was added."""

        def __init__(self):
            self.store = InMemoryGraphPlanStore()

        def save(self, record):
            return self.store.save(record)

        def get(self, **kwargs):
            return self.store.get(**kwargs)

        def list(self, **kwargs):
            return self.store.list(**kwargs)

        def compare_and_set(self, **kwargs):
            return self.store.compare_and_set(**kwargs)

    runtime = build_graph_runtime(build_test_backend(), LegacyPlanStore())
    receipt = runtime.record(request())
    assert receipt.accepted
    assert not runtime.record(replace(request(), summary="Changed body")).accepted
    original = runtime.plan_store.get(
        pot_id="capture", plan_id=receipt.metadata["plan_id"]
    )
    reserved, created = await runtime.plan_store.reserve_idempotency_async(
        record=original,
        idempotency_key=request().idempotency_key,
        request_fingerprint=_request_fingerprint(original.original_payload),
    )
    assert not created
    assert reserved.plan_id == original.plan_id

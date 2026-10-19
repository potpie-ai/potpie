"""Public semantic writes require atomic adapters and retain request identity."""

import asyncio
from dataclasses import replace

import pytest

from potpie_context_engine.api import SemanticMutationRequest, build_graph_runtime
from potpie_context_engine.core.errors import CapabilityNotImplemented
from potpie_context_engine.testing import (
    InMemoryGraphBackend,
    InMemoryGraphInboxStore,
    InMemoryGraphPlanStore,
)
from tests.conformance.test_public_graph_runtime import (
    _AsyncOnlyBackend,
    _AsyncStore,
    _SyncOnlyBackend,
    _definition,
    _request,
)


class _UncoordinatedMutation:
    """A custom adapter offering writes without authoritative atomic execution."""

    def __init__(self, inner):
        self.inner = inner
        self.apply_calls = 0

    def apply(self, plan, **kwargs):
        self.apply_calls += 1
        return self.inner.apply(plan, **kwargs)

    def readiness(self, pot_id):
        return self.inner.readiness(pot_id)

    def invalidate(self, **kwargs):
        return self.inner.invalidate(**kwargs)

    def reset_pot(self, pot_id):
        return self.inner.reset_pot(pot_id)


class _UncoordinatedBackend(_SyncOnlyBackend):
    def _replace_inner(self, inner):
        super()._replace_inner(inner)
        self.mutation = _UncoordinatedMutation(inner.mutation)


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_public_semantic_write_refuses_adapter_without_atomic_execution(mode):
    backend = _UncoordinatedBackend()
    runtime = build_graph_runtime(
        backend, InMemoryGraphPlanStore(), InMemoryGraphInboxStore(), _definition()
    )
    with pytest.raises(CapabilityNotImplemented, match="atomic_mutation"):
        if mode == "sync":
            runtime.mutate(_request())
        else:
            asyncio.run(runtime.mutate_async(_request()))
    assert backend.mutation.apply_calls == 0
    assert backend.inner.mutation.current_version("pot:widgets") == 0
    assert backend.inner.store.rows == []


@pytest.mark.parametrize("mode", ["sync", "async", "async_only_ports"])
def test_public_direct_semantic_retry_retains_execution_and_rejects_changed_body(mode):
    async def journey():
        backend = (
            _AsyncOnlyBackend()
            if mode == "async_only_ports"
            else InMemoryGraphBackend()
        )
        plans = InMemoryGraphPlanStore()
        runtime = build_graph_runtime(
            backend,
            _AsyncStore(plans) if mode == "async_only_ports" else plans,
            InMemoryGraphInboxStore(),
            _definition(),
        )
        request = replace(_request(), idempotency_key="direct:stable")

        async def mutate(item):
            return (
                runtime.mutate(item)
                if mode == "sync"
                else await runtime.mutate_async(item)
            )

        first = await mutate(request)
        assert first.ok
        record = plans.get(pot_id=request.pot_id, plan_id=first.plan_id)
        assert record is not None and record.prepared is not None
        mutation = (
            backend.inner.mutation if mode == "async_only_ports" else backend.mutation
        )
        original = mutation.lookup_execution(
            record.lowered_batch,
            expected_pot_id=request.pot_id,
            mutation_id=first.mutation_id,
        )
        assert original.state == "completed"
        assert original.result.execution is not None
        assert original.result.execution.version_before == 0
        assert original.result.execution.version_after == 1

        replay = await mutate(request)
        assert replay.ok
        assert replay.plan_id == first.plan_id
        assert replay.mutation_id == first.mutation_id
        assert mutation.current_version(request.pot_id) == 1
        assert (
            mutation.lookup_execution(
                record.lowered_batch,
                expected_pot_id=request.pot_id,
                mutation_id=first.mutation_id,
            )
            == original
        )

        changed = SemanticMutationRequest.parse(
            {
                "pot_id": request.pot_id,
                "idempotency_key": request.idempotency_key,
                "operations": [
                    {
                        **request.operations[0].raw,
                        "description": "Changed semantic content.",
                    }
                ],
            }
        )
        rejected = await mutate(changed)
        assert not rejected.ok
        assert "idempotency" in (rejected.detail or "").lower()
        assert mutation.current_version(request.pot_id) == 1
        assert (
            sum(item.reserves_idempotency for item in plans.list(pot_id=request.pot_id))
            == 1
        )
        assert plans.get(pot_id=request.pot_id, plan_id=first.plan_id) == record
        assert (
            mutation.lookup_execution(
                record.lowered_batch,
                expected_pot_id=request.pot_id,
                mutation_id=first.mutation_id,
            )
            == original
        )

    asyncio.run(journey())

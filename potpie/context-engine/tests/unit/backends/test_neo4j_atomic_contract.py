from types import SimpleNamespace
from datetime import datetime, timezone

import pytest

from potpie_context_engine.core.workbench_service import GraphWorkbenchService
from potpie_context_engine.core.errors import CapabilityNotImplemented
from potpie_context_engine.adapters.outbound.graph.backends.neo4j_backend import (
    Neo4jGraphBackend,
)
from potpie_context_engine.testing import InMemoryGraphPlanStore
from potpie_context_engine.core.graph_mutations import ProvenanceContext
from potpie_context_engine.core.mutation_execution import PreparedMutation
from potpie_context_engine.core.reconciliation import MutationBatch


class LegacyWriter:
    enabled = True

    async def upsert_entities(self, pot_id, items, provenance):
        return len(items)

    async def upsert_edges(self, pot_id, items, provenance):
        return len(items)

    async def delete_edges(self, pot_id, items, provenance):
        return len(items)

    async def invalidate(self, pot_id, items, provenance):
        return len(items)


def test_injected_legacy_writer_reports_limit_and_workbench_rejects_it():
    backend = Neo4jGraphBackend(SimpleNamespace(), writer=LegacyWriter())
    readiness = backend.mutation.readiness("test")
    assert readiness.ready
    assert not readiness.capability_ready["atomic_mutation"]
    assert not readiness.capability_ready["durable_mutation_receipts"]
    workbench = GraphWorkbenchService(
        backend=backend, plan_store=InMemoryGraphPlanStore()
    )
    with pytest.raises(CapabilityNotImplemented, match="authoritative revisions"):
        workbench.propose(
            {
                "operations": [
                    {
                        "op": "upsert_entity",
                        "subject": {
                            "key": "service:api",
                            "type": "Service",
                            "properties": {"summary": "API service"},
                        },
                    }
                ]
            },
            pot_id="test",
        )


def test_disabled_native_writer_does_not_report_ready():
    settings = SimpleNamespace(is_enabled=lambda: False)
    backend = Neo4jGraphBackend(settings)
    result = backend.mutation.readiness("test")
    assert not result.ready
    assert not result.capability_ready["mutation"]
    assert not result.capability_ready["atomic_mutation"]


@pytest.mark.asyncio
async def test_legacy_writer_never_reports_missing_execution_as_absent():
    backend = Neo4jGraphBackend(SimpleNamespace(), writer=LegacyWriter())
    assert not backend.mutation.capabilities.atomic_writes
    assert not backend.mutation.capabilities.durable_receipts
    lookup = await backend.mutation.lookup_execution_async(
        MutationBatch(), expected_pot_id="test", mutation_id="unknown"
    )
    assert lookup.state == "unsupported"
    with pytest.raises(NotImplementedError, match="native revision"):
        await backend.mutation.current_version_async("test")
    with pytest.raises(NotImplementedError, match="native atomic"):
        await backend.mutation.compare_and_apply_async(
            MutationBatch(), expected_pot_id="test", expected_version=0
        )


@pytest.mark.asyncio
async def test_sync_methods_reject_a_running_loop_without_starting_work():
    backend = Neo4jGraphBackend(SimpleNamespace(), writer=LegacyWriter())
    calls = (
        lambda: backend.mutation.apply(MutationBatch(), expected_pot_id="test"),
        lambda: backend.mutation.current_version("test"),
        lambda: backend.mutation.compare_and_apply(
            MutationBatch(), expected_pot_id="test", expected_version=0
        ),
        lambda: backend.mutation.lookup_execution(
            MutationBatch(), expected_pot_id="test", mutation_id="unknown"
        ),
        lambda: backend.mutation.reset_pot("test"),
    )
    for call in calls:
        with pytest.raises(RuntimeError, match="corresponding async"):
            call()


@pytest.mark.asyncio
async def test_changed_preparation_is_rejected_before_opening_the_driver():
    from potpie_context_engine.adapters.outbound.graph.neo4j_atomic import apply_atomic
    from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
        MutationExecutionReuseError,
    )
    from potpie_context_engine.core.definition import DEFAULT_GRAPH_DEFINITION

    def unexpected_driver():
        pytest.fail("invalid preparation must fail before touching Neo4j")

    prepared = PreparedMutation(
        pot_id="test",
        mutation_id="reserved",
        request_fingerprint="request",
        batch_fingerprint="different-batch",
        expected_version=0,
        prepared_at=datetime.now(timezone.utc),
    )
    with pytest.raises(MutationExecutionReuseError, match="prepared"):
        await apply_atomic(
            SimpleNamespace(_new_driver=unexpected_driver),
            MutationBatch(),
            expected_pot_id="test",
            provenance_context=ProvenanceContext(
                mutation_id="reserved", prepared=prepared
            ),
            definition=DEFAULT_GRAPH_DEFINITION,
        )

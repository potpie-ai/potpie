from types import SimpleNamespace

from potpie_context_core.workbench_service import GraphWorkbenchService
from potpie_context_engine.adapters.outbound.graph.backends.neo4j_backend import Neo4jGraphBackend
from potpie_context_engine.testing import InMemoryGraphPlanStore


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


def test_injected_legacy_writer_uses_workbench_fallback_and_reports_limit():
    backend = Neo4jGraphBackend(SimpleNamespace(), writer=LegacyWriter())
    readiness = backend.mutation.readiness("test")
    assert readiness.ready
    assert not readiness.capability_ready["atomic_mutation"]
    assert not readiness.capability_ready["durable_mutation_receipts"]
    workbench = GraphWorkbenchService(backend=backend, plan_store=InMemoryGraphPlanStore())
    plan = workbench.propose({"operations": [{"op": "upsert_entity", "subject": {
        "key": "service:api", "type": "Service", "properties": {"summary": "API service"},
    }}]}, pot_id="test")
    assert plan.ok
    assert workbench.commit(plan.plan_id, pot_id="test").ok


def test_disabled_native_writer_does_not_report_ready():
    settings = SimpleNamespace(is_enabled=lambda: False)
    backend = Neo4jGraphBackend(settings)
    result = backend.mutation.readiness("test")
    assert not result.ready
    assert not result.capability_ready["mutation"]
    assert not result.capability_ready["atomic_mutation"]

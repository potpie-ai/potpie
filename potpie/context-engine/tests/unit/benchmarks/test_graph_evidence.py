"""Benchmark helpers consume the same evidence contract as product reads."""

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.benchmarks.core.graph_inspect import (
    _parse_envelope_claims,
    snapshot_graph,
)
from potpie_context_engine.benchmarks.core.query import resolve_context
from potpie_context_engine.benchmarks.core.scenario import QuerySpec
from potpie_context_engine.core.ports.agent_context import RecordRequest


def test_benchmark_reads_current_graph_service_evidence():
    graph = DefaultGraphService(backend=InMemoryGraphBackend())
    receipt = graph.record(
        RecordRequest(
            pot_id="bench",
            record_type="fix",
            summary="Close leaked connections",
            details={"fix_id": "fix:pool", "fix_steps": ["close connections"]},
            scope={"service": "api"},
            source_refs=("test:pool",),
        )
    )
    assert receipt.accepted
    response = resolve_context(
        graph, "bench", QuerySpec(intent="debugging", include=("prior_bugs",))
    )
    assert any(
        item["payload"].get("predicate") == "RESOLVED"
        for item in response["result"]["items"]
    )
    snapshot = snapshot_graph(graph, "bench")
    assert snapshot.edges_by_type("RESOLVED")
    assert snapshot.entities_by_label("Fix")


@pytest.mark.parametrize(
    "response",
    [{}, {"result": None}, {"result": {"items": [None, {}, {"payload": {}}]}}],
    ids=["missing", "null", "malformed-items"],
)
def test_benchmark_sample_ignores_absent_claims(response):
    snapshot = _parse_envelope_claims(response)
    assert snapshot.entities == []
    assert snapshot.edges == []


def test_benchmark_sample_preserves_qualified_claims():
    snapshot = _parse_envelope_claims(
        {
            "result": {
                "items": [
                    {
                        "payload": {
                            "subject_key": "service:api",
                            "object_key": "datastore:db",
                            "predicate": "USES",
                            "environment": "production",
                            "fact": "api uses db",
                            "source_ref": "test:config",
                        }
                    }
                ]
            }
        }
    )
    (edge,) = snapshot.edges
    assert (edge.from_label, edge.type, edge.to_label) == (
        "Service",
        "USES",
        "DataStore",
    )
    assert edge.properties == {
        "environment": "production",
        "fact": "api uses db",
        "source_ref": "test:config",
    }

"""Backend and result-budget regressions for the agent read surface."""

from __future__ import annotations

import pytest

from potpie_context_core.agent_envelope import AgentEnvelope, CoverageReport, EvidenceItem
from potpie_context_core.ports.claim_query import ClaimRow
from potpie_context_core.ports.graph_service import GraphEntitySearchRequest, GraphReadRequest
from potpie_context_core.ports.agent_context import ResolveRequest
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend, _Analytics,
)
from potpie_context_engine.adapters.outbound.graph.in_memory_reader import InMemoryClaimQueryStore
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.application.services.read_orchestrator import _describe_search_result
from potpie.cli.commands import graph as graph_cli, query as query_cli


pytestmark = pytest.mark.unit


def _service() -> DefaultGraphService:
    backend = InMemoryGraphBackend()
    backend.store.add(ClaimRow(
        pot_id="p", predicate="DEPENDS_ON", subject_key="service:api",
        object_key="service:ledger", claim_key="claim:dependency",
        fact="api calls ledger", source_refs=("repo:manifest",),
    ))
    backend.store.set_entity_properties(
        pot_id="p", entity_key="service:api", properties={"name": "API"},
    )
    return DefaultGraphService(backend=backend)


def test_search_and_inline_read_use_one_bulk_property_call_each(monkeypatch) -> None:
    service = _service()
    calls: list[tuple[str, ...]] = []

    def bulk(self, *, pot_id, entity_keys):
        keys = tuple(entity_keys)
        calls.append(keys)
        return {key: dict(self.entity_property_index.get((pot_id, key), {})) for key in keys}

    def scalar(self, *, pot_id, entity_key):
        raise AssertionError(f"unexpected scalar property query for {entity_key}")

    monkeypatch.setattr(InMemoryClaimQueryStore, "entity_properties_many", bulk)
    monkeypatch.setattr(InMemoryClaimQueryStore, "entity_properties", scalar)

    entities = service.search_entities(GraphEntitySearchRequest(
        pot_id="p", query="service:api", limit=1,
    )).entities
    assert entities[0].key == "service:api"
    assert entities[0].name == "API"
    assert len(calls) == 1

    calls.clear()
    result = service.read(GraphReadRequest(
        pot_id="p", subgraph="infra_topology", view="service_neighborhood",
        scope={"service": "api"}, depth=1,
    ))
    assert result.items
    # The reader and inline projection each hydrate once; neither loops over
    # entities with scalar backend calls.
    assert len(calls) == 2


def test_named_read_reuses_one_counts_snapshot(monkeypatch) -> None:
    service = _service()
    calls = {"counts": 0, "freshness": 0, "quality": 0}
    original_counts = _Analytics.counts
    original_freshness = _Analytics.freshness

    def counts(self, pot_id):
        calls["counts"] += 1
        return original_counts(self, pot_id)

    def freshness(self, pot_id):
        calls["freshness"] += 1
        return original_freshness(self, pot_id)

    def quality(self, pot_id):
        calls["quality"] += 1
        raise AssertionError("quality should be projected from the same counts")

    monkeypatch.setattr(_Analytics, "counts", counts)
    monkeypatch.setattr(_Analytics, "freshness", freshness)
    monkeypatch.setattr(_Analytics, "quality", quality)
    result = service.read(GraphReadRequest(
        pot_id="p", subgraph="infra_topology", view="service_neighborhood",
        scope={"service": "api"}, depth=1,
    ))

    assert calls == {"counts": 1, "freshness": 1, "quality": 0}
    assert result.subgraph_versions["_global"] == result.quality["backend"]["claim_count"] == 1


def test_total_budget_preserves_ranked_order_and_names_omitted_family() -> None:
    items = tuple(EvidenceItem(
        include=family, candidate_key=f"claim:{index}", score=1 - index / 10,
        payload={"fact": f"fact {index}"}, coverage_status="complete",
    ) for index, family in enumerate(("docs", "docs", "resources")))
    envelope = AgentEnvelope(
        pot_id="p", intent="docs", items=items,
        coverage=(CoverageReport(include="docs", status="complete", candidate_pool=2),
                  CoverageReport(include="resources", status="complete", candidate_pool=1)),
    )
    result = _describe_search_result(
        envelope, query="fact", searched_families=["docs", "resources"], max_items=1,
    )

    assert [item.candidate_key for item in result.items] == ["claim:0"]
    assert result.metadata["returned_by_family"] == {"docs": 1, "resources": 0}
    assert result.metadata["omitted_by_total_budget"] == {"docs": 1, "resources": 1}
    assert result.metadata["families_with_candidates_omitted"] == ["resources"]


def test_fix_cause_steps_verification_and_source_status_survive_public_reads() -> None:
    service = _service()
    store = service.backend.store
    store.add(ClaimRow(
        pot_id="p", predicate="REPRODUCES", subject_key="bug_pattern:cookie",
        object_key="service:api", claim_key="claim:symptom",
        fact="Second daemon causes a 401", source_refs=("test:symptom",),
    ))
    store.add(ClaimRow(
        pot_id="p", predicate="RESOLVED", subject_key="fix:cookie",
        object_key="bug_pattern:cookie", claim_key="claim:fix",
        fact="Use a daemon-specific cookie name", source_refs=("test:fix",),
    ))
    store.add(ClaimRow(
        pot_id="p", predicate="VERIFIED", subject_key="activity:check",
        object_key="fix:cookie", claim_key="claim:check",
        fact="Two-daemon test passed", source_refs=("test:check",),
        properties={"outcome": "passed"},
    ))
    store.set_entity_properties(
        pot_id="p", entity_key="fix:cookie", properties={
            "root_cause": "Cookie name was shared across local daemons",
            "fix_steps": ["Suffix the cookie name per daemon"],
            "verification_status": "passed", "source_status": "local only",
        },
    )
    read = service.read(GraphReadRequest(
        pot_id="p", subgraph="debugging", view="prior_occurrences",
        scope={"service": "api"}, detail="full", relations="full",
    ))
    resolved = service.resolve(ResolveRequest(
        pot_id="p", include=("prior_bugs",), scope={"service": "api"},
    ))
    for text in (
        graph_cli._read_human(read, format_="raw", sort="auto", dedupe="auto", event_limit=None),
        query_cli._envelope_human(resolved),
        str(read.to_dict()), str(resolved.to_dict()),
    ):
        assert "Cookie name was shared" in text
        assert "Suffix the cookie name" in text
        assert "passed" in text
        assert "local only" in text

    sl = service.backend.inspection.neighborhood(
        pot_id="p", entity_key="fix:cookie", depth=1, limit=10,
    )
    payload = {
        "entity_key": "fix:cookie", "identity_status": "exact", "detail": "full",
        "node_count": len(sl.nodes), "truncated": sl.truncated,
        "relations": [graph_cli._neighborhood_relation(edge) for edge in sl.edges],
        "nodes": [{"key": node.key, "labels": list(node.labels),
                   "properties": dict(node.properties)} for node in sl.nodes],
        "edges": [{"predicate": edge.predicate, "from": edge.from_key,
                   "to": edge.to_key, "properties": dict(edge.properties)} for edge in sl.edges],
    }
    neighborhood_text = graph_cli._neighborhood_human(payload)
    assert "Cookie name was shared" in neighborhood_text
    assert "Suffix the cookie name" in neighborhood_text
    assert "local only" in neighborhood_text

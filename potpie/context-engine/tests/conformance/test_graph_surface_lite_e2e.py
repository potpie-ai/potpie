"""Graph Surface Lite end-to-end acceptance tests (Graph V1.5 Step 13).

Exercises the surface through ``HostShell`` / ``DefaultGraphService`` the way an
agent would: discover → read → search → mutate → record, plus cross-process
embedded persistence and paraphrase retrieval through the bundled local embedder.
"""

from __future__ import annotations

import pytest

from potpie.context_engine.adapters.outbound.graph.backends.embedded_backend import EmbeddedGraphBackend
from potpie.context_engine.adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from potpie.context_engine.adapters.outbound.intelligence.local_embedder import build_embedder
from potpie.context_engine.application.services.graph_service import DefaultGraphService
from potpie.context_engine.domain.ports.agent_context import RecordRequest, ResolveRequest
from potpie.context_engine.domain.ports.claim_query import ClaimQueryFilter
from potpie.context_engine.domain.ports.services.graph_service import (
    GraphCatalogRequest,
    GraphEntitySearchRequest,
    GraphReadRequest,
)
from potpie.context_engine.domain.semantic_mutations import SemanticMutationRequest

pytestmark = pytest.mark.unit

POT = "local/default"


def _service(embedder=True) -> DefaultGraphService:
    be = InMemoryGraphBackend(embedder=build_embedder() if embedder else None)
    return DefaultGraphService(backend=be)


def _link_payload(**over) -> dict:
    op = {
        "op": "link_entities",
        "subgraph": "infra_topology",
        "subject": {"key": "service:payments-api", "type": "Service"},
        "predicate": "DEPENDS_ON",
        "object": {"key": "service:ledger-api", "type": "Service"},
        "truth": "source_observation",
        "evidence": [{"source_ref": "repo:manifest", "authority": "repository_metadata"}],
        "description": "payments depends on ledger to post entries",
    }
    op.update(over)
    return {"pot_id": POT, "operations": [op]}


# 1. catalog shows canonical backed views
def test_catalog_backed_and_planned() -> None:
    cat = _service().catalog(GraphCatalogRequest(pot_id=POT)).to_dict()
    backed = {v["name"] for v in cat["views"] if v["backed"]}
    assert "debugging.prior_occurrences" in backed
    assert "decisions.active_decisions" in backed


# 2. read returns data for a backed view
def test_read_backed_view_returns_data() -> None:
    svc = _service()
    svc.mutate(SemanticMutationRequest.parse(_link_payload()))
    env = svc.read(
        GraphReadRequest(
            pot_id=POT,
            subgraph="infra_topology",
            view="service_neighborhood",
            scope={"service": "payments-api"},
            depth=2,
        )
    )
    assert env.items
    assert env.view == "infra_topology.service_neighborhood"
    assert env.subgraph_versions["_global"] >= 1


# 3. search-entities finds entities from a prior mutation
def test_search_entities_finds_prior_mutation() -> None:
    svc = _service()
    svc.mutate(SemanticMutationRequest.parse(_link_payload()))
    res = svc.search_entities(
        GraphEntitySearchRequest(pot_id=POT, query="ledger", type="Service")
    )
    keys = {e.key for e in res.entities}
    assert "service:ledger-api" in keys


# 4. mutate --dry-run validates without writing
def test_dry_run_does_not_write() -> None:
    svc = _service()
    res = svc.mutate(SemanticMutationRequest.parse(_link_payload(), dry_run=True))
    assert res.status == "validated"
    assert res.would_apply is True
    # Nothing persisted.
    res2 = svc.search_entities(GraphEntitySearchRequest(pot_id=POT, query="ledger"))
    assert not res2.entities


# 5. mutate applies low-risk link_entities
def test_apply_low_risk_link() -> None:
    svc = _service()
    res = svc.mutate(SemanticMutationRequest.parse(_link_payload()))
    assert res.status == "applied"
    assert res.auto_committed
    assert res.mutations_applied["edge_upserts"] == 1
    assert res.mutation_id


# 6. mutate rejects invalid endpoint pairs
def test_reject_invalid_endpoints() -> None:
    svc = _service()
    payload = _link_payload(
        object={"key": "repo:foo", "type": "Repository"}, truth="agent_claim", evidence=[]
    )
    res = svc.mutate(SemanticMutationRequest.parse(payload))
    assert res.status == "rejected"
    assert any(i.code == "invalid_endpoints" for i in res.issues if i.is_error)


# 7. mutate gates high-risk operations on explicit approval
def test_high_risk_supersede_requires_approval() -> None:
    svc = _service()
    svc.mutate(SemanticMutationRequest.parse(_link_payload()))
    payload = {
        "pot_id": POT,
        "operations": [
            {
                "op": "supersede_claim",
                "subgraph": "infra_topology",
                "subject": {"key": "service:payments-api", "type": "Service"},
                "predicate": "DEPENDS_ON",
                "object": {"key": "service:ledger-api", "type": "Service"},
                "superseded_by": {"key": "service:ledger-v2", "type": "Service"},
                "reason": "dependency target changed",
                "description": "payments depends on ledger-v2 after the service rename",
            }
        ],
    }
    res = svc.mutate(SemanticMutationRequest.parse(payload))
    assert res.status == "review_required"
    assert res.auto_committed is False

    approved = svc.mutate(
        SemanticMutationRequest.parse(
            payload,
            allow_review_required=True,
            approved_by="user:alice",
        )
    )
    assert approved.status == "applied"
    rows = svc.backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))
    assert len(rows) == 1
    assert rows[0].object_key == "service:ledger-v2"


# 8. context_record uses the semantic mutation path with the same metadata
def test_context_record_uses_semantic_path() -> None:
    svc = _service()
    receipt = svc.record(
        RecordRequest(
            pot_id=POT,
            record_type="preference",
            summary="wrap external calls in tenacity retry",
            details={"policy_kind": "resilience", "prescription": "wrap external calls in tenacity retry"},
            scope={"service": "payments-api", "language": "python"},
        )
    )
    assert receipt.accepted
    assert receipt.metadata["graph_contract_version"] == "v1.5"
    assert receipt.metadata["truth"] == "preference"
    assert receipt.metadata["subgraph"] == "decisions"
    assert receipt.metadata["claim_keys"]
    # It surfaces through the coding_preferences reader (POLICY_APPLIES_TO).
    env = svc.resolve(
        ResolveRequest(
            pot_id=POT,
            intent="feature",
            include=("coding_preferences",),
            scope={"service": "payments-api", "language": "python"},
        )
    )
    prefs = [i for i in env.items if i.include == "coding_preferences"]
    assert prefs
    assert "tenacity" in dict(prefs[0].payload).get("fact", "")


def test_record_and_graph_mutate_produce_same_metadata() -> None:
    """context_record and graph mutate stamp the same V1.5 claim metadata."""
    svc = _service()
    svc.record(
        RecordRequest(
            pot_id=POT,
            record_type="preference",
            summary="prefer ruff for linting",
            details={"policy_kind": "style", "prescription": "prefer ruff for linting"},
            scope={"language": "python"},
        )
    )
    rows = svc.backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id=POT, predicate_in=("POLICY_APPLIES_TO",))
    )
    assert rows
    row = rows[0]
    # Same V1.5 metadata a direct graph-mutate write carries.
    assert row.truth == "preference"
    assert row.subgraph == "decisions"
    assert row.graph_contract_version == "v1.5"
    assert row.claim_key
    assert row.ontology_version == "2026-06-graph"


# 9. embedded backend persists V1.5 metadata across CLI processes
def test_embedded_persists_metadata_across_processes(tmp_path) -> None:
    # Process 1: write through a fresh embedded backend.
    be1 = EmbeddedGraphBackend(home=tmp_path, embedder=build_embedder())
    svc1 = DefaultGraphService(backend=be1)
    res = svc1.mutate(SemanticMutationRequest.parse(_link_payload()))
    assert res.status == "applied"

    # Process 2: a brand-new backend instance reads from the same JSON file.
    be2 = EmbeddedGraphBackend(home=tmp_path, embedder=build_embedder())
    svc2 = DefaultGraphService(backend=be2)
    rows = svc2.backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id=POT, predicate_in=("DEPENDS_ON",))
    )
    assert rows
    row = rows[0]
    assert row.truth == "source_observation"
    assert row.subgraph == "infra_topology"
    assert row.graph_contract_version == "v1.5"
    # The persisted embedding survived the JSON round-trip.
    assert row.fact_embedding is not None and len(row.fact_embedding) > 0


# 12. an agent-authored description is embedded locally and a paraphrase retrieves it
def test_paraphrase_retrieval_via_local_embedder() -> None:
    svc = _service(embedder=True)
    # Two competing claims; the relevant one is described for retrieval.
    svc.mutate(
        SemanticMutationRequest.parse(
            {
                "pot_id": POT,
                "operations": [
                    {
                        "op": "assert_claim",
                        "subgraph": "decisions",
                        "subject": {"key": "preference:retry-external", "type": "Preference"},
                        "predicate": "POLICY_APPLIES_TO",
                        "object": {"key": "service:payments-api", "type": "Service"},
                        "truth": "preference",
                        "description": "wrap external calls in tenacity retry with backoff",
                    },
                    {
                        "op": "assert_claim",
                        "subgraph": "decisions",
                        "subject": {"key": "preference:tabs", "type": "Preference"},
                        "predicate": "POLICY_APPLIES_TO",
                        "object": {"key": "service:payments-api", "type": "Service"},
                        "truth": "preference",
                        "description": "use four-space indentation in python files",
                    },
                ],
            }
        )
    )
    rows = svc.backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id=POT,
            predicate_in=("POLICY_APPLIES_TO",),
            fact_query="add retries to outbound payment calls",
        )
    )
    # The retry preference ranks above the indentation one on a paraphrase that
    # shares no exact words with either (real vector recall, no API key).
    assert rows
    assert rows[0].subject_key == "preference:retry-external"


def test_match_mode_is_vector_with_embedder_lexical_without() -> None:
    assert _service(embedder=True).data_plane_status(POT).match_mode == "vector"
    assert _service(embedder=False).data_plane_status(POT).match_mode == "lexical"

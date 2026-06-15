"""Graph Surface Lite contract tests (Graph V1.5 Step 0).

Locks the catalog contract, the honest op partitioning, and — critically — that
the MCP surface still exposes exactly the four ``context_*`` tools (the new
graph surface is CLI-only in V1.5).
"""

from __future__ import annotations

from typing import Any

import pytest

from adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from application.services.graph_service import DefaultGraphService
from domain.graph_contract import GRAPH_CONTRACT_VERSION, ONTOLOGY_VERSION
from domain.graph_mutations import ProvenanceContext
from domain.ports.agent_context import RecordRequest
from domain.ports.claim_query import ClaimRow
from domain.ports.services.graph_service import (
    GraphCatalogRequest,
    GraphEntitySearchRequest,
    GraphReadRequest,
)
from domain.reconciliation import MutationBatch, MutationResult, MutationSummary
from domain.semantic_mutations import SemanticMutationRequest

pytestmark = pytest.mark.unit


@pytest.fixture()
def service() -> DefaultGraphService:
    return DefaultGraphService(backend=InMemoryGraphBackend())


def test_catalog_reports_contract_and_ontology_version(service) -> None:
    cat = service.catalog(GraphCatalogRequest(pot_id="p")).to_dict()
    assert cat["graph_contract_version"] == GRAPH_CONTRACT_VERSION == "v1.5"
    assert cat["ontology_version"] == ONTOLOGY_VERSION


def test_catalog_lists_the_four_commands(service) -> None:
    cat = service.catalog(GraphCatalogRequest(pot_id="p")).to_dict()
    assert cat["commands"] == ["catalog", "read", "search-entities", "mutate"]


def test_catalog_truth_classes(service) -> None:
    cat = service.catalog(GraphCatalogRequest(pot_id="p")).to_dict()
    assert "authoritative_fact" in cat["truth_classes"]
    assert "agent_claim" in cat["truth_classes"]
    assert "preference" in cat["truth_classes"]


def test_catalog_op_partitions_are_honest(service) -> None:
    cat = service.catalog(GraphCatalogRequest(pot_id="p")).to_dict()
    # Only applicable ops are advertised as mutation_operations.
    assert "link_entities" in cat["mutation_operations"]
    assert "assert_claim" in cat["mutation_operations"]
    # supersede/merge are review_required, NOT applicable.
    assert "supersede_claim" in cat["review_required_operations"]
    assert "supersede_claim" not in cat["mutation_operations"]
    # patch/transition are deferred to V2, surfaced honestly.
    assert "patch_entity" in cat["deferred_operations"]
    assert "patch_entity" not in cat["mutation_operations"]


def test_catalog_shows_backed_and_planned_views(service) -> None:
    cat = service.catalog(GraphCatalogRequest(pot_id="p")).to_dict()
    by_name = {v["name"]: v for v in cat["views"]}
    assert by_name["bugs.prior_occurrences"]["backed"] is True
    assert by_name["decisions.active_decisions"]["backed"] is True
    assert by_name["ownership.owner_context"]["backed"] is True
    assert by_name["docs.reference_context"]["backed"] is True


def test_catalog_filters_by_subgraph(service) -> None:
    cat = service.catalog(GraphCatalogRequest(pot_id="p", subgraph="bugs")).to_dict()
    assert {v["subgraph"] for v in cat["views"]} == {"bugs"}
    assert [v["name"] for v in cat["views"]] == ["bugs.prior_occurrences"]


def test_catalog_rejects_unknown_subgraph(service) -> None:
    with pytest.raises(ValueError, match="unknown graph subgraph"):
        service.catalog(GraphCatalogRequest(pot_id="p", subgraph="missing"))


def test_catalog_includes_ranking_inputs(service) -> None:
    cat = service.catalog(GraphCatalogRequest(pot_id="p")).to_dict()
    by_name = {v["name"]: v for v in cat["views"]}
    assert "ranking_inputs" in by_name["bugs.prior_occurrences"]
    assert "semantic_similarity" in by_name["bugs.prior_occurrences"]["ranking_inputs"]
    assert (
        "resolution_status" not in by_name["bugs.prior_occurrences"]["ranking_inputs"]
    )


def test_catalog_carries_entity_types_and_predicates(service) -> None:
    cat = service.catalog(GraphCatalogRequest(pot_id="p")).to_dict()
    labels = {e["label"] for e in cat["entity_types"]}
    assert {"Service", "BugPattern", "Preference", "CodeAsset", "Adapter"} <= labels
    predicates = {p["name"] for p in cat["predicates"]}
    assert {
        "DEPENDS_ON",
        "POLICY_APPLIES_TO",
        "REPRODUCES",
        "USES_ADAPTER",
    } <= predicates


def test_catalog_reports_match_mode(service) -> None:
    # No embedder wired in this backend → lexical, labeled (not silent).
    cat = service.catalog(GraphCatalogRequest(pot_id="p")).to_dict()
    assert cat["match_mode"] in ("vector", "lexical")


def test_read_assembles_inline_relations_for_view(service) -> None:
    request = SemanticMutationRequest.parse(
        {
            "pot_id": "p",
            "operations": [
                {
                    "op": "link_entities",
                    "subgraph": "infra_topology",
                    "subject": {
                        "key": "service:payments-api",
                        "type": "Service",
                        "summary": "Payments API service.",
                    },
                    "predicate": "DEPENDS_ON",
                    "object": {
                        "key": "service:ledger-api",
                        "type": "Service",
                        "description": "Ledger API service.",
                    },
                    "truth": "source_observation",
                    "evidence": [{"source_ref": "repo:manifest"}],
                    "description": "payments depends on ledger to post entries",
                }
            ],
        }
    )
    service.mutate(request)

    env = service.read(
        GraphReadRequest(
            pot_id="p",
            view="infra_topology.service_neighborhood",
            scope={"service": "payments-api"},
            depth=1,
        )
    )

    assert env.metadata["read_shape"] == "entity_relations"
    by_key = {i.candidate_key: dict(i.payload) for i in env.items}
    payments = by_key["service:payments-api"]
    assert payments["entity"]["labels"] == ["Service"]
    assert payments["entity"]["summary"] == "Payments API service."
    assert payments["entity"]["description"] == "Payments API service."
    ledger = by_key["service:ledger-api"]
    assert ledger["entity"]["summary"] == "Ledger API service."
    assert ledger["entity"]["description"] == "Ledger API service."
    dep_rel = next(
        rel
        for rel in payments["relations"]
        if rel["predicate"] == "DEPENDS_ON" and rel["direction"] == "out"
    )
    assert dep_rel["related_entity"]["summary"] == "Ledger API service."
    assert any(
        rel["predicate"] == "DEPENDS_ON"
        and rel["direction"] == "out"
        and rel["related_key"] == "service:ledger-api"
        for rel in payments["relations"]
    )


def test_preferences_view_assembles_structured_policy_relation(service) -> None:
    request = SemanticMutationRequest.parse(
        {
            "pot_id": "p",
            "operations": [
                {
                    "op": "assert_claim",
                    "subgraph": "preferences",
                    "subject": {
                        "key": "preference:cli-errors",
                        "type": "Preference",
                        "properties": {
                            "policy_kind": "error_handling",
                            "prescription": "Emit structured CLI errors with next actions.",
                            "strength": "strong",
                            "audience": "path",
                        },
                    },
                    "predicate": "POLICY_APPLIES_TO",
                    "object": {
                        "key": "code:potpie-context-engine:adapters/inbound/cli",
                        "type": "CodeAsset",
                    },
                    "truth": "preference",
                    "description": "CLI graph commands should return structured errors.",
                    "extra": {
                        "file_path": "potpie/context-engine/adapters/inbound/cli",
                        "language": "python",
                    },
                }
            ],
        }
    )
    service.mutate(request)

    env = service.read(
        GraphReadRequest(
            pot_id="p",
            view="preferences.active_preferences",
            scope={
                "path": "potpie/context-engine/adapters/inbound/cli/commands/graph.py",
                "language": "python",
            },
            limit=5,
        )
    )

    assert env.metadata["read_shape"] == "entity_relations"
    rels = [rel for item in env.items for rel in dict(item.payload)["relations"]]
    policy_rel = next(rel for rel in rels if rel["predicate"] == "POLICY_APPLIES_TO")
    assert policy_rel["properties"]["prescription"].startswith("Emit structured CLI")
    assert policy_rel["properties"]["policy_kind"] == "error_handling"


def test_read_dedupes_duplicate_inline_relation_rows(service) -> None:
    store = service.backend.claim_query
    row = ClaimRow(
        pot_id="p",
        predicate="POLICY_APPLIES_TO",
        subject_key="preference:cli-errors",
        object_key="code:potpie-context-engine:adapters/inbound/cli",
        claim_key="claim:cli-errors-policy",
        truth="preference",
        evidence_strength="attested",
        source_ref="repo:cli-guide",
        source_refs=("repo:cli-guide",),
        fact="CLI graph commands should return structured errors.",
        properties={
            "code_scope": {"language": "python"},
            "policy_kind": "error_handling",
            "prescription": "Emit structured CLI errors with next actions.",
        },
    )
    store.add(row)
    store.add(row)

    env = service.read(
        GraphReadRequest(
            pot_id="p",
            view="preferences.active_preferences",
            scope={"language": "python"},
            limit=5,
        )
    )

    assert env.metadata["read_shape"] == "entity_relations"
    assert env.metadata["inline_relation_count"] == 2
    by_key = {item.candidate_key: dict(item.payload) for item in env.items}
    assert len(by_key["preference:cli-errors"]["relations"]) == 1
    assert (
        by_key["preference:cli-errors"]["relations"][0]["related_key"]
        == "code:potpie-context-engine:adapters/inbound/cli"
    )
    assert (
        len(by_key["code:potpie-context-engine:adapters/inbound/cli"]["relations"]) == 1
    )


def test_features_view_does_not_return_generic_infra_edges(service) -> None:
    request = SemanticMutationRequest.parse(
        {
            "pot_id": "p",
            "operations": [
                {
                    "op": "link_entities",
                    "subgraph": "infra_topology",
                    "subject": {"key": "service:search-api", "type": "Service"},
                    "predicate": "DEFINED_IN",
                    "object": {
                        "key": "repo:github.com/acme/widgets",
                        "type": "Repository",
                    },
                    "truth": "source_observation",
                    "evidence": [{"source_ref": "repo:manifest"}],
                    "description": "search api lives in widgets repo",
                },
                {
                    "op": "assert_claim",
                    "subgraph": "features",
                    "subject": {
                        "key": "repo:github.com/acme/widgets",
                        "type": "Repository",
                    },
                    "predicate": "PROVIDES",
                    "object": {
                        "key": "feature:search",
                        "type": "Feature",
                        "summary": "Search capability",
                    },
                    "truth": "source_observation",
                    "evidence": [{"source_ref": "repo:README"}],
                    "description": "README says widgets provides search.",
                },
            ],
        }
    )
    service.mutate(request)

    env = service.read(
        GraphReadRequest(
            pot_id="p",
            view="features.provided",
            scope={"anchor_entity_key": "repo:github.com/acme/widgets"},
        )
    )

    assert env.metadata["read_shape"] == "entity_relations"
    rels = [rel for item in env.items for rel in dict(item.payload)["relations"]]
    assert {rel["predicate"] for rel in rels} == {"PROVIDES"}


def test_search_entities_derives_summary_for_old_nodes_without_summary(service) -> None:
    store = service.backend.claim_query
    store.add(
        ClaimRow(
            pot_id="p",
            predicate="DEPENDS_ON",
            subject_key="service:web",
            object_key="service:auth",
            fact="web calls auth",
        )
    )
    store.set_entity_label(
        pot_id="p", entity_key="service:web", labels=("Entity", "Service")
    )
    store.set_entity_properties(
        pot_id="p", entity_key="service:web", properties={"name": "web"}
    )

    result = service.search_entities(
        GraphEntitySearchRequest(pot_id="p", query="web auth", type="Service")
    ).to_dict()

    web = result["entities"][0]
    assert web["key"] == "service:web"
    assert web["summary"] == "web"
    assert web["description"] == "web"


def test_search_entities_projects_canonical_labels_from_key_prefix(service) -> None:
    store = service.backend.claim_query
    store.add(
        ClaimRow(
            pot_id="p",
            predicate="PROVIDES",
            subject_key="repo:github.com/potpie-ai/potpie",
            object_key="feature:context-graph",
            fact="potpie repo provides context graph memory",
            properties={"semantic_similarity": 0.9},
        )
    )
    store.set_entity_label(
        pot_id="p",
        entity_key="repo:github.com/potpie-ai/potpie",
        labels=("Entity", "Repository", "Feature", "APIContract"),
    )

    result = service.search_entities(
        GraphEntitySearchRequest(pot_id="p", query="context graph", type="Repository")
    ).to_dict()

    repo = next(
        e for e in result["entities"] if e["key"] == "repo:github.com/potpie-ai/potpie"
    )
    assert repo["labels"] == ["Repository"]


def test_read_projects_canonical_labels_from_key_prefix(service) -> None:
    store = service.backend.claim_query
    store.add(
        ClaimRow(
            pot_id="p",
            predicate="PROVIDES",
            subject_key="repo:github.com/potpie-ai/potpie",
            object_key="feature:context-graph",
            fact="potpie repo provides context graph memory",
            claim_key="claim:provides",
            subgraph="features",
        )
    )
    store.set_entity_label(
        pot_id="p",
        entity_key="repo:github.com/potpie-ai/potpie",
        labels=("Entity", "Repository", "Feature", "APIContract"),
    )
    store.set_entity_label(
        pot_id="p",
        entity_key="feature:context-graph",
        labels=("Entity", "Feature", "Adapter"),
    )

    env = service.read(
        GraphReadRequest(
            pot_id="p",
            view="features.provided",
            scope={"anchor_entity_key": "repo:github.com/potpie-ai/potpie"},
        )
    )

    by_key = {i.candidate_key: dict(i.payload) for i in env.items}
    assert by_key["repo:github.com/potpie-ai/potpie"]["entity"]["labels"] == [
        "Repository"
    ]
    assert by_key["feature:context-graph"]["entity"]["labels"] == ["Feature"]


def test_fix_record_creates_scoped_bug_and_failed_attempt_memory(service) -> None:
    receipt = service.record(
        RecordRequest(
            pot_id="p",
            record_type="fix",
            summary="Pass --pot when graph scope is ambiguous.",
            details={
                "symptom_signature": "graph read fails with ambiguous pot",
                "fix_steps": ["Pass --pot or configure the active pot."],
                "attempted_failed_fixes": ["Retry graph read without a pot."],
            },
            scope={"service": "context-engine"},
        )
    )

    assert receipt.accepted
    env = service.read(
        GraphReadRequest(
            pot_id="p",
            view="bugs.prior_occurrences",
            query="ambiguous pot graph read",
            scope={"service": "context-engine"},
            limit=10,
        )
    )

    rels = [rel for item in env.items for rel in dict(item.payload)["relations"]]
    predicates = {rel["predicate"] for rel in rels}
    assert {"REPRODUCES", "RESOLVED", "ATTEMPTED_FIX_FAILED"} <= predicates


def test_decision_record_creates_decided_and_affects_memory(service) -> None:
    receipt = service.record(
        RecordRequest(
            pot_id="p",
            record_type="decision",
            summary="Use semantic graph mutations for durable context writes.",
            details={
                "title": "Semantic mutations own context writes",
                "rationale": "One validated write path keeps graph memory coherent.",
                "affects_refs": ["service:context-engine", "code:context-engine:graph"],
            },
            scope={"service": "context-engine"},
        )
    )

    assert receipt.accepted
    env = service.read(
        GraphReadRequest(
            pot_id="p",
            view="decisions.active_decisions",
            query="semantic mutations context writes",
            scope={"service": "context-engine"},
            limit=10,
        )
    )

    rels = [rel for item in env.items for rel in dict(item.payload)["relations"]]
    assert {"DECIDED", "AFFECTS"} <= {rel["predicate"] for rel in rels}


# --- mutate provenance -------------------------------------------------------


class _CapturingMutation:
    def __init__(self) -> None:
        self.calls: list[tuple[MutationBatch, str, ProvenanceContext | None]] = []

    def apply(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> MutationResult:
        self.calls.append((plan, expected_pot_id, provenance_context))
        return MutationResult(
            ok=True,
            mutation_id="mutation-1",
            mutation_summary=MutationSummary(
                entity_upserts_applied=len(plan.entity_upserts),
                edge_upserts_applied=len(plan.edge_upserts),
                invalidations_applied=len(plan.invalidations),
            ),
        )

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"unexpected mutation port call: {name}")


def test_mutate_passes_lowered_provenance_to_mutation_port() -> None:
    backend = InMemoryGraphBackend()
    mutation = _CapturingMutation()
    backend._mutation = mutation
    service = DefaultGraphService(backend=backend)
    request = SemanticMutationRequest.parse(
        {
            "pot_id": "p",
            "idempotency_key": "idem-1",
            "created_by": {
                "surface": "cli",
                "harness": "codex",
                "user": "user:alice",
            },
            "operations": [
                {
                    "op": "link_entities",
                    "subgraph": "infra_topology",
                    "subject": {"key": "service:payments-api", "type": "Service"},
                    "predicate": "DEPENDS_ON",
                    "object": {"key": "service:ledger-api", "type": "Service"},
                    "truth": "source_observation",
                    "evidence": [{"source_ref": "repo:manifest"}],
                    "description": "payments calls ledger to post entries",
                }
            ],
        }
    )

    result = service.mutate(request)

    assert result.ok is True
    assert len(mutation.calls) == 1
    _batch, expected_pot_id, provenance = mutation.calls[0]
    assert expected_pot_id == "p"
    assert provenance is not None
    assert provenance.source_ref == "idem-1"
    assert provenance.created_by_agent == "codex"
    assert provenance.actor_user_id == "user:alice"
    assert provenance.actor_surface == "cli"
    assert provenance.actor_client_name == "codex"


# --- MCP non-negotiable: exactly four context_* tools -----------------------


def test_mcp_exposes_exactly_four_context_tools() -> None:
    """The Graph Surface Lite surface is CLI-only in V1.5; MCP stays at four."""
    import asyncio

    from adapters.inbound.mcp import server

    tools = asyncio.run(server.mcp.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "context_resolve",
        "context_search",
        "context_record",
        "context_status",
    }
    # No graph_* tools leaked onto the MCP surface.
    assert not any(n.startswith("graph") for n in names)

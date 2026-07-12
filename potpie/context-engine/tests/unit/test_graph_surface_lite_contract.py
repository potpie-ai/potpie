"""Graph Surface Lite contract tests (Graph V1.5 Step 0).

Locks the catalog contract, the honest op partitioning, and — critically — that
the MCP surface still exposes exactly the four ``context_*`` tools (the new
graph surface is CLI-only in V1.5).
"""

from __future__ import annotations

from typing import Any

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.domain.graph_contract import (
    GRAPH_CONTRACT_VERSION,
    ONTOLOGY_VERSION,
)
from potpie_context_engine.domain.graph_mutations import ProvenanceContext
from potpie_context_engine.domain.ports.agent_context import RecordRequest
from potpie_context_engine.domain.ports.claim_query import ClaimRow
from potpie_context_engine.domain.ports.services.graph_service import (
    GraphCatalogRequest,
    GraphEntitySearchRequest,
    GraphReadRequest,
)
from potpie_context_engine.domain.graph_views import UnknownGraphViewError
from potpie_context_engine.domain.reconciliation import (
    MutationBatch,
    MutationResult,
    MutationSummary,
)
from potpie_context_engine.domain.semantic_mutations import SemanticMutationRequest

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
    # 7B high-risk corrections are applicable through plan/commit approval.
    assert "supersede_claim" in cat["mutation_operations"]
    assert "merge_duplicate_entities" in cat["mutation_operations"]
    assert "supersede_claim" not in cat["review_required_operations"]
    assert "merge_duplicate_entities" not in cat["review_required_operations"]
    # Phase 7A entity correction ops are applicable through plan/commit gating.
    assert "patch_entity" in cat["mutation_operations"]
    assert "transition_state" in cat["mutation_operations"]
    assert "patch_entity" not in cat["deferred_operations"]
    assert "transition_state" not in cat["deferred_operations"]


def test_catalog_shows_backed_and_planned_views(service) -> None:
    cat = service.catalog(GraphCatalogRequest(pot_id="p")).to_dict()
    by_name = {v["name"]: v for v in cat["views"]}
    assert by_name["debugging.prior_occurrences"]["backed"] is True
    assert by_name["decisions.active_decisions"]["backed"] is True
    assert by_name["code_topology.ownership_by_path"]["backed"] is True
    assert by_name["knowledge.document_context"]["backed"] is True


def test_catalog_filters_by_subgraph(service) -> None:
    cat = service.catalog(
        GraphCatalogRequest(pot_id="p", subgraph="debugging")
    ).to_dict()
    assert {v["subgraph"] for v in cat["views"]} == {"debugging"}
    assert [v["name"] for v in cat["views"]] == ["debugging.prior_occurrences"]


def test_catalog_rejects_unknown_subgraph(service) -> None:
    with pytest.raises(ValueError, match="unknown graph subgraph"):
        service.catalog(GraphCatalogRequest(pot_id="p", subgraph="missing"))


def test_catalog_unknown_subgraph_carries_include_guidance(service) -> None:
    # Audit item 17 first-contact path: catalog is the first command agents
    # run, so `graph catalog --subgraph docs` gets the same migration
    # guidance as read/describe.
    with pytest.raises(ValueError, match="knowledge.document_context") as e:
        service.catalog(GraphCatalogRequest(pot_id="p", subgraph="docs"))
    err = e.value
    assert err.detail["did_you_mean"]["matched_include"] == "docs"
    assert err.recommended_next_action == (
        "potpie graph read --subgraph knowledge --view document_context"
    )


def test_catalog_includes_ranking_inputs(service) -> None:
    cat = service.catalog(GraphCatalogRequest(pot_id="p")).to_dict()
    by_name = {v["name"]: v for v in cat["views"]}
    assert "ranking_inputs" in by_name["debugging.prior_occurrences"]
    assert (
        "semantic_similarity"
        in by_name["debugging.prior_occurrences"]["ranking_inputs"]
    )
    assert (
        "resolution_status"
        not in by_name["debugging.prior_occurrences"]["ranking_inputs"]
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
            subgraph="infra_topology",
            view="service_neighborhood",
            scope={"service": "payments-api"},
            depth=1,
        )
    )

    assert env.read_shape == "entity_relations"
    by_key = {item["entity_key"]: item for item in env.items}
    payments = by_key["service:payments-api"]
    assert payments["entity_type"] == "Service"
    assert payments["summary"] == "Payments API service."
    ledger = by_key["service:ledger-api"]
    assert ledger["summary"] == "Ledger API service."
    dep_rel = next(
        rel
        for rel in payments["relations"]
        if rel["predicate"] == "DEPENDS_ON" and rel["direction"] == "out"
    )
    assert dep_rel["related_key"] == "service:ledger-api"
    assert any(
        rel["predicate"] == "DEPENDS_ON"
        and rel["direction"] == "out"
        and rel["related_key"] == "service:ledger-api"
        for rel in payments["relations"]
    )


def test_read_to_dict_defaults_to_compact_relation_summaries(service) -> None:
    request = SemanticMutationRequest.parse(
        {
            "pot_id": "p",
            "operations": [
                {
                    "op": "link_entities",
                    "subgraph": "infra_topology",
                    "subject": {"key": "service:payments-api", "type": "Service"},
                    "predicate": "DEPENDS_ON",
                    "object": {"key": "service:ledger-api", "type": "Service"},
                    "truth": "source_observation",
                    "evidence": [{"source_ref": "repo:manifest"}],
                    "description": "payments depends on ledger",
                }
            ],
        }
    )
    service.mutate(request)

    env = service.read(
        GraphReadRequest(
            pot_id="p",
            subgraph="infra_topology",
            view="service_neighborhood",
            scope={"service": "payments-api"},
            depth=1,
        )
    )
    payload = env.to_dict()

    assert payload["detail"] == "compact"
    assert payload["relations_detail"] == "summary"
    item = next(
        item
        for item in payload["items"]
        if item["entity_key"] == "service:payments-api"
    )
    assert "relations" not in item
    assert item["relation_count"] >= 1
    assert "DEPENDS_ON" in item["relation_predicates"]


def test_read_to_dict_full_detail_preserves_relation_payload(service) -> None:
    request = SemanticMutationRequest.parse(
        {
            "pot_id": "p",
            "operations": [
                {
                    "op": "link_entities",
                    "subgraph": "infra_topology",
                    "subject": {"key": "service:payments-api", "type": "Service"},
                    "predicate": "DEPENDS_ON",
                    "object": {"key": "service:ledger-api", "type": "Service"},
                    "truth": "source_observation",
                    "evidence": [{"source_ref": "repo:manifest"}],
                    "description": "payments depends on ledger",
                }
            ],
        }
    )
    service.mutate(request)

    env = service.read(
        GraphReadRequest(
            pot_id="p",
            subgraph="infra_topology",
            view="service_neighborhood",
            scope={"service": "payments-api"},
            depth=1,
            detail="full",
            relations="full",
        )
    )
    payload = env.to_dict()

    assert payload["detail"] == "full"
    item = next(
        item
        for item in payload["items"]
        if item["entity_key"] == "service:payments-api"
    )
    assert item["relations"][0]["predicate"] == "DEPENDS_ON"
    assert "breakdown" in item


def test_preferences_view_assembles_structured_policy_relation(service) -> None:
    request = SemanticMutationRequest.parse(
        {
            "pot_id": "p",
            "operations": [
                {
                    "op": "assert_claim",
                    "subgraph": "decisions",
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
            subgraph="decisions",
            view="preferences_for_scope",
            scope={
                "path": "potpie/context-engine/adapters/inbound/cli/commands/graph.py",
                "language": "python",
            },
            limit=5,
        )
    )

    assert env.read_shape == "entity_relations"
    rels = [rel for item in env.items for rel in item["relations"]]
    policy_rel = next(rel for rel in rels if rel["predicate"] == "POLICY_APPLIES_TO")
    assert policy_rel["properties"]["prescription"].startswith("Emit structured CLI")
    assert policy_rel["properties"]["policy_kind"] == "error_handling"


def test_preferences_view_accepts_direct_service_scope(service) -> None:
    store = service.backend.claim_query
    store.add(
        ClaimRow(
            pot_id="p",
            predicate="POLICY_APPLIES_TO",
            subject_key="preference:service-errors",
            object_key="service:context-engine",
            claim_key="claim:service-errors",
            truth="preference",
            evidence_strength="attested",
            source_refs=("repo:prefs",),
            fact="Context engine service code should emit structured errors.",
            properties={
                "policy_kind": "error_handling",
                "prescription": "Emit structured errors from service boundaries.",
            },
        )
    )

    env = service.read(
        GraphReadRequest(
            pot_id="p",
            subgraph="decisions",
            view="preferences_for_scope",
            scope={"service": "context-engine"},
            limit=5,
        )
    )

    assert env.unsupported == ()
    assert env.items
    assert env.items[0]["entity_key"] == "preference:service-errors"


def test_infra_read_accepts_include_unqualified_environment_filter(service) -> None:
    service.backend.claim_query.add(
        ClaimRow(
            pot_id="p",
            predicate="DEPENDS_ON",
            subject_key="service:payments-api",
            object_key="service:ledger-api",
            claim_key="claim:payments-ledger",
            truth="source_observation",
            source_refs=("repo:manifest",),
            fact="payments depends on ledger",
        )
    )

    env = service.read(
        GraphReadRequest(
            pot_id="p",
            subgraph="infra_topology",
            view="service_neighborhood",
            scope={
                "service": "payments-api",
                "include_unqualified_environment": "true",
            },
            environment="prod",
            depth=1,
        )
    )

    assert env.unsupported == ()
    assert env.items
    assert env.source_refs == ("repo:manifest",)


def test_timeline_read_filters_by_exact_source_ref(service) -> None:
    store = service.backend.claim_query
    store.add(
        ClaimRow(
            pot_id="p",
            predicate="TOUCHED",
            subject_key="activity:github:issue-881",
            object_key="repo:github.com/potpie-ai/potpie",
            claim_key="claim:issue-881",
            truth="timeline_event",
            source_refs=("github:potpie-ai/potpie#issue/881",),
            fact="Issue 881 reported graph source-ref lookup gaps.",
        )
    )
    store.add(
        ClaimRow(
            pot_id="p",
            predicate="TOUCHED",
            subject_key="activity:github:issue-882",
            object_key="repo:github.com/potpie-ai/potpie",
            claim_key="claim:issue-882",
            truth="timeline_event",
            source_refs=("github:potpie-ai/potpie#issue/882",),
            fact="Issue 882 is unrelated.",
        )
    )

    env = service.read(
        GraphReadRequest(
            pot_id="p",
            subgraph="recent_changes",
            view="timeline",
            source_refs=("github:potpie-ai/potpie#issue/881",),
            limit=10,
        )
    )

    assert env.source_refs == ("github:potpie-ai/potpie#issue/881",)
    assert "activity:github:issue-881" in {item["entity_key"] for item in env.items}
    assert all(
        item["source_refs"] == ["github:potpie-ai/potpie#issue/881"]
        for item in env.items
    )


def test_entity_search_filters_by_exact_source_ref(service) -> None:
    service.backend.claim_query.add(
        ClaimRow(
            pot_id="p",
            predicate="PROVIDES",
            subject_key="repo:github.com/potpie-ai/potpie",
            object_key="feature:context-graph",
            claim_key="claim:context-graph",
            truth="source_observation",
            source_refs=("github:potpie-ai/potpie#pull/955",),
            fact="The repo provides context graph commands.",
        )
    )
    service.backend.claim_query.add(
        ClaimRow(
            pot_id="p",
            predicate="PROVIDES",
            subject_key="repo:github.com/potpie-ai/potpie",
            object_key="feature:other",
            claim_key="claim:other",
            truth="source_observation",
            source_refs=("github:potpie-ai/potpie#pull/956",),
            fact="The repo provides another feature.",
        )
    )

    result = service.search_entities(
        GraphEntitySearchRequest(
            pot_id="p",
            query="context graph",
            source_refs=("github:potpie-ai/potpie#pull/955",),
            limit=10,
        )
    )

    assert {entity.key for entity in result.entities} == {
        "repo:github.com/potpie-ai/potpie",
        "feature:context-graph",
    }


def test_entity_search_external_id_matches_claim_source_ref(service) -> None:
    service.backend.claim_query.add(
        ClaimRow(
            pot_id="p",
            predicate="TOUCHED",
            subject_key="activity:github:issue-881",
            object_key="repo:github.com/potpie-ai/potpie",
            claim_key="claim:issue-881",
            truth="timeline_event",
            source_refs=("github:potpie-ai/potpie#issue/881",),
            fact="Issue 881 reported graph source-ref lookup gaps.",
        )
    )

    result = service.search_entities(
        GraphEntitySearchRequest(
            pot_id="p",
            query="source ref lookup",
            external_id="github:potpie-ai/potpie#issue/881",
            limit=10,
        )
    )

    assert {entity.key for entity in result.entities} == {
        "activity:github:issue-881",
        "repo:github.com/potpie-ai/potpie",
    }


def test_entity_search_filters_by_source_system_and_family(service) -> None:
    store = service.backend.claim_query
    store.add(
        ClaimRow(
            pot_id="p",
            predicate="TOUCHED",
            subject_key="activity:github:issue-881",
            object_key="repo:github.com/potpie-ai/potpie",
            claim_key="claim:github-issue",
            truth="timeline_event",
            source_system="github",
            source_refs=("github:potpie-ai/potpie#issue/881",),
            fact="GitHub issue 881 tracks source-ref lookup.",
        )
    )
    store.add(
        ClaimRow(
            pot_id="p",
            predicate="TOUCHED",
            subject_key="activity:linear:eng-881",
            object_key="repo:github.com/potpie-ai/potpie",
            claim_key="claim:linear-issue",
            truth="timeline_event",
            source_system="linear",
            source_refs=("linear:ENG-881",),
            fact="Linear issue 881 tracks a different item.",
        )
    )

    result = service.search_entities(
        GraphEntitySearchRequest(
            pot_id="p",
            query="issue 881",
            source_system="github",
            source_family="github",
            limit=10,
        )
    )

    assert "activity:github:issue-881" in {entity.key for entity in result.entities}
    assert "activity:linear:eng-881" not in {entity.key for entity in result.entities}


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
            subgraph="decisions",
            view="preferences_for_scope",
            scope={"language": "python"},
            limit=5,
        )
    )

    assert env.read_shape == "entity_relations"
    assert env.inline_relation_count == 2
    by_key = {item["entity_key"]: item for item in env.items}
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
            subgraph="features",
            view="feature_context",
            scope={"anchor_entity_key": "repo:github.com/acme/widgets"},
        )
    )

    assert env.read_shape == "entity_relations"
    rels = [rel for item in env.items for rel in item["relations"]]
    assert {rel["predicate"] for rel in rels} == {"PROVIDES"}


def test_read_returns_unsupported_for_filters_outside_view_contract(service) -> None:
    env = service.read(
        GraphReadRequest(
            pot_id="p",
            subgraph="debugging",
            view="prior_occurrences",
            query="timeout",
            scope={"service": "api", "language": "python"},
        )
    )

    assert env.items == ()
    assert env.unsupported[0]["name"] == "language"
    assert env.coverage[0]["status"] == "unsupported"


def test_read_missing_required_scope_is_validation_failure(service) -> None:
    env = service.read(
        GraphReadRequest(
            pot_id="p",
            subgraph="features",
            view="feature_context",
            limit=5,
        )
    )

    body = env.to_dict()
    assert body["ok"] is False
    assert body["status"] == "missing_required_scope"
    assert "requires one of" in body["message"]
    assert env.items == ()
    assert env.unsupported[0]["reason"] == "missing_required_scope"
    assert env.coverage[0]["status"] == "unsupported"
    assert env.quality["reason"] == "missing_required_scope"


def test_describe_routes_through_service(service) -> None:
    # `graph describe` must answer from the service (daemon-side ontology),
    # not a CLI-local contract lookup — same routing as every graph command.
    from potpie_context_engine.domain.ports.services.graph_service import (
        GraphDescribeRequest,
    )

    payload = service.describe(
        GraphDescribeRequest(subgraph="debugging", view="prior_occurrences")
    )
    assert payload["contract_kind"] == "graph_workbench_ontology"
    assert payload["view"]["name"] == "debugging.prior_occurrences"

    with pytest.raises(ValueError, match="knowledge.document_context") as e:
        service.describe(GraphDescribeRequest(subgraph="docs"))
    assert e.value.detail["did_you_mean"]["matched_include"] == "docs"


def test_read_unknown_view_suggests_canonical_view_for_include_guess(service) -> None:
    # Audit item 17: `--subgraph docs` guesses the include family; the error
    # must return migration guidance, never accept the legacy name as input.
    with pytest.raises(UnknownGraphViewError, match="knowledge.document_context") as e:
        service.read(GraphReadRequest(pot_id="p", subgraph="docs", view="relevant"))
    err = e.value
    assert err.detail["did_you_mean"]["view"] == "knowledge.document_context"
    assert err.detail["did_you_mean"]["matched_include"] == "docs"
    assert err.recommended_next_action == (
        "potpie graph read --subgraph knowledge --view document_context"
    )


def test_read_unknown_view_without_guidance_stays_plain(service) -> None:
    with pytest.raises(ValueError, match="unknown graph view") as e:
        service.read(GraphReadRequest(pot_id="p", subgraph="nope", view="nada"))
    assert getattr(e.value, "detail", None) is None
    assert getattr(e.value, "recommended_next_action", None) is None


def test_read_coverage_is_keyed_by_view_name(service) -> None:
    env = service.read(
        GraphReadRequest(pot_id="p", subgraph="recent_changes", view="timeline")
    )
    assert env.coverage
    for row in env.coverage:
        assert row["view"] == "recent_changes.timeline"
        assert "include" not in row


def test_infra_read_keeps_environment_qualified_edges_isolated(service) -> None:
    store = service.backend.claim_query
    store.add(
        ClaimRow(
            pot_id="p",
            predicate="DEPENDS_ON",
            subject_key="service:payments-api",
            object_key="service:ledger-staging",
            fact="payments uses staging ledger",
            environment="staging",
            subgraph="infra_topology",
        )
    )
    store.add(
        ClaimRow(
            pot_id="p",
            predicate="DEPENDS_ON",
            subject_key="service:payments-api",
            object_key="service:ledger-prod",
            fact="payments uses production ledger",
            environment="prod",
            subgraph="infra_topology",
        )
    )

    env = service.read(
        GraphReadRequest(
            pot_id="p",
            subgraph="infra_topology",
            view="service_neighborhood",
            scope={"service": "payments-api"},
            environment="staging",
            depth=1,
        )
    )

    rels = [rel for item in env.items for rel in item["relations"]]
    assert {rel["environment"] for rel in rels} == {"staging"}
    assert {rel["related_key"] for rel in rels} == {
        "service:ledger-staging",
        "service:payments-api",
    }


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

    web = next(
        entity for entity in result["entities"] if entity["key"] == "service:web"
    )
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


def test_search_entities_filters_by_subgraph_truth_and_scope(service) -> None:
    store = service.backend.claim_query
    store.add(
        ClaimRow(
            pot_id="p",
            predicate="PROVIDES",
            subject_key="repo:github.com/acme/widgets",
            object_key="feature:payments",
            fact="widgets provides payments",
            subgraph="features",
            truth="agent_claim",
            properties={"semantic_similarity": 0.9},
        )
    )
    store.add(
        ClaimRow(
            pot_id="p",
            predicate="DEPENDS_ON",
            subject_key="service:payments",
            object_key="service:ledger",
            fact="payments depends on ledger",
            subgraph="infra_topology",
            truth="source_observation",
            properties={"semantic_similarity": 0.95},
        )
    )

    result = service.search_entities(
        GraphEntitySearchRequest(
            pot_id="p",
            query="payments",
            type="Feature",
            subgraph="features",
            truth="agent_claim",
            scope={"repo": "github.com/acme/widgets"},
        )
    ).to_dict()

    assert [entity["key"] for entity in result["entities"]] == ["feature:payments"]


def test_search_entities_omits_supporting_claims_by_default(service) -> None:
    store = service.backend.claim_query
    store.add(
        ClaimRow(
            pot_id="p",
            predicate="PROVIDES",
            subject_key="repo:github.com/acme/widgets",
            object_key="feature:payments",
            fact="widgets provides payments",
            subgraph="features",
            properties={"semantic_similarity": 0.9},
        )
    )

    result = service.search_entities(
        GraphEntitySearchRequest(pot_id="p", query="payments")
    ).to_dict()

    assert result["entities"]
    assert all(entity["supporting_claims"] == [] for entity in result["entities"])


def test_search_entities_can_include_bounded_supporting_claims(service) -> None:
    store = service.backend.claim_query
    store.add(
        ClaimRow(
            pot_id="p",
            predicate="PROVIDES",
            subject_key="repo:github.com/acme/widgets",
            object_key="feature:payments",
            fact="widgets provides payments",
            subgraph="features",
            properties={"semantic_similarity": 0.9},
        )
    )

    result = service.search_entities(
        GraphEntitySearchRequest(
            pot_id="p",
            query="payments",
            supporting_claims=1,
        )
    ).to_dict()

    by_key = {entity["key"]: entity for entity in result["entities"]}
    assert len(by_key["feature:payments"]["supporting_claims"]) == 1


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
            subgraph="features",
            view="feature_context",
            scope={"anchor_entity_key": "repo:github.com/potpie-ai/potpie"},
        )
    )

    by_key = {item["entity_key"]: item for item in env.items}
    assert by_key["repo:github.com/potpie-ai/potpie"]["entity_type"] == "Repository"
    assert by_key["feature:context-graph"]["entity_type"] == "Feature"


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
            subgraph="debugging",
            view="prior_occurrences",
            query="ambiguous pot graph read",
            scope={"service": "context-engine"},
            limit=10,
        )
    )

    rels = [rel for item in env.items for rel in item["relations"]]
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
            subgraph="decisions",
            view="active_decisions",
            query="semantic mutations context writes",
            scope={"service": "context-engine"},
            limit=10,
        )
    )

    rels = [rel for item in env.items for rel in item["relations"]]
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

"""Tests for the executable Graph V2 workbench ontology contract."""

from __future__ import annotations

from dataclasses import replace

import pytest

from domain.graph_contract import (
    APPLICABLE_MUTATION_OPS,
    DEFERRED_OPS,
    REVIEW_REQUIRED_OPS,
)
from domain.graph_workbench_ontology import (
    assert_ontology_contract_coherent,
    describe_contract,
    ontology_contract,
    rank_views_for_task,
)

pytestmark = pytest.mark.unit


def test_describe_contract_teaches_backed_view_usage() -> None:
    payload = describe_contract(
        subgraph="debugging",
        view="prior_occurrences",
        include_examples=True,
    )

    assert payload["contract_kind"] == "graph_workbench_ontology"
    assert payload["subgraph"]["name"] == "debugging"
    assert payload["view"]["name"] == "debugging.prior_occurrences"
    assert payload["view"]["backed"] is True
    assert "query" in payload["view"]["supported_filters"]
    assert "REPRODUCES" in payload["view"]["inline_relations"]
    assert payload["view"]["examples"][0]["command"].startswith("potpie graph read")


def test_task_ranking_prioritizes_debugging_workflow_context() -> None:
    ranking = rank_views_for_task("debug staging timeout after deployment")

    ranked_subgraphs = [entry["subgraph"] for entry in ranking]
    assert ranked_subgraphs.index("debugging") < ranked_subgraphs.index("features")
    assert ranked_subgraphs.index("recent_changes") < ranked_subgraphs.index("features")
    assert ranked_subgraphs.index("infra_topology") < ranked_subgraphs.index("features")
    assert ranked_subgraphs.index("decisions") < ranked_subgraphs.index("features")


def test_mutation_policies_match_graph_contract_partitions() -> None:
    policies = {
        policy.operation: policy.availability
        for policy in ontology_contract().mutation_policies
    }

    for op in APPLICABLE_MUTATION_OPS:
        assert policies[op] == "applicable"
    for op in REVIEW_REQUIRED_OPS:
        assert policies[op] == "review_required"
    for op in DEFERRED_OPS:
        assert policies[op] == "deferred"


def test_entity_contract_exposes_patch_and_lifecycle_rules() -> None:
    payload = describe_contract(subgraph="decisions", include_examples=False)
    entities = {item["label"]: item for item in payload["subgraph"]["entity_types"]}

    decision = entities["Decision"]
    assert "summary" in decision["patchable_properties"]
    assert "description" in decision["patchable_properties"]
    assert "accepted" in decision["lifecycle_states"]
    assert decision["lifecycle_transitions"]["proposed"] == ["accepted", "rejected"]
    assert decision["lifecycle_transitions"]["accepted"] == [
        "deprecated",
        "superseded",
    ]


def test_contract_check_rejects_unsupported_view_include() -> None:
    contract = ontology_contract()
    first_subgraph = contract.subgraphs[0]
    bad_view = replace(first_subgraph.views[0], v1_include="not_an_include")
    bad_subgraph = replace(
        first_subgraph,
        views=(bad_view, *first_subgraph.views[1:]),
    )
    bad_contract = replace(
        contract,
        subgraphs=(bad_subgraph, *contract.subgraphs[1:]),
    )

    with pytest.raises(RuntimeError, match="unsupported include"):
        assert_ontology_contract_coherent(bad_contract)


def test_contract_check_rejects_invalid_mutation_policy() -> None:
    contract = ontology_contract()
    bad_policy = replace(contract.mutation_policies[0], operation="unknown_op")
    bad_contract = replace(
        contract,
        mutation_policies=(bad_policy, *contract.mutation_policies[1:]),
    )

    with pytest.raises(RuntimeError, match="unknown op"):
        assert_ontology_contract_coherent(bad_contract)

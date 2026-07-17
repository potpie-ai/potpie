"""Contract tests for ``potpie_context_core.domain.graph_contract`` (Graph V1.5 Step 1).

Locks the versioned contract constants, the truth/op/risk vocabularies, the
key-prefix standardization decision, and the env-qualified edge identity key.
"""

from __future__ import annotations

import pytest

from potpie_context_core.domain.graph_contract import (
    APPLICABLE_MUTATION_OPS,
    DEFERRED_OPS,
    GRAPH_CONTRACT_VERSION,
    KNOWN_MUTATION_OPS,
    ONTOLOGY_VERSION,
    REVIEW_REQUIRED_OPS,
    TRUTH_CLASSES,
    MutationRisk,
    SemanticMutationOp,
    canonical_key_prefix,
    edge_identity_key,
    entity_key_matches_type,
    entity_key_prefix,
    evidence_strength_for_truth,
    is_supported_contract_version,
    make_claim_key,
    normalize_entity_key,
    normalize_key_prefix,
)

pytestmark = pytest.mark.unit


def test_versions() -> None:
    assert GRAPH_CONTRACT_VERSION == "v1.5"
    assert ONTOLOGY_VERSION == "2026-06-graph"


def test_ontology_version_mirrors_contract() -> None:
    from potpie_context_core.domain.ontology import ONTOLOGY_VERSION as ont_version

    assert ont_version == ONTOLOGY_VERSION


def test_supported_contract_version() -> None:
    assert is_supported_contract_version("v1.5")
    assert is_supported_contract_version(None)  # convenience for hand-written payloads
    assert not is_supported_contract_version("v2")
    assert not is_supported_contract_version("v1")


def test_truth_classes_match_plan() -> None:
    assert set(TRUTH_CLASSES) == {
        "authoritative_fact",
        "source_observation",
        "agent_claim",
        "user_decision",
        "preference",
        "timeline_event",
        "quality_finding",
    }


def test_truth_maps_to_ranker_evidence_strength() -> None:
    # Every truth class lands on a strength the ranker actually scores.
    from potpie_context_engine.domain.ranking import _STRENGTH_TO_SCORE

    for truth in TRUTH_CLASSES:
        assert evidence_strength_for_truth(truth) in _STRENGTH_TO_SCORE
    assert evidence_strength_for_truth("authoritative_fact") == "deterministic"
    assert evidence_strength_for_truth("agent_claim") == "stated"
    assert evidence_strength_for_truth(None) == "stated"


def test_op_partitions_are_disjoint_and_complete() -> None:
    applicable = set(APPLICABLE_MUTATION_OPS)
    review = set(REVIEW_REQUIRED_OPS)
    deferred = set(DEFERRED_OPS)
    assert applicable & review == set()
    assert applicable & deferred == set()
    assert review & deferred == set()
    assert applicable | review | deferred == KNOWN_MUTATION_OPS
    assert KNOWN_MUTATION_OPS == {op.value for op in SemanticMutationOp}


def test_review_required_ops_are_empty_after_phase_7b_promotion() -> None:
    assert set(REVIEW_REQUIRED_OPS) == set()


def test_phase_7_correction_ops_are_applicable() -> None:
    assert {
        "patch_entity",
        "transition_state",
        "supersede_claim",
        "merge_duplicate_entities",
    } <= set(APPLICABLE_MUTATION_OPS)
    assert set(DEFERRED_OPS) == set()


def test_risk_tiers() -> None:
    assert {r.value for r in MutationRisk} == {"low", "medium", "high"}


# --- Key prefix standardization --------------------------------------------


def test_key_prefix_normalization_strips_only() -> None:
    assert normalize_key_prefix(" bug_pattern ") == "bug_pattern"
    assert normalize_key_prefix("bug-pattern") == "bug-pattern"
    assert normalize_key_prefix("service") == "service"


def test_normalize_entity_key_preserves_body_hyphens() -> None:
    # Prefixes are exact; the body may still contain hyphens.
    assert normalize_entity_key("service:payments-api") == "service:payments-api"
    assert (
        normalize_entity_key(" bug_pattern:refund-race ") == "bug_pattern:refund-race"
    )
    assert normalize_entity_key("no-colon-here") == "no-colon-here"


def test_entity_key_prefix() -> None:
    assert entity_key_prefix("service:payments-api") == "service"
    assert entity_key_prefix("bug_pattern:x") == "bug_pattern"
    assert entity_key_prefix("bug-pattern:x") == "bug-pattern"
    assert entity_key_prefix("unprefixed") is None


def test_canonical_key_prefix_reads_ontology() -> None:
    assert canonical_key_prefix("Service") == "service"
    assert canonical_key_prefix("BugPattern") == "bug_pattern"
    assert canonical_key_prefix("Nonexistent") is None


def test_entity_key_matches_type_requires_canonical_prefix() -> None:
    assert entity_key_matches_type("service:payments-api", "Service")
    assert entity_key_matches_type("bug_pattern:refund-race", "BugPattern")
    assert not entity_key_matches_type("bug-pattern:refund-race", "BugPattern")
    # Wrong prefix for the type is rejected.
    assert not entity_key_matches_type("repo:foo", "Service")


def test_public_entity_key_prefixes_are_consistent_with_ontology() -> None:
    """Every public entity type's key prefix round-trips through the helpers."""
    from potpie_context_core.domain.ontology import ENTITY_TYPES

    for label, spec in ENTITY_TYPES.items():
        if not spec.public:
            continue
        key = f"{spec.key_prefix}:example"
        assert entity_key_matches_type(key, label), label


# --- Edge identity ----------------------------------------------------------


def test_edge_identity_key_without_environment() -> None:
    assert edge_identity_key("service:a", "depends_on", "service:b") == (
        "service:a",
        "DEPENDS_ON",
        "service:b",
    )


def test_edge_identity_key_folds_in_environment() -> None:
    prod = edge_identity_key("service:a", "DEPENDS_ON", "service:b", environment="prod")
    staging = edge_identity_key(
        "service:a", "DEPENDS_ON", "service:b", environment="staging"
    )
    assert prod == ("service:a", "DEPENDS_ON", "service:b", "prod")
    # An env-qualified edge has a different identity from its cross-env twin —
    # so adding the staging edge cannot supersede the prod edge.
    assert prod != staging


def test_edge_identity_preserves_canonical_prefixes() -> None:
    assert edge_identity_key("bug_pattern:x", "reproduces", "service:y") == (
        "bug_pattern:x",
        "REPRODUCES",
        "service:y",
    )


# --- Claim key --------------------------------------------------------------


def test_make_claim_key_is_deterministic() -> None:
    args = dict(
        pot_id="local/default",
        subgraph="infra_topology",
        subject_key="service:payments-api",
        predicate="DEPENDS_ON",
        object_component="service:ledger-api",
        discriminator="repo:manifest:services/payments/service.yaml",
    )
    assert make_claim_key(**args) == make_claim_key(**args)


def test_make_claim_key_environment_distinguishes() -> None:
    base = dict(
        pot_id="p",
        subgraph="infra_topology",
        subject_key="service:a",
        predicate="DEPENDS_ON",
        object_component="service:b",
        discriminator="ref",
    )
    assert make_claim_key(**base, environment="prod") != make_claim_key(
        **base, environment="staging"
    )


def test_make_claim_key_hashes_value_objects() -> None:
    # A long free-text value object is hashed so keys stay bounded.
    key = make_claim_key(
        pot_id="p",
        subgraph="debugging",
        subject_key="bug_pattern:x",
        predicate="REPRODUCES",
        object_component="a very long free-text value that is not an entity key " * 5,
    )
    assert key.startswith("claim:p:debugging:bug_pattern:x:REPRODUCES:")
    assert len(key) < 200

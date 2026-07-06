"""Unit tests for the unified context-engine ontology.

Covers topology + memory + timeline catalogs, plus validation and
predicate-family helpers. Memory-tier coverage lives in
``test_record_types.py`` and the coherence-invariant module's own checks.
"""

from __future__ import annotations

import pytest

from domain.graph_mutations import EdgeUpsert, EntityUpsert
from domain.ontology import (
    ALLOWED_LIFECYCLE_STATUSES,
    CANONICAL_EDGE_TYPES,
    CANONICAL_LABELS,
    ONTOLOGY_VERSION,
    SINGLETON_EDGE_TYPES,
    allowed_edge_types_between,
    predicate_family_for_episodic_supersede,
    temporal_subject_key_for_edge,
    validate_structural_mutations,
)

pytestmark = pytest.mark.unit


# --- Catalog ---------------------------------------------------------------


def test_version_is_the_unified_version() -> None:
    # Owned by domain.graph_contract (the V1.5 contract home) and mirrored on
    # the ontology module; the graph catalog reports the same string.
    assert ONTOLOGY_VERSION == "2026-06-graph"


def test_catalog_contains_the_seven_topology_entities() -> None:
    for label in (
        "Repository",
        "Service",
        "Environment",
        "DataStore",
        "Cluster",
        "Team",
        "Person",
    ):
        assert label in CANONICAL_LABELS


def test_catalog_contains_memory_tier_anchors() -> None:
    for label in ("Preference", "Policy", "BugPattern", "Fix", "Decision"):
        assert label in CANONICAL_LABELS


def test_catalog_contains_timeline_entities() -> None:
    for label in ("Activity", "Period"):
        assert label in CANONICAL_LABELS


def test_catalog_contains_the_seven_topology_edges() -> None:
    for edge in (
        "DEFINED_IN",
        "DEPLOYED_TO",
        "DEPENDS_ON",
        "USES",
        "HOSTED_ON",
        "OWNED_BY",
        "MEMBER_OF",
    ):
        assert edge in CANONICAL_EDGE_TYPES


def test_catalog_contains_memory_predicates() -> None:
    for edge in (
        "POLICY_APPLIES_TO",
        "REPRODUCES",
        "RESOLVED",
        "ATTEMPTED_FIX_FAILED",
        "VERIFIED",
        "DECIDED",
        "AFFECTS",
    ):
        assert edge in CANONICAL_EDGE_TYPES


def test_catalog_contains_timeline_predicates() -> None:
    for edge in ("TOUCHED", "PERFORMED", "AUTHORED", "MENTIONS", "IN_PERIOD"):
        assert edge in CANONICAL_EDGE_TYPES


def test_catalog_contains_feature_entity() -> None:
    assert "Feature" in CANONICAL_LABELS


def test_catalog_contains_code_asset_entity() -> None:
    assert "CodeAsset" in CANONICAL_LABELS


def test_feature_key_prefix_convention() -> None:
    from domain.ontology import entity_spec

    spec = entity_spec("Feature")
    assert spec is not None
    assert spec.key_prefix == "feature"
    assert spec.project_map_family == "features"


def test_catalog_contains_feature_predicates() -> None:
    for edge in ("PROVIDES", "IMPLEMENTED_IN"):
        assert edge in CANONICAL_EDGE_TYPES


def test_provides_allows_repo_and_service_to_feature() -> None:
    assert "PROVIDES" in allowed_edge_types_between(("Repository",), ("Feature",))
    assert "PROVIDES" in allowed_edge_types_between(("Service",), ("Feature",))
    assert "PROVIDES" not in allowed_edge_types_between(("Feature",), ("Service",))


def test_implemented_in_allows_feature_to_repo_service_codeasset() -> None:
    assert "IMPLEMENTED_IN" in allowed_edge_types_between(("Feature",), ("Repository",))
    assert "IMPLEMENTED_IN" in allowed_edge_types_between(("Feature",), ("Service",))
    assert "IMPLEMENTED_IN" in allowed_edge_types_between(("Feature",), ("CodeAsset",))
    assert "IMPLEMENTED_IN" in allowed_edge_types_between(("Feature",), ("FILE",))
    assert "IMPLEMENTED_IN" not in allowed_edge_types_between(
        ("Repository",), ("Feature",)
    )


def test_catalog_contains_infra_adapter_config_entities() -> None:
    for label in ("Adapter", "ConfigVariable", "DeploymentTarget"):
        assert label in CANONICAL_LABELS


def test_catalog_contains_infra_adapter_config_predicates() -> None:
    for edge in ("USES_ADAPTER", "CONFIGURES", "DEPLOYED_WITH"):
        assert edge in CANONICAL_EDGE_TYPES


def test_validates_feature_provides_plan() -> None:
    entities = [
        EntityUpsert("repo:github.com/acme/shop", ("Entity", "Repository"), {}),
        EntityUpsert("feature:checkout", ("Entity", "Feature"), {}),
    ]
    edges = [
        EdgeUpsert("PROVIDES", "repo:github.com/acme/shop", "feature:checkout", {}),
        EdgeUpsert(
            "IMPLEMENTED_IN", "feature:checkout", "repo:github.com/acme/shop", {}
        ),
    ]
    assert validate_structural_mutations(entities, edges, []) == []


def test_related_to_is_the_generic_fallback_edge() -> None:
    assert "RELATED_TO" in CANONICAL_EDGE_TYPES


def test_allowed_lifecycle_statuses_export() -> None:
    assert "unknown" in ALLOWED_LIFECYCLE_STATUSES
    assert "completed" in ALLOWED_LIFECYCLE_STATUSES


# --- Validation ------------------------------------------------------------


def _valid_topology_plan() -> tuple[list[EntityUpsert], list[EdgeUpsert]]:
    entities = [
        EntityUpsert("service:auth", ("Entity", "Service"), {}),
        EntityUpsert("environment:prod", ("Entity", "Environment"), {}),
        EntityUpsert("team:identity", ("Entity", "Team"), {}),
    ]
    edges = [
        EdgeUpsert("DEPLOYED_TO", "service:auth", "environment:prod", {}),
        EdgeUpsert("OWNED_BY", "service:auth", "team:identity", {}),
    ]
    return entities, edges


def test_validates_canonical_entities_and_edges() -> None:
    entities, edges = _valid_topology_plan()
    assert validate_structural_mutations(entities, edges, []) == []


def test_rejects_unknown_label() -> None:
    errors = validate_structural_mutations(
        [EntityUpsert("x:1", ("Entity", "Bogus"), {})], [], []
    )
    assert any("unknown canonical labels" in e for e in errors)


def test_rejects_unknown_edge_type() -> None:
    errors = validate_structural_mutations(
        [], [EdgeUpsert("NOPE", "service:auth", "environment:prod", {})], []
    )
    assert any("unknown canonical edge type" in e for e in errors)


def test_rejects_invalid_edge_endpoint_labels_when_known_in_batch() -> None:
    entities = [
        EntityUpsert("service:auth", ("Entity", "Service"), {}),
        EntityUpsert("team:identity", ("Entity", "Team"), {}),
    ]
    # DEPLOYED_TO is Service->Environment, not Service->Team.
    edges = [EdgeUpsert("DEPLOYED_TO", "service:auth", "team:identity", {})]
    errors = validate_structural_mutations(entities, edges, [])
    assert any("invalid endpoint labels" in e for e in errors)


def test_allowed_edge_types_between_service_and_environment() -> None:
    assert "DEPLOYED_TO" in allowed_edge_types_between(("Service",), ("Environment",))


# --- Cardinality + predicate families --------------------------------------


def test_owned_by_is_the_only_singleton() -> None:
    assert SINGLETON_EDGE_TYPES == frozenset({"OWNED_BY"})


def test_owner_binding_predicate_family() -> None:
    assert predicate_family_for_episodic_supersede("OWNED_BY") == "owner_binding"
    # owner_binding groups contradictions by the owned subject.
    assert (
        temporal_subject_key_for_edge("OWNED_BY", "service:auth", "team:x")
        == "service:auth"
    )


def test_multi_binding_predicate_families_are_not_exclusive() -> None:
    assert predicate_family_for_episodic_supersede("USES") is None
    assert (
        temporal_subject_key_for_edge("USES", "service:auth", "datastore:redis") is None
    )

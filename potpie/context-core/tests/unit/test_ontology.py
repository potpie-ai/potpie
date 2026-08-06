"""Unit tests for the unified context-engine ontology.

Covers topology + memory + timeline catalogs, plus validation and
predicate-family helpers. Memory-tier coverage lives in
``test_record_types.py`` and the coherence-invariant module's own checks.
"""

from __future__ import annotations

import pytest

from potpie_context_core.graph_mutations import EdgeUpsert, EntityUpsert
from potpie_context_core.identity import (
    IdentityClass,
    IdentitySpec,
    validate_entity_key,
)
from potpie_context_core.ontology import (
    ALLOWED_LIFECYCLE_STATUSES,
    CANONICAL_EDGE_TYPES,
    CANONICAL_LABELS,
    ENTITY_PROPERTY_SIGNATURES,
    FACT_FAMILY_FRESHNESS_TTL_HOURS,
    ONTOLOGY_VERSION,
    SINGLETON_EDGE_TYPES,
    allowed_edge_types_between,
    edge_spec,
    entity_spec,
    fact_family_for_label,
    predicate_family_for_episodic_supersede,
    temporal_subject_key_for_edge,
    validate_structural_mutations,
)

pytestmark = pytest.mark.unit


# --- Catalog ---------------------------------------------------------------


def test_version_is_the_unified_version() -> None:
    # Owned by potpie_context_core.graph_contract (the V1.5 contract home) and mirrored on
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
    from potpie_context_core.ontology import entity_spec

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


# --- Reference material (documents + sections) ------------------------------


def test_document_is_a_public_slug_alias_entity() -> None:
    spec = entity_spec("Document")
    assert spec is not None
    assert spec.public is True
    assert spec.identity_class is IdentityClass.SLUG_ALIAS
    assert spec.key_prefix == "document"
    assert spec.identity_policy == "document:<slug>"


def test_document_section_is_a_public_slug_alias_entity() -> None:
    spec = entity_spec("DocumentSection")
    assert spec is not None
    assert spec.public is True
    assert spec.identity_class is IdentityClass.SLUG_ALIAS
    assert spec.key_prefix == "docsection"
    assert spec.identity_policy == "docsection:<doc>:<section>"


def test_document_and_section_use_their_own_documents_fact_family() -> None:
    # P9: section claims must not share the soft-fail ``evidence`` family with
    # Observation, or a doc corpus inherits that family's freshness / ranking
    # policy and crowds project memory.
    assert fact_family_for_label("Document") == "documents"
    assert fact_family_for_label("DocumentSection") == "documents"
    assert fact_family_for_label("Observation") == "evidence"
    assert "documents" in FACT_FAMILY_FRESHNESS_TTL_HOURS
    assert FACT_FAMILY_FRESHNESS_TTL_HOURS["documents"] == 12 * 7 * 24


def test_document_and_section_keys_validate_against_the_identity_registry() -> None:
    document = IdentitySpec(
        label="Document", klass=IdentityClass.SLUG_ALIAS, key_prefix="document"
    )
    section = IdentitySpec(
        label="DocumentSection",
        klass=IdentityClass.SLUG_ALIAS,
        key_prefix="docsection",
    )
    assert validate_entity_key(document, "document:q3-review")
    assert validate_entity_key(section, "docsection:q3-review:capacity")
    assert not validate_entity_key(section, "docsection:")
    # A 12-hex body is still a legal slug body, which is exactly why the
    # promotion leaves existing ``document:<hash>`` nodes valid.
    assert validate_entity_key(document, "document:a1b2c3d4e5f6")


def test_section_of_and_documents_are_canonical_predicates() -> None:
    for edge in ("SECTION_OF", "DOCUMENTS"):
        assert edge in CANONICAL_EDGE_TYPES


def test_section_of_only_links_a_section_to_a_document() -> None:
    assert "SECTION_OF" in allowed_edge_types_between(
        ("DocumentSection",), ("Document",)
    )
    assert "SECTION_OF" not in allowed_edge_types_between(
        ("Document",), ("DocumentSection",)
    )
    assert "SECTION_OF" not in allowed_edge_types_between(
        ("DocumentSection",), ("Service",)
    )


def test_section_of_rejects_a_bad_endpoint_pair_in_a_plan() -> None:
    entities = [
        EntityUpsert("docsection:q3:capacity", ("Entity", "DocumentSection"), {}),
        EntityUpsert("service:payments-api", ("Entity", "Service"), {}),
    ]
    edges = [
        EdgeUpsert("SECTION_OF", "docsection:q3:capacity", "service:payments-api", {})
    ]
    errors = validate_structural_mutations(entities, edges, [])
    assert any("invalid endpoint labels" in e for e in errors)


def test_documents_accepts_any_target_from_either_source_label() -> None:
    assert "DOCUMENTS" in allowed_edge_types_between(("Document",), ("Service",))
    assert "DOCUMENTS" in allowed_edge_types_between(
        ("DocumentSection",), ("Repository",)
    )
    assert "DOCUMENTS" in allowed_edge_types_between(("Document",), ("BugPattern",))
    # The source is not a wildcard — only a document or one of its sections
    # is reference material.
    assert "DOCUMENTS" not in allowed_edge_types_between(("Service",), ("Document",))


def test_documents_does_not_infer_its_source_label() -> None:
    # Document vs DocumentSection is genuinely ambiguous on this edge, so the
    # classifier must not guess.
    spec = edge_spec("DOCUMENTS")
    assert spec is not None
    assert spec.source_inferred_labels == ()


def test_section_of_infers_both_endpoint_labels() -> None:
    spec = edge_spec("SECTION_OF")
    assert spec is not None
    assert spec.source_inferred_labels == ("DocumentSection",)
    assert spec.target_inferred_labels == ("Document",)


def test_document_structure_is_an_exclusive_predicate_family() -> None:
    assert predicate_family_for_episodic_supersede("SECTION_OF") == "document_structure"
    assert (
        temporal_subject_key_for_edge(
            "SECTION_OF", "docsection:q3:capacity", "document:q3"
        )
        == "docsection:q3:capacity"
    )


def test_reference_material_entities_declare_distinctive_signatures() -> None:
    # One signature property is sufficient for the classifier, so a generic
    # name mislabels unrelated entities. ``source_kind`` is this codebase's
    # provenance word and buys nothing here — the ``document:`` key prefix
    # already classifies a document — so Document declares none at all.
    assert "source_kind" not in ENTITY_PROPERTY_SIGNATURES
    assert ENTITY_PROPERTY_SIGNATURES["chunk_count"] == ("DocumentSection",)


def test_document_declares_no_text_patterns() -> None:
    # ``\bdocument\b`` would drag every note onto the Document label; the
    # write path names documents explicitly instead.
    spec = entity_spec("Document")
    assert spec is not None
    assert spec.text_patterns == ()


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


def test_singleton_predicates_are_owned_by_and_section_of() -> None:
    # A service has one live owner; a section has one parent document. Every
    # other predicate accumulates, so this set stays small on purpose.
    assert SINGLETON_EDGE_TYPES == frozenset({"OWNED_BY", "SECTION_OF"})


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

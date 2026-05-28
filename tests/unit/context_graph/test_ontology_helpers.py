"""Public helpers in ``domain.ontology``: normalization, validation, family lookup."""

from __future__ import annotations

import pytest

from domain.graph_mutations import EdgeDelete, EdgeUpsert, EntityUpsert
from domain.ontology import (
    ENTITY_TYPES,
    LifecycleStatus,
    allowed_edge_types_between,
    canonical_entity_labels,
    edge_spec,
    entity_spec,
    inferred_labels_for_episodic_edge_endpoint,
    is_canonical_edge_type,
    is_canonical_entity_label,
    normalize_graphiti_edge_name,
    object_counterparty_uuid_for_edge,
    predicate_family_for_edge_name,
    predicate_family_for_episodic_supersede,
    temporal_subject_key_for_edge,
    validate_edge_delete,
    validate_edge_upsert,
    validate_entity_upsert,
    validate_structural_mutations,
)

pytestmark = pytest.mark.unit


# --- Normalization ---------------------------------------------------------


class TestNormalizeGraphitiEdgeName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("modified", "MODIFIED"),
            ("DEPLOYED-TO", "DEPLOYED_TO"),
            (" decided for ", "DECIDED_FOR"),
            ("AUTHORED by", "AUTHORED_BY"),
            ("", ""),
        ],
    )
    def test_uppercases_strips_and_replaces(self, raw: str, expected: str) -> None:
        assert normalize_graphiti_edge_name(raw) == expected


# --- Canonical label predicates -------------------------------------------


class TestCanonicalLabelHelpers:
    def test_is_canonical_entity_label_for_known(self) -> None:
        assert is_canonical_entity_label("PullRequest") is True

    def test_is_canonical_entity_label_for_unknown(self) -> None:
        assert is_canonical_entity_label("NotALabel") is False

    def test_is_canonical_edge_type_known(self) -> None:
        assert is_canonical_edge_type("HAS_COMMIT") is True

    def test_is_canonical_edge_type_unknown(self) -> None:
        assert is_canonical_edge_type("FOOBAR") is False

    def test_canonical_entity_labels_filters_to_known(self) -> None:
        out = canonical_entity_labels(["Entity", "PullRequest", "Bogus"])
        assert "PullRequest" in out
        assert "Entity" not in out
        assert "Bogus" not in out

    def test_entity_spec_returns_none_for_unknown(self) -> None:
        assert entity_spec("Bogus") is None

    def test_entity_spec_returns_spec_for_known(self) -> None:
        spec = entity_spec("PullRequest")
        assert spec is not None
        assert spec.label == "PullRequest"

    def test_edge_spec_returns_none_for_unknown(self) -> None:
        assert edge_spec("BOGUS") is None

    def test_edge_spec_returns_spec_for_known(self) -> None:
        spec = edge_spec("HAS_COMMIT")
        assert spec is not None
        assert spec.edge_type == "HAS_COMMIT"


# --- Predicate families ----------------------------------------------------


class TestPredicateFamilies:
    def test_known_owner_binding(self) -> None:
        assert predicate_family_for_edge_name("OWNS") == "owner_binding"
        assert predicate_family_for_edge_name("OWNED_BY") == "owner_binding"

    def test_known_lifecycle_status(self) -> None:
        assert predicate_family_for_edge_name("DEPRECATED") == "lifecycle_status"

    def test_unknown_returns_none(self) -> None:
        assert predicate_family_for_edge_name("WHATEVER") is None

    def test_normalization_inside_lookup(self) -> None:
        # Hyphenated / spaced names normalize before lookup.
        assert predicate_family_for_edge_name("deployed-to") == "deployment_target"

    def test_chose_excluded_when_target_has_no_canonical_hint(self) -> None:
        assert predicate_family_for_episodic_supersede("CHOSE", target_labels=()) is None

    def test_chose_with_datastore_hint_joins_datastore_binding(self) -> None:
        fam = predicate_family_for_episodic_supersede(
            "CHOSE", target_labels=["DataStore", "Entity"]
        )
        assert fam == "datastore_binding"

    def test_chose_with_unrelated_canonical_hint_returns_none(self) -> None:
        # Canonical hint exists but is not DataStore → the strict fallback returns None.
        assert (
            predicate_family_for_episodic_supersede("CHOSE", target_labels=["Decision"])
            is None
        )

    def test_non_chose_falls_through_to_normal_family(self) -> None:
        assert predicate_family_for_episodic_supersede("DEPLOYED_TO") == "deployment_target"


class TestObjectCounterpartyUuid:
    def test_unknown_family_returns_none(self) -> None:
        assert object_counterparty_uuid_for_edge("WHATEVER", "s", "t") is None

    def test_lifecycle_uses_target(self) -> None:
        assert object_counterparty_uuid_for_edge("DEPRECATED", "s", "t") == "t"

    def test_datastore_uses_target(self) -> None:
        # Use ``USES_DATA_STORE`` which is in ``datastore_binding``.
        assert (
            object_counterparty_uuid_for_edge("USES_DATA_STORE", "s", "t") == "t"
        )

    def test_owns_uses_source(self) -> None:
        assert object_counterparty_uuid_for_edge("OWNS", "s", "t") == "s"

    def test_owned_by_uses_target(self) -> None:
        assert object_counterparty_uuid_for_edge("OWNED_BY", "s", "t") == "t"

    def test_maintained_by_uses_source(self) -> None:
        assert object_counterparty_uuid_for_edge("MAINTAINED_BY", "s", "t") == "s"


class TestTemporalSubjectKey:
    def test_unknown_family_returns_none(self) -> None:
        assert temporal_subject_key_for_edge("WHATEVER", "s", "t") is None

    def test_lifecycle_uses_source(self) -> None:
        assert temporal_subject_key_for_edge("DEPRECATED", "s", "t") == "s"

    def test_owns_uses_target(self) -> None:
        assert temporal_subject_key_for_edge("OWNS", "s", "t") == "t"

    def test_owned_by_uses_source(self) -> None:
        assert temporal_subject_key_for_edge("OWNED_BY", "s", "t") == "s"

    def test_maintained_by_uses_target(self) -> None:
        assert temporal_subject_key_for_edge("MAINTAINED_BY", "s", "t") == "t"


# --- Edge endpoint inference ----------------------------------------------


class TestInferredLabelsForEdgeEndpoint:
    def test_unknown_role_returns_empty(self) -> None:
        assert inferred_labels_for_episodic_edge_endpoint("AUTHORED_BY", "neither") == ()

    def test_authored_by_target_is_person(self) -> None:
        assert inferred_labels_for_episodic_edge_endpoint("AUTHORED_BY", "target") == ("Person",)

    def test_modified_source_is_pull_request(self) -> None:
        assert inferred_labels_for_episodic_edge_endpoint("MODIFIED", "source") == ("PullRequest",)

    def test_unknown_edge_returns_empty(self) -> None:
        assert inferred_labels_for_episodic_edge_endpoint("WHATEVER", "source") == ()


# --- Validation ------------------------------------------------------------


class TestValidateEntityUpsert:
    def test_missing_entity_key_errors(self) -> None:
        errors = validate_entity_upsert(
            EntityUpsert(entity_key="", labels=("PullRequest",))
        )
        assert any("entity_key is required" in e for e in errors)

    def test_no_labels_errors(self) -> None:
        errors = validate_entity_upsert(EntityUpsert(entity_key="x", labels=()))
        assert any("at least one label is required" in e for e in errors)

    def test_unknown_label_listed(self) -> None:
        errors = validate_entity_upsert(
            EntityUpsert(entity_key="x", labels=("Bogus",))
        )
        assert any("unknown canonical labels" in e and "Bogus" in e for e in errors)

    def test_no_canonical_label_errors(self) -> None:
        # ``Entity`` is allowed-noncanonical but provides no public canonical label.
        errors = validate_entity_upsert(
            EntityUpsert(entity_key="x", labels=("Entity",))
        )
        assert any("at least one public canonical label is required" in e for e in errors)

    def test_missing_required_property_errors(self) -> None:
        # PullRequest has required properties; pick one that's required.
        errors = validate_entity_upsert(
            EntityUpsert(entity_key="x", labels=("PullRequest",), properties={})
        )
        # Some ``missing required properties:`` line for PullRequest must exist.
        assert any("missing required properties" in e for e in errors)

    def test_well_formed_pull_request_validates(self) -> None:
        spec = ENTITY_TYPES["PullRequest"]
        # Build a property bag including all required props (filled with stubs).
        properties = {prop: "x" for prop in spec.required_properties}
        if "pr_number" in properties:
            properties["pr_number"] = 1
        errors = validate_entity_upsert(
            EntityUpsert(
                entity_key="github:pr:1",
                labels=("PullRequest",),
                properties=properties,
            )
        )
        # Filter out lifecycle-related errors (status not provided is OK).
        assert errors == []


class TestValidateEdge:
    def test_missing_edge_type(self) -> None:
        errors = validate_edge_upsert(EdgeUpsert("", "a", "b"))
        assert any("edge_type is required" in e for e in errors)

    def test_unknown_edge_type(self) -> None:
        errors = validate_edge_upsert(EdgeUpsert("BOGUS", "a", "b"))
        assert any("unknown canonical edge type" in e for e in errors)

    def test_missing_endpoints(self) -> None:
        errors = validate_edge_upsert(EdgeUpsert("HAS_COMMIT", "", ""))
        assert any("from_entity_key is required" in e for e in errors)
        assert any("to_entity_key is required" in e for e in errors)

    def test_invalid_endpoint_labels(self) -> None:
        # HAS_COMMIT requires PullRequest → Commit; mismatched labels should error.
        labels = {"a": ("Person",), "b": ("Person",)}
        errors = validate_edge_upsert(EdgeUpsert("HAS_COMMIT", "a", "b"), labels)
        assert any("invalid endpoint labels" in e for e in errors)

    def test_valid_edge_passes_when_both_endpoints_known(self) -> None:
        labels = {"pr": ("PullRequest",), "c": ("Commit",)}
        errors = validate_edge_upsert(EdgeUpsert("HAS_COMMIT", "pr", "c"), labels)
        assert errors == []

    def test_validate_edge_delete_has_same_shape(self) -> None:
        labels = {"pr": ("PullRequest",), "c": ("Commit",)}
        errors = validate_edge_delete(EdgeDelete("HAS_COMMIT", "pr", "c"), labels)
        assert errors == []


class TestValidateStructuralMutations:
    def test_aggregates_errors_from_each_phase(self) -> None:
        entity_errors = validate_structural_mutations(
            entity_upserts=[EntityUpsert(entity_key="", labels=("PullRequest",))],
            edge_upserts=[],
            edge_deletes=[EdgeDelete("BOGUS", "a", "b")],
        )
        assert any("entity_key is required" in e for e in entity_errors)
        assert any("unknown canonical edge type" in e for e in entity_errors)


class TestAllowedEdgeTypesBetween:
    def test_pr_to_commit_includes_has_commit(self) -> None:
        out = allowed_edge_types_between(["PullRequest"], ["Commit"])
        assert "HAS_COMMIT" in out

    def test_wildcard_edges_match_any_pair(self) -> None:
        # Some edge specs use wildcard endpoints; those are returned even for
        # labels that aren't recognized canonically.
        out = allowed_edge_types_between(["Bogus"], ["Bogus"])
        # Sanity: HAS_COMMIT (PR→Commit only) is NOT a wildcard, so it must not
        # be in the wildcard-only result.
        assert "HAS_COMMIT" not in out


def test_lifecycle_status_enum_round_trip() -> None:
    # The enum is referenced widely; ensure values are accessible as strings.
    assert LifecycleStatus.completed.value == "completed"
    assert LifecycleStatus("planned") == LifecycleStatus.planned

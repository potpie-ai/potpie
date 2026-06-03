"""Heuristics for Graphiti episodic edge-type collapse and lifecycle inference."""

from __future__ import annotations

import pytest

from domain.extraction_edges import (
    GENERIC_ACTION_EDGE_NAME,
    LIFECYCLE_STATUS_VALUES,
    classify_episodic_edge,
    coalesce_lifecycle,
    generic_modified_ratio_before_normalize,
    infer_lifecycle_status,
    is_legitimate_pr_code_modified,
    normalize_relation_name,
    remap_vague_modified,
)

pytestmark = pytest.mark.unit


class TestNormalizeRelationName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("modified", "MODIFIED"),
            ("MODIFIED", "MODIFIED"),
            ("  causes incident  ", "CAUSES_INCIDENT"),
            ("deployed-to", "DEPLOYED_TO"),
            ("Deployed to", "DEPLOYED_TO"),
        ],
    )
    def test_uppercases_and_collapses_separators(self, raw: str, expected: str) -> None:
        assert normalize_relation_name(raw) == expected

    def test_none_safe(self) -> None:
        assert normalize_relation_name(None) == ""  # type: ignore[arg-type]


class TestIsLegitimatePrCodeModified:
    def test_pr_to_code_file_is_legit(self) -> None:
        assert is_legitimate_pr_code_modified(
            ["PullRequest"], ["FILE"]
        ) is True

    def test_pr_to_function_is_legit(self) -> None:
        assert is_legitimate_pr_code_modified(["PullRequest"], ["FUNCTION"]) is True

    def test_pr_to_codeasset_is_legit(self) -> None:
        assert is_legitimate_pr_code_modified(["PullRequest"], ["CodeAsset"]) is True

    def test_non_pr_source_rejected(self) -> None:
        assert is_legitimate_pr_code_modified(["Issue"], ["FILE"]) is False

    def test_pr_to_non_code_target_rejected(self) -> None:
        assert is_legitimate_pr_code_modified(["PullRequest"], ["Decision"]) is False


class TestInferLifecycleStatus:
    @pytest.mark.parametrize(
        "phrase,status",
        [
            ("This service was decommissioned last week", "decommissioned"),
            ("module is being decommissioning", "decommissioned"),
            ("the gateway was torn down", "decommissioned"),
            ("removed from production", "decommissioned"),
            ("this API is deprecated", "deprecated"),
            ("end-of-life next quarter", "deprecated"),
            ("EOL", "deprecated"),
            ("sunset announcement", "deprecated"),
            ("will be migrated next sprint", "planned"),
            ("planned for Q2 2026", "planned"),
            ("on the roadmap", "planned"),
            ("upcoming release", "planned"),
            ("proposed RFC", "proposed"),
            ("we may introduce a new layer", "proposed"),
            ("could add a queue", "proposed"),
            ("currently rolling out the change", "in_progress"),
            ("being migrated to v2", "in_progress"),
            ("rollout is underway", "in_progress"),
            ("merged the PR", "completed"),
            ("has been deployed to production", "completed"),
            ("now uses the new module", "completed"),
        ],
    )
    def test_buckets(self, phrase: str, status: str) -> None:
        assert infer_lifecycle_status(phrase) == status

    def test_empty_is_unknown(self) -> None:
        assert infer_lifecycle_status("") == "unknown"
        assert infer_lifecycle_status("   ") == "unknown"

    def test_no_signal_is_unknown(self) -> None:
        assert infer_lifecycle_status("a b c") == "unknown"

    def test_decommissioned_outranks_deprecated(self) -> None:
        # The earlier check wins when both phrases appear.
        assert infer_lifecycle_status("decommissioned and deprecated") == "decommissioned"


class TestCoalesceLifecycle:
    def test_existing_valid_value_wins(self) -> None:
        assert coalesce_lifecycle("completed", "planned") == "completed"

    def test_existing_unknown_falls_back_to_inferred(self) -> None:
        assert coalesce_lifecycle("unknown", "planned") == "planned"

    def test_existing_invalid_value_falls_back(self) -> None:
        assert coalesce_lifecycle("not-a-real-status", "completed") == "completed"

    def test_existing_none_falls_back(self) -> None:
        assert coalesce_lifecycle(None, "deprecated") == "deprecated"

    def test_lifecycle_values_are_strings(self) -> None:
        # Sanity: the constant must contain the canonical statuses we test for.
        assert "completed" in LIFECYCLE_STATUS_VALUES
        assert "unknown" in LIFECYCLE_STATUS_VALUES


class TestRemapVagueModified:
    @pytest.mark.parametrize(
        "fact,expected",
        [
            ("we migrated the auth flow", "MIGRATED_TO"),
            ("decommissioned the gateway", "DECOMMISSIONED"),
            ("torn down old store", "DECOMMISSIONED"),
            ("shut down legacy service", "DECOMMISSIONED"),
            ("deprecated the v1 endpoint", "DEPRECATED"),
            ("reached end of life last quarter", "DEPRECATED"),
            ("the module will be split", "PLANNED"),
            ("on the roadmap for next quarter", "PLANNED"),
            ("auth was added as a service", "DELIVERED"),
            ("introduced new model", "DELIVERED"),
            ("rolled out feature flag", "DELIVERED"),
            ("merged the long-awaited PR", "DELIVERED"),
            ("replaces the previous repo", "REPLACES"),
            ("used instead of older client", "REPLACES"),
            ("depends on graphql-core", "DEPENDS_ON"),
            ("relies on redis", "DEPENDS_ON"),
            ("deployed to production env", "DEPLOYED_TO"),
            ("deployed to staging environment", "DEPLOYED_TO"),
            ("the bug caused an outage", "CAUSED"),
            ("led to data loss", "CAUSED"),
            ("added telemetry around the path", "ADDED_TO"),
            ("added instrumentation everywhere", "ADDED_TO"),
        ],
    )
    def test_maps_specific_verbs(self, fact: str, expected: str) -> None:
        assert remap_vague_modified(fact) == expected

    def test_unknown_falls_back_to_generic(self) -> None:
        assert remap_vague_modified("just some random sentence") == GENERIC_ACTION_EDGE_NAME

    def test_empty_returns_generic(self) -> None:
        assert remap_vague_modified("") == GENERIC_ACTION_EDGE_NAME


class TestClassifyEpisodicEdge:
    def test_legitimate_pr_modified_keeps_modified(self) -> None:
        name, lifecycle = classify_episodic_edge(
            "MODIFIED",
            "PR shipped feature",
            ["PullRequest"],
            ["FILE"],
            allowed_normalized_names=frozenset({"MODIFIED"}),
        )
        assert name == "MODIFIED"
        # ``shipped`` triggers ``completed``.
        assert lifecycle == "completed"

    def test_vague_modified_remapped_when_allowed(self) -> None:
        name, lifecycle = classify_episodic_edge(
            "MODIFIED",
            "decommissioned the gateway",
            ["Service"],
            ["Service"],
            allowed_normalized_names=frozenset({"DECOMMISSIONED"}),
        )
        assert name == "DECOMMISSIONED"
        assert lifecycle == "decommissioned"

    def test_vague_modified_falls_back_to_generic_when_replacement_disallowed(self) -> None:
        name, _ = classify_episodic_edge(
            "MODIFIED",
            "decommissioned the gateway",
            ["Service"],
            ["Service"],
            allowed_normalized_names=frozenset(),  # nothing allowed
        )
        assert name == GENERIC_ACTION_EDGE_NAME

    def test_already_allowed_relation_passes_through(self) -> None:
        name, _ = classify_episodic_edge(
            "AUTHORED_BY",
            "Alice authored the PR",
            ["PullRequest"],
            ["Person"],
            allowed_normalized_names=frozenset({"AUTHORED_BY"}),
        )
        assert name == "AUTHORED_BY"

    def test_existing_lifecycle_preserved_when_valid(self) -> None:
        _, lifecycle = classify_episodic_edge(
            "AUTHORED_BY",
            "merged",  # would infer ``completed``
            ["PullRequest"],
            ["Person"],
            allowed_normalized_names=frozenset({"AUTHORED_BY"}),
            existing_lifecycle="planned",
        )
        assert lifecycle == "planned"

    def test_unknown_existing_lifecycle_falls_back_to_inferred(self) -> None:
        _, lifecycle = classify_episodic_edge(
            "AUTHORED_BY",
            "merged",
            ["PullRequest"],
            ["Person"],
            allowed_normalized_names=frozenset({"AUTHORED_BY"}),
            existing_lifecycle="unknown",
        )
        assert lifecycle == "completed"

    def test_non_string_existing_lifecycle_treated_as_none(self) -> None:
        _, lifecycle = classify_episodic_edge(
            "AUTHORED_BY",
            "merged",
            ["PullRequest"],
            ["Person"],
            allowed_normalized_names=frozenset({"AUTHORED_BY"}),
            existing_lifecycle=42,  # type: ignore[arg-type]
        )
        assert lifecycle == "completed"


class _FakeEdge:
    def __init__(self, name: str, source_uuid: str, target_uuid: str) -> None:
        self.name = name
        self.source_node_uuid = source_uuid
        self.target_node_uuid = target_uuid


class TestGenericModifiedRatio:
    def test_empty_edges_yields_zero(self) -> None:
        assert generic_modified_ratio_before_normalize([], {}) == 0.0

    def test_legit_pr_modified_excluded(self) -> None:
        edges = [
            _FakeEdge("MODIFIED", "u1", "u2"),
            _FakeEdge("DEPLOYED_TO", "u3", "u4"),
        ]
        labels = {
            "u1": ("PullRequest",),
            "u2": ("FILE",),
        }
        assert generic_modified_ratio_before_normalize(edges, labels) == 0.0

    def test_vague_modified_counted(self) -> None:
        edges = [
            _FakeEdge("MODIFIED", "u1", "u2"),
            _FakeEdge("MODIFIED", "u3", "u4"),  # legit PR→file
            _FakeEdge("AUTHORED_BY", "u5", "u6"),
        ]
        labels = {
            "u1": ("Service",),
            "u2": ("Service",),
            "u3": ("PullRequest",),
            "u4": ("FILE",),
        }
        # 1 vague out of 3 edges → 1/3.
        ratio = generic_modified_ratio_before_normalize(edges, labels)
        assert ratio == pytest.approx(1 / 3)

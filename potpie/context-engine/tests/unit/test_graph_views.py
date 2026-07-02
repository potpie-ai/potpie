"""Tests for the V2-style read view map (Graph V1.5 Step 2)."""

from __future__ import annotations

import pytest

from domain.agent_context_port import READER_BACKED_INCLUDES
from domain.graph_views import (
    GRAPH_VIEWS,
    INCLUDE_TO_VIEW,
    UnknownGraphViewError,
    backed_views,
    include_guess_guidance,
    view_for_include,
    view_spec,
    views_for_catalog,
)

pytestmark = pytest.mark.unit


def test_initial_view_map_present() -> None:
    expected = {
        "debugging.prior_occurrences",
        "recent_changes.timeline",
        "infra_topology.service_neighborhood",
        "features.feature_context",
        "decisions.preferences_for_scope",
        "admin.inspection_slice",
        "decisions.active_decisions",
        "code_topology.ownership_by_path",
        "knowledge.document_context",
    }
    assert expected <= set(GRAPH_VIEWS)


def test_view_routes_to_v1_include() -> None:
    assert view_spec("debugging.prior_occurrences").v1_include == "prior_bugs"
    assert (
        view_spec("decisions.preferences_for_scope").v1_include == "coding_preferences"
    )
    assert (
        view_spec("infra_topology.service_neighborhood").v1_include == "infra_topology"
    )
    assert view_spec("features.feature_context").v1_include == "features"


def test_backed_derived_from_reader_registry() -> None:
    for spec in GRAPH_VIEWS.values():
        assert spec.backed == (spec.v1_include in READER_BACKED_INCLUDES), spec.name


def test_use_case_views_are_backed() -> None:
    assert view_spec("decisions.active_decisions").backed
    assert view_spec("code_topology.ownership_by_path").backed
    assert view_spec("knowledge.document_context").backed
    assert view_spec("debugging.prior_occurrences").backed
    assert view_spec("infra_topology.service_neighborhood").backed
    assert view_spec("features.feature_context").backed


def test_traversal_flag_only_on_neighborhood_views() -> None:
    traversal = {v.name for v in GRAPH_VIEWS.values() if v.traversal}
    assert traversal == {
        "infra_topology.service_neighborhood",
        "features.feature_context",
    }


def test_neighborhood_declares_depth_direction_environment() -> None:
    spec = view_spec("infra_topology.service_neighborhood")
    assert {"depth", "direction", "environment"} <= set(spec.inputs)


def test_neighborhood_documents_environment_filter_rule() -> None:
    entry = next(
        e
        for e in views_for_catalog()
        if e["name"] == "infra_topology.service_neighborhood"
    )
    rule = entry["extra"]["environment_filter"]
    assert rule["default"] == "qualified_only"
    assert rule["include_unqualified_scope_key"] == "include_unqualified_environment"


def test_bugs_view_inlines_fix_relations() -> None:
    spec = view_spec("debugging.prior_occurrences")
    assert "RESOLVED" in spec.inline_relations
    assert "REPRODUCES" in spec.inline_relations


def test_catalog_entries_shape() -> None:
    entries = views_for_catalog()
    assert entries
    for e in entries:
        assert {"name", "v1_include", "backed", "ranking_inputs", "extra"} <= set(e)


def test_backed_views_subset() -> None:
    names = {v.name for v in backed_views()}
    assert "debugging.prior_occurrences" in names
    assert "decisions.active_decisions" in names


def test_include_to_view_is_total_and_one_to_one() -> None:
    # Every view's include family maps back to exactly that view; a duplicate
    # v1_include would make the migration guidance ambiguous.
    assert len(INCLUDE_TO_VIEW) == len(GRAPH_VIEWS)
    for spec in GRAPH_VIEWS.values():
        assert INCLUDE_TO_VIEW[spec.v1_include] == spec.name
    # Totality: every reader-backed include has a view, so `graph status`
    # backed_views and report_status graph_view pointers can never silently
    # drop a backed capability (enforced at import time by
    # _check_views_coherent).
    assert set(READER_BACKED_INCLUDES) <= set(INCLUDE_TO_VIEW)


def test_view_for_include_resolves_family() -> None:
    assert view_for_include("docs").name == "knowledge.document_context"
    assert view_for_include("timeline").name == "recent_changes.timeline"
    assert view_for_include(" prior_bugs ").name == "debugging.prior_occurrences"
    assert view_for_include("nope") is None
    assert view_for_include("") is None


def test_include_guess_guidance_maps_subgraph_family() -> None:
    # The audit's failed guess: `graph read --subgraph docs --view relevant`.
    guidance = include_guess_guidance("docs", "relevant")
    assert guidance is not None
    assert guidance["view"] == "knowledge.document_context"
    assert guidance["matched_include"] == "docs"
    assert guidance["read_command"] == (
        "potpie graph read --subgraph knowledge --view document_context"
    )


def test_include_guess_guidance_maps_view_family() -> None:
    guidance = include_guess_guidance("knowledge", "docs")
    assert guidance is not None
    assert guidance["view"] == "knowledge.document_context"
    assert guidance["matched_include"] == "docs"


def test_include_guess_guidance_matches_unique_view_basename() -> None:
    guidance = include_guess_guidance("docs", "document_context")
    assert guidance is not None
    assert guidance["view"] == "knowledge.document_context"


def test_include_guess_guidance_none_for_unrecognized() -> None:
    assert include_guess_guidance("nope", "nada") is None
    assert include_guess_guidance(None, None) is None


def test_include_guess_guidance_never_redirects_a_valid_subgraph() -> None:
    # 'decisions' and 'features' are both include families AND subgraph names;
    # a near-miss view under the valid subgraph must not be redirected to the
    # include family's view — a confidently-wrong recommended_next_action
    # would send agents to the wrong sibling view.
    assert include_guess_guidance("decisions", "preferences_for_scop") is None
    assert include_guess_guidance("features", "feature_contex") is None


def test_include_guess_guidance_prefers_view_basename_over_include() -> None:
    # 'timeline' is both an include family and a view basename; a valid view
    # under the wrong subgraph is a relocation, not legacy include usage.
    guidance = include_guess_guidance("knowledge", "timeline")
    assert guidance is not None
    assert guidance["view"] == "recent_changes.timeline"
    assert guidance["matched_include"] is None


def test_include_guess_guidance_still_maps_include_in_view_position() -> None:
    guidance = include_guess_guidance("decisions", "decisions")
    assert guidance is not None
    assert guidance["view"] == "decisions.active_decisions"
    assert guidance["matched_include"] == "decisions"


def test_unknown_graph_view_error_carries_guidance() -> None:
    guidance = include_guess_guidance("docs", "relevant")
    err = UnknownGraphViewError(
        "unknown",
        did_you_mean=guidance,
        recommended_next_action=guidance["read_command"],
    )
    assert isinstance(err, ValueError)
    assert err.detail == {"did_you_mean": guidance}
    assert err.recommended_next_action == guidance["read_command"]
    bare = UnknownGraphViewError("unknown")
    assert bare.detail is None
    assert bare.recommended_next_action is None

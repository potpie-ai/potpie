"""Tests for the V2-style read view map (Graph V1.5 Step 2)."""

from __future__ import annotations

import pytest

from potpie.context_engine.domain.agent_context_port import READER_BACKED_INCLUDES
from potpie.context_engine.domain.graph_views import (
    GRAPH_VIEWS,
    backed_views,
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
    assert traversal == {"infra_topology.service_neighborhood", "features.feature_context"}


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

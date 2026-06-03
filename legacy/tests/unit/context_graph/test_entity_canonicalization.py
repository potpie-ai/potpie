"""Entity-key canonicalization and merge of duplicate upserts in a ReconciliationPlan."""

from __future__ import annotations

from datetime import datetime

import pytest

from domain.context_events import EventRef
from domain.entity_canonicalization import (
    canonicalize_reconciliation_plan,
    normalize_entity_key,
)
from domain.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
)
from domain.reconciliation import EpisodeDraft, ReconciliationPlan

pytestmark = pytest.mark.unit


# --- Helpers ---------------------------------------------------------------


def _empty_plan() -> ReconciliationPlan:
    """Build a minimal plan to mutate in place."""
    return ReconciliationPlan(
        event_ref=EventRef(event_id="e1", source_system="github", pot_id="p1"),
        summary="test",
        episodes=[
            EpisodeDraft(
                name="ep",
                episode_body="b",
                source_description="d",
                reference_time=datetime(2026, 4, 27),
            )
        ],
    )


# --- normalize_entity_key --------------------------------------------------


class TestNormalizeEntityKey:
    def test_empty_returns_empty(self) -> None:
        assert normalize_entity_key("") == ""

    def test_whitespace_only_returns_stripped_empty(self) -> None:
        assert normalize_entity_key("   ") == ""

    def test_lowercases_and_strips(self) -> None:
        assert normalize_entity_key("  Agents  ") == "agents"

    def test_collapses_internal_whitespace_to_underscore(self) -> None:
        assert normalize_entity_key("context resolve  service") == "context_resolve_service"

    def test_preserves_scheme_prefix(self) -> None:
        # The whole key is lowercased, including ``github:pr:``.
        assert normalize_entity_key("Github:PR:42") == "github:pr:42"

    def test_idempotent(self) -> None:
        normalized = normalize_entity_key(" Foo  Bar ")
        assert normalize_entity_key(normalized) == normalized

    def test_internal_tabs_collapse(self) -> None:
        assert normalize_entity_key("a\tb\nc") == "a_b_c"


# --- canonicalize_reconciliation_plan: entities ----------------------------


class TestCanonicalizeEntities:
    def test_no_op_when_keys_already_canonical(self) -> None:
        plan = _empty_plan()
        plan.entity_upserts = [EntityUpsert("agents", labels=("Service",))]
        merges = canonicalize_reconciliation_plan(plan)
        assert merges == 0
        assert [e.entity_key for e in plan.entity_upserts] == ["agents"]
        assert plan.warnings == []

    def test_drift_collapses_to_canonical(self) -> None:
        plan = _empty_plan()
        plan.entity_upserts = [
            EntityUpsert(
                entity_key="Agents",
                labels=("Service",),
                properties={"name": "Agents", "owner": "team-a"},
            ),
            EntityUpsert(
                entity_key="agents",
                labels=("Component",),
                properties={"name": "agents-canonical", "tier": "platform"},
            ),
        ]

        merges = canonicalize_reconciliation_plan(plan)

        assert merges == 1
        assert len(plan.entity_upserts) == 1
        merged = plan.entity_upserts[0]
        assert merged.entity_key == "agents"
        # Labels unioned in first-seen order.
        assert merged.labels == ("Service", "Component")
        # First-seen-wins on property values.
        assert merged.properties["name"] == "Agents"
        assert merged.properties["owner"] == "team-a"
        assert merged.properties["tier"] == "platform"
        assert any("canonicalized 1" in w for w in plan.warnings)

    def test_three_way_collision_counts_two_merges(self) -> None:
        plan = _empty_plan()
        plan.entity_upserts = [
            EntityUpsert("Foo", labels=("A",), properties={"x": 1}),
            EntityUpsert("foo", labels=("B",), properties={"x": 2}),
            EntityUpsert(" foo ", labels=("C",), properties={"y": 9}),
        ]
        merges = canonicalize_reconciliation_plan(plan)
        assert merges == 2
        assert len(plan.entity_upserts) == 1
        merged = plan.entity_upserts[0]
        assert merged.labels == ("A", "B", "C")
        # First-seen-wins: ``x`` came in as 1 first.
        assert merged.properties == {"x": 1, "y": 9}


# --- canonicalize_reconciliation_plan: edges -------------------------------


class TestCanonicalizeEdges:
    def test_edge_endpoints_rewritten_to_canonical(self) -> None:
        plan = _empty_plan()
        plan.entity_upserts = [
            EntityUpsert("Foo", labels=("A",)),
            EntityUpsert("Bar", labels=("B",)),
        ]
        plan.edge_upserts = [EdgeUpsert("DEPENDS_ON", "Foo", "Bar")]
        canonicalize_reconciliation_plan(plan)
        edge = plan.edge_upserts[0]
        assert edge.from_entity_key == "foo"
        assert edge.to_entity_key == "bar"

    def test_self_loop_after_canonicalization_dropped(self) -> None:
        plan = _empty_plan()
        plan.entity_upserts = [EntityUpsert("Foo", labels=("A",))]
        plan.edge_upserts = [EdgeUpsert("DEPENDS_ON", "Foo", "foo")]
        canonicalize_reconciliation_plan(plan)
        assert plan.edge_upserts == []

    def test_duplicate_edge_after_canonicalization_deduped(self) -> None:
        plan = _empty_plan()
        plan.entity_upserts = [
            EntityUpsert("Foo", labels=("A",)),
            EntityUpsert("Bar", labels=("B",)),
        ]
        plan.edge_upserts = [
            EdgeUpsert("DEPENDS_ON", "Foo", "Bar"),
            EdgeUpsert("DEPENDS_ON", "foo", "bar"),
        ]
        canonicalize_reconciliation_plan(plan)
        assert len(plan.edge_upserts) == 1

    def test_edge_endpoint_only_keys_rewritten(self) -> None:
        # Edge endpoints can reference keys that aren't declared as entities;
        # they should still get normalized.
        plan = _empty_plan()
        plan.edge_upserts = [EdgeUpsert("RELATES_TO", "Alpha", "Beta")]
        canonicalize_reconciliation_plan(plan)
        edge = plan.edge_upserts[0]
        assert edge.from_entity_key == "alpha"
        assert edge.to_entity_key == "beta"

    def test_edge_deletes_canonicalized_and_self_loops_dropped(self) -> None:
        plan = _empty_plan()
        plan.entity_upserts = [EntityUpsert("Foo", labels=("A",))]
        plan.edge_deletes = [
            EdgeDelete("LINK", "Foo", "Bar"),
            EdgeDelete("LINK", "Foo", "foo"),  # collapses to self-loop after norm
        ]
        canonicalize_reconciliation_plan(plan)
        assert len(plan.edge_deletes) == 1
        assert plan.edge_deletes[0].from_entity_key == "foo"
        assert plan.edge_deletes[0].to_entity_key == "bar"


# --- canonicalize_reconciliation_plan: invalidations -----------------------


class TestCanonicalizeInvalidations:
    def test_target_entity_canonicalized(self) -> None:
        plan = _empty_plan()
        plan.entity_upserts = [EntityUpsert("Foo", labels=("A",))]
        plan.invalidations = [
            InvalidationOp(
                target_entity_key="Foo",
                target_edge=None,
                reason="superseded",
                superseded_by_key="NewFoo",
            )
        ]
        canonicalize_reconciliation_plan(plan)
        inv = plan.invalidations[0]
        assert inv.target_entity_key == "foo"
        assert inv.superseded_by_key == "newfoo"

    def test_target_edge_canonicalized(self) -> None:
        plan = _empty_plan()
        plan.invalidations = [
            InvalidationOp(
                target_entity_key=None,
                target_edge=("LINK", "Alpha", "Beta"),
                reason="x",
            )
        ]
        canonicalize_reconciliation_plan(plan)
        assert plan.invalidations[0].target_edge == ("LINK", "alpha", "beta")

    def test_self_loop_target_edge_dropped(self) -> None:
        plan = _empty_plan()
        plan.invalidations = [
            InvalidationOp(
                target_entity_key=None,
                target_edge=("LINK", "Foo", "foo"),
                reason="x",
            )
        ]
        canonicalize_reconciliation_plan(plan)
        assert plan.invalidations == []

    def test_invalidation_without_targets_kept(self) -> None:
        plan = _empty_plan()
        plan.invalidations = [
            InvalidationOp(
                target_entity_key=None,
                target_edge=None,
                reason="generic note",
            )
        ]
        canonicalize_reconciliation_plan(plan)
        assert len(plan.invalidations) == 1

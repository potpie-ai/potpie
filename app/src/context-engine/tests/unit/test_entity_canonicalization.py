"""Entity key normalization and duplicate merge before reconciliation apply.

Covers the canonicalization step added to close the ``Agents`` / ``agents`` and
``context_resolve`` duplicate-drift gap (docs/context-graph/implementation-next-steps.md
#3, reviewed 2026-04-22).
"""

from __future__ import annotations

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
from domain.reconciliation import ReconciliationPlan

pytestmark = pytest.mark.unit


def _plan(**kwargs) -> ReconciliationPlan:
    return ReconciliationPlan(
        event_ref=EventRef(event_id="evt-1", pot_id="pot-1", source_system="test"),
        summary="test",
        episodes=[],
        **kwargs,
    )


# --- normalize_entity_key --------------------------------------------------------


def test_normalize_lowercases_and_trims() -> None:
    assert normalize_entity_key(" Agents ") == "agents"


def test_normalize_collapses_internal_whitespace() -> None:
    assert normalize_entity_key("context  resolve\ttool") == "context_resolve_tool"


def test_normalize_passthrough_canonical_scheme_keys() -> None:
    assert normalize_entity_key("github:pr:owner/repo:42") == "github:pr:owner/repo:42"


def test_normalize_empty_and_whitespace() -> None:
    assert normalize_entity_key("") == ""
    assert normalize_entity_key("   ") == ""


# --- canonicalize_reconciliation_plan --------------------------------------------


def test_collapses_case_variant_duplicates_and_unions_labels() -> None:
    plan = _plan(
        entity_upserts=[
            EntityUpsert("Agents", ("Entity", "Feature"), {"name": "Agents"}),
            EntityUpsert("agents", ("Entity", "Component"), {"summary": "agent runtime"}),
        ]
    )
    merges = canonicalize_reconciliation_plan(plan)

    assert merges == 1
    assert len(plan.entity_upserts) == 1
    ent = plan.entity_upserts[0]
    assert ent.entity_key == "agents"
    assert set(ent.labels) == {"Entity", "Feature", "Component"}
    # first-seen wins for scalars
    assert ent.properties["name"] == "Agents"
    assert ent.properties["summary"] == "agent runtime"
    assert any("canonicalized 1" in w for w in plan.warnings)


def test_rewrites_edges_and_drops_self_loops_after_merge() -> None:
    plan = _plan(
        entity_upserts=[
            EntityUpsert("Agents", ("Entity",)),
            EntityUpsert("agents", ("Entity",)),
            EntityUpsert("context_resolve", ("Entity",)),
        ],
        edge_upserts=[
            EdgeUpsert("RELATED_TO", "Agents", "agents"),  # becomes self-loop -> drop
            EdgeUpsert("RELATED_TO", "Agents", "context_resolve"),
            EdgeUpsert("RELATED_TO", "agents", "context_resolve"),  # duplicate after merge
        ],
    )
    canonicalize_reconciliation_plan(plan)

    assert len(plan.entity_upserts) == 2
    assert {e.entity_key for e in plan.entity_upserts} == {"agents", "context_resolve"}
    assert len(plan.edge_upserts) == 1
    edge = plan.edge_upserts[0]
    assert (edge.from_entity_key, edge.to_entity_key) == ("agents", "context_resolve")


def test_rewrites_edge_endpoints_not_declared_as_entities() -> None:
    plan = _plan(
        edge_upserts=[EdgeUpsert("RELATED_TO", " Agents ", "context  resolve")],
    )
    canonicalize_reconciliation_plan(plan)

    assert len(plan.edge_upserts) == 1
    edge = plan.edge_upserts[0]
    assert edge.from_entity_key == "agents"
    assert edge.to_entity_key == "context_resolve"


def test_rewrites_invalidation_targets() -> None:
    plan = _plan(
        invalidations=[
            InvalidationOp(
                target_entity_key="Agents",
                target_edge=None,
                reason="obsolete",
                superseded_by_key="agents_v2",
            ),
            InvalidationOp(
                target_entity_key=None,
                target_edge=("RELATED_TO", "Agents", "X"),
                reason="self",
            ),
            InvalidationOp(
                target_entity_key=None,
                target_edge=("RELATED_TO", "Agents", "agents"),
                reason="collapses_to_self_loop",
            ),
        ]
    )
    canonicalize_reconciliation_plan(plan)

    assert len(plan.invalidations) == 2
    assert plan.invalidations[0].target_entity_key == "agents"
    assert plan.invalidations[0].superseded_by_key == "agents_v2"
    assert plan.invalidations[1].target_edge == ("RELATED_TO", "agents", "x")


def test_canonical_scheme_keys_pass_through_unchanged() -> None:
    plan = _plan(
        entity_upserts=[
            EntityUpsert(
                "github:pr:owner/repo:42", ("Entity", "PullRequest"), {"pr_number": 42}
            ),
            EntityUpsert("github:user:alice", ("Entity", "Person")),
        ]
    )
    merges = canonicalize_reconciliation_plan(plan)

    assert merges == 0
    assert {e.entity_key for e in plan.entity_upserts} == {
        "github:pr:owner/repo:42",
        "github:user:alice",
    }
    assert plan.warnings == []


def test_edge_dedupe_preserves_first_properties() -> None:
    plan = _plan(
        entity_upserts=[
            EntityUpsert("a", ("Entity",)),
            EntityUpsert("b", ("Entity",)),
        ],
        edge_upserts=[
            EdgeUpsert("RELATED_TO", "A", "B", {"confidence": 0.9}),
            EdgeUpsert("RELATED_TO", "a", "b", {"confidence": 0.5}),
        ],
    )
    canonicalize_reconciliation_plan(plan)

    assert len(plan.edge_upserts) == 1
    # last-write-wins on exact key collisions matches pre-existing edge dedupe
    assert plan.edge_upserts[0].properties == {"confidence": 0.5}


def test_no_warning_when_nothing_merged() -> None:
    plan = _plan(entity_upserts=[EntityUpsert("a", ("Entity",))])
    merges = canonicalize_reconciliation_plan(plan)
    assert merges == 0
    assert plan.warnings == []


def test_edge_deletes_rewritten_and_self_loops_dropped() -> None:
    plan = _plan(
        edge_deletes=[
            EdgeDelete("RELATED_TO", "Agents", "agents"),  # self-loop post-normalize
            EdgeDelete("RELATED_TO", "Agents", "X"),
        ]
    )
    canonicalize_reconciliation_plan(plan)

    assert len(plan.edge_deletes) == 1
    assert plan.edge_deletes[0].from_entity_key == "agents"
    assert plan.edge_deletes[0].to_entity_key == "x"

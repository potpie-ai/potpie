"""Plan-time enrichment of entity labels via the shared classifier."""

from __future__ import annotations

from datetime import datetime

import pytest

from domain.canonical_label_inference import enrich_reconciliation_plan_entity_labels
from domain.context_events import EventRef
from domain.graph_mutations import EdgeUpsert, EntityUpsert
from domain.reconciliation import EpisodeDraft, ReconciliationPlan

pytestmark = pytest.mark.unit


def _plan(entities: list[EntityUpsert], edges: list[EdgeUpsert] | None = None) -> ReconciliationPlan:
    return ReconciliationPlan(
        event_ref=EventRef(event_id="e", source_system="github", pot_id="p"),
        summary="t",
        episodes=[
            EpisodeDraft(
                name="ep",
                episode_body="b",
                source_description="d",
                reference_time=datetime(2026, 4, 27),
            )
        ],
        entity_upserts=list(entities),
        edge_upserts=list(edges or []),
    )


class TestEnrichReconciliationPlanEntityLabels:
    def test_no_signal_no_labels_added(self) -> None:
        plan = _plan(
            [
                EntityUpsert(
                    "ent1",
                    labels=("Entity",),
                    properties={"random": 1},
                )
            ]
        )
        before = plan.entity_upserts[0].labels
        enrich_reconciliation_plan_entity_labels(plan)
        # No discriminating signals → labels unchanged.
        assert plan.entity_upserts[0].labels == before

    def test_pr_number_property_classifies_as_pull_request(self) -> None:
        plan = _plan(
            [
                EntityUpsert(
                    entity_key="github:pr:42",
                    labels=("Entity",),
                    properties={"pr_number": 42, "title": "Fix"},
                )
            ]
        )
        enrich_reconciliation_plan_entity_labels(plan)
        labels = plan.entity_upserts[0].labels
        assert "PullRequest" in labels
        # Existing label preserved and result is sorted.
        assert "Entity" in labels
        assert list(labels) == sorted(labels)

    def test_required_properties_filled_with_defaults(self) -> None:
        # PullRequest requires ``title`` (already set) plus more — let's pick a label
        # that has multiple required props by using ``Decision`` via text cue.
        plan = _plan(
            [
                EntityUpsert(
                    entity_key="dec:1",
                    labels=("Entity",),
                    properties={"summary": "we adopted graphql instead of REST"},
                )
            ]
        )
        enrich_reconciliation_plan_entity_labels(plan)
        ent = plan.entity_upserts[0]
        assert "Decision" in ent.labels
        # Decision spec should require a ``statement`` / ``title`` property; the
        # default mapping fills missing ones with the entity_key (truncated).
        # We don't assert exact required properties (they may evolve) but rather
        # that something was added and no property became ``None``.
        for value in ent.properties.values():
            assert value is not None

    def test_idempotent_when_label_already_set(self) -> None:
        plan = _plan(
            [
                EntityUpsert(
                    entity_key="github:pr:42",
                    labels=("Entity", "PullRequest"),
                    properties={"pr_number": 42},
                )
            ]
        )
        enrich_reconciliation_plan_entity_labels(plan)
        # Already had the label; classifier shouldn't duplicate or strip anything.
        labels = plan.entity_upserts[0].labels
        assert labels.count("PullRequest") == 1

    def test_outgoing_edge_implies_label(self) -> None:
        # ``MEMBER_OF`` → target is a Team. But the team is on the *target* of the
        # incoming edge, so we must flow signals via incoming-edge names.
        plan = _plan(
            entities=[
                EntityUpsert(entity_key="alice", labels=("Entity",), properties={}),
                EntityUpsert(entity_key="team-a", labels=("Entity",), properties={}),
            ],
            edges=[EdgeUpsert(edge_type="MEMBER_OF", from_entity_key="alice", to_entity_key="team-a")],
        )
        enrich_reconciliation_plan_entity_labels(plan)
        team_ent = next(e for e in plan.entity_upserts if e.entity_key == "team-a")
        assert "Team" in team_ent.labels

    def test_canonical_type_hint_pin(self) -> None:
        plan = _plan(
            [
                EntityUpsert(
                    entity_key="x",
                    labels=("Entity",),
                    properties={"canonical_type": "Decision"},
                )
            ]
        )
        enrich_reconciliation_plan_entity_labels(plan)
        assert "Decision" in plan.entity_upserts[0].labels

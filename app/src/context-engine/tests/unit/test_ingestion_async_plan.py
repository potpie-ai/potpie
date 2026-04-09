"""Ingestion async: plan split and JSON codec."""

from datetime import datetime, timezone

from domain.context_events import EventRef
from domain.graph_mutations import EntityUpsert
from domain.reconciliation import EpisodeDraft, ReconciliationPlan

from application.use_cases.reconciliation_plan_codec import (
    reconciliation_plan_from_dict,
    reconciliation_plan_to_dict,
)
from application.use_cases.split_reconciliation_plan import split_reconciliation_plan_into_steps


def test_split_moves_structural_to_last_episode_only() -> None:
    ref = EventRef(event_id="e1", source_system="gh", pot_id="p1")
    plan = ReconciliationPlan(
        event_ref=ref,
        summary="s",
        episodes=[
            EpisodeDraft(
                name="a",
                episode_body="ea",
                source_description="src",
                reference_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
            EpisodeDraft(
                name="b",
                episode_body="eb",
                source_description="src",
                reference_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
            ),
        ],
        entity_upserts=[
            EntityUpsert(entity_key="k", labels=("Entity",), properties={}),
        ],
    )
    slices = split_reconciliation_plan_into_steps(plan)
    assert len(slices) == 2
    assert slices[0].episodes[0].name == "a"
    assert slices[0].entity_upserts == []
    assert slices[1].episodes[0].name == "b"
    assert len(slices[1].entity_upserts) == 1


def test_reconciliation_plan_json_roundtrip() -> None:
    ref = EventRef(event_id="e1", source_system="gh", pot_id="p1")
    plan = ReconciliationPlan(
        event_ref=ref,
        summary="s",
        episodes=[
            EpisodeDraft(
                name="n",
                episode_body="b",
                source_description="src",
                reference_time=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
            ),
        ],
        entity_upserts=[
            EntityUpsert(entity_key="k", labels=("L",), properties={"x": 1}),
        ],
    )
    d = reconciliation_plan_to_dict(plan)
    back = reconciliation_plan_from_dict(d)
    assert back.summary == plan.summary
    assert back.episodes[0].name == plan.episodes[0].name
    assert back.entity_upserts[0].entity_key == "k"

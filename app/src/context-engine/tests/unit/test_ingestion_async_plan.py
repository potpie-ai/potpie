"""Ingestion async: plan split and JSON codec."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from domain.context_events import EventRef
from domain.graph_mutations import EntityUpsert
from domain.ingestion_kinds import (
    EPISODE_STEP_APPLIED,
    EPISODE_STEP_PENDING,
    INGESTION_KIND_RAW_EPISODE,
    STEP_KIND_AGENT_PLAN_SLICE,
)
from domain.ports.reconciliation_ledger import ContextEventRow, EpisodeStepRow
from domain.reconciliation import EpisodeDraft, ReconciliationPlan

from application.use_cases.apply_episode_step import apply_episode_step_for_event
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


def test_raw_event_agent_plan_slice_applies_as_plan() -> None:
    ref = EventRef(event_id="e1", source_system="manual", pot_id="p1")
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
    )
    event = ContextEventRow(
        id="e1",
        pot_id="p1",
        provider="github",
        provider_host="github.com",
        repo_name="o/r",
        source_system="manual",
        event_type="raw_episode",
        action="submit",
        source_id="manual_1",
        source_event_id=None,
        payload={},
        occurred_at=None,
        received_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        status="episodes_queued",
        ingestion_kind=INGESTION_KIND_RAW_EPISODE,
    )
    step = EpisodeStepRow(
        id="s1",
        pot_id="p1",
        event_id="e1",
        sequence=1,
        step_kind=STEP_KIND_AGENT_PLAN_SLICE,
        step_json=reconciliation_plan_to_dict(plan),
        status=EPISODE_STEP_PENDING,
        attempt_count=0,
        applied_at=None,
        error=None,
        run_id="run-1",
    )

    ledger = MagicMock()
    ledger.get_event_by_id.return_value = event
    ledger.get_episode_step.return_value = step
    ledger.max_applied_sequence.return_value = None
    ledger.list_episode_steps.return_value = [
        EpisodeStepRow(
            id=step.id,
            pot_id=step.pot_id,
            event_id=step.event_id,
            sequence=step.sequence,
            step_kind=step.step_kind,
            step_json=step.step_json,
            status=EPISODE_STEP_APPLIED,
            attempt_count=step.attempt_count,
            applied_at=step.applied_at,
            error=step.error,
            run_id=step.run_id,
        )
    ]
    episodic = MagicMock()
    episodic.write_episode_drafts.return_value = ["episode-1"]

    result = apply_episode_step_for_event(
        episodic,
        MagicMock(),
        ledger,
        "e1",
        1,
    )

    assert result.ok
    assert result.episode_uuids == ["episode-1"]
    episodic.write_episode_drafts.assert_called_once()
    ledger.record_event_reconciled.assert_called_once_with("e1")

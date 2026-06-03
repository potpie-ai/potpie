"""Sanity checks for ingestion event-store domain models and ports (Phase 1 contracts)."""

from __future__ import annotations

from datetime import datetime, timezone

from domain.ingestion_event_models import (
    CreateIngestionEventParams,
    EpisodeStep,
    EventListFilters,
    EventListPage,
    EventReceipt,
    EventTransition,
    ExecutionResult,
    IngestionEvent,
    IngestionPlan,
    IngestionSubmissionRequest,
    PlanWithSteps,
)
from domain.ports.event_planner import EventPlanner
from domain.ports.event_query_service import EventQueryService
from domain.ports.ingestion_event_queue import IngestionQueue
from domain.ports.ingestion_event_store import IngestionEventStore
from domain.ports.context_graph import ContextGraphPort
from domain.ports.ingestion_submission import IngestionSubmissionService
from domain.ports.step_executor import StepExecutor


def test_ingestion_event_and_receipt_construct() -> None:
    now = datetime.now(timezone.utc)
    ev = IngestionEvent(
        event_id="e1",
        pot_id="p1",
        ingestion_kind="raw_episode",
        source_channel="cli",
        source_system="local",
        event_type="ingest",
        action="submit",
        source_id="src1",
        dedup_key=None,
        status="queued",
        stage="accepted",
        submitted_at=now,
        started_at=None,
        completed_at=None,
        error=None,
        payload={},
        metadata={},
        step_total=1,
        step_done=0,
        step_error=0,
    )
    receipt = EventReceipt(event_id=ev.event_id, status="queued", terminal_event=None)
    assert receipt.event_id == "e1"


def test_plan_with_steps_tuple() -> None:
    plan = IngestionPlan(
        plan_id="pl1",
        event_id="e1",
        planner_type="deterministic_raw",
        version=1,
        summary=None,
    )
    step = EpisodeStep(
        step_id="s1",
        event_id="e1",
        pot_id="p1",
        sequence=0,
        kind="raw_episode",
        status="queued",
        input={},
        attempt_count=0,
        result=None,
        error=None,
        queued_at=None,
        started_at=None,
        completed_at=None,
    )
    bundle = PlanWithSteps(plan=plan, steps=(step,))
    assert len(bundle.steps) == 1


def test_create_and_transition_params() -> None:
    c = CreateIngestionEventParams(
        event_id="e1",
        pot_id="p1",
        ingestion_kind="raw_episode",
        source_channel="http",
        source_system="api",
        event_type="ingest",
        action="post",
        source_id="src1",
        dedup_key="k",
        status="queued",
        stage=None,
        payload={},
    )
    assert c.dedup_key == "k"
    t = EventTransition(to_status="processing", to_stage="planning")
    assert t.to_stage == "planning"


def test_submission_request() -> None:
    r = IngestionSubmissionRequest(
        pot_id="p1",
        ingestion_kind="raw_episode",
        source_channel="cli",
        source_system="local",
        event_type="ingest",
        action="run",
        payload={"text": "x"},
        idempotency_key="ik",
    )
    assert r.idempotency_key == "ik"


def test_list_filters_and_page() -> None:
    page = EventListPage(items=(), next_cursor=None)
    f = EventListFilters(statuses=("done",), ingestion_kinds=("raw_episode",))
    assert page.next_cursor is None
    assert f.statuses == ("done",)


def test_execution_result() -> None:
    ex = ExecutionResult(
        step_id="s1",
        success=True,
        episode_ref=None,
        structural_effects={},
        error=None,
    )
    assert ex.success is True


def test_protocols_are_importable() -> None:
    """Structural contracts: import-time load for CI."""
    assert IngestionSubmissionService is not None
    assert IngestionEventStore is not None
    assert IngestionQueue is not None
    assert EventPlanner is not None
    assert StepExecutor is not None
    assert EventQueryService is not None
    assert ContextGraphPort is not None

"""Sanity checks for ingestion event-store domain models and ports."""

from __future__ import annotations

from datetime import datetime, timezone

from potpie_context_engine.domain.ingestion_event_models import (
    CreateIngestionEventParams,
    EventListFilters,
    EventListPage,
    EventReceipt,
    EventTransition,
    IngestionEvent,
    IngestionSubmissionRequest,
)
from potpie_context_engine.domain.ports.context_graph import ContextGraphPort
from potpie_context_engine.domain.ports.event_query_service import EventQueryService
from potpie_context_engine.domain.ports.ingestion_event_store import IngestionEventStore
from potpie_context_engine.domain.ports.ingestion_submission import (
    IngestionSubmissionService,
)


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


def test_protocols_are_importable() -> None:
    """Structural contracts: import-time load for CI."""
    assert IngestionSubmissionService is not None
    assert IngestionEventStore is not None
    assert EventQueryService is not None
    assert ContextGraphPort is not None

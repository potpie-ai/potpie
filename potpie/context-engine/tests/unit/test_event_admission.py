"""Tests for ``admit_event`` (event admission + enqueue)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from potpie_context_engine.application.services.event_admission import admit_event
from potpie_context_engine.application.services.ingestion_submission_service import (
    DefaultIngestionSubmissionService,
)
from potpie_context_engine.bootstrap import sentry_metrics_runtime
from potpie_context_core.domain.actor import Actor
from potpie_context_core.domain.context_events import ContextEvent, EventScope
from potpie_context_engine.domain.ingestion_kinds import INGESTION_KIND_AGENT_RECONCILIATION
from potpie_context_engine.domain.ingestion_event_models import IngestionEvent, IngestionSubmissionRequest
from potpie_context_core.domain.ports.pot_resolution import single_github_repo_pot

pytestmark = pytest.mark.unit


def _event() -> ContextEvent:
    return ContextEvent(
        event_id="evt-1",
        source_system="github",
        event_type="pull_request",
        action="merged",
        pot_id="pot-1",
        provider="github",
        provider_host="github.com",
        repo_name="o/r",
        source_id="pr_42_merged",
        ingestion_kind=INGESTION_KIND_AGENT_RECONCILIATION,
        occurred_at=datetime(2026, 5, 6, 9, 0, tzinfo=timezone.utc),
        received_at=datetime(2026, 5, 6, 9, 0, tzinfo=timezone.utc),
        payload={"title": "Merge X"},
    )


def _scope() -> EventScope:
    return EventScope(
        pot_id="pot-1",
        provider="github",
        provider_host="github.com",
        repo_name="o/r",
    )


def test_inserts_event_and_enqueues_batch() -> None:
    ledger = MagicMock()
    ledger.append_event.return_value = ("evt-1", True)
    batches = MagicMock()
    batches.upsert_open_batch_for_pot.return_value = "batch-abc"
    jobs = MagicMock()

    out = admit_event(ledger, batches, jobs, _scope(), _event())

    assert out.inserted is True
    assert out.event_id == "evt-1"
    assert out.batch_id == "batch-abc"
    ledger.append_event.assert_called_once()
    ledger.mark_event_queued.assert_called_once_with("evt-1")
    batches.upsert_open_batch_for_pot.assert_called_once_with("pot-1", "evt-1")
    jobs.enqueue_batch.assert_called_once_with("batch-abc")


def test_duplicate_event_skips_batch_and_enqueue() -> None:
    """A duplicate idempotent submit must not retrigger work."""
    ledger = MagicMock()
    ledger.append_event.return_value = ("evt-1", False)
    batches = MagicMock()
    jobs = MagicMock()

    out = admit_event(ledger, batches, jobs, _scope(), _event())

    assert out.inserted is False
    assert out.batch_id is None
    batches.upsert_open_batch_for_pot.assert_not_called()
    ledger.mark_event_queued.assert_not_called()
    jobs.enqueue_batch.assert_not_called()


def test_enqueue_failure_does_not_raise() -> None:
    """Broker outage at enqueue time must not break event admission.

    The event + batch are durable in Postgres; the next event for this pot
    will re-enqueue.
    """
    ledger = MagicMock()
    ledger.append_event.return_value = ("evt-1", True)
    batches = MagicMock()
    batches.upsert_open_batch_for_pot.return_value = "batch-abc"
    jobs = MagicMock()
    jobs.enqueue_batch.side_effect = RuntimeError("broker down")

    out = admit_event(ledger, batches, jobs, _scope(), _event())

    assert out.inserted is True
    assert out.batch_id == "batch-abc"
    jobs.enqueue_batch.assert_called_once_with("batch-abc")


def test_submission_service_mirrors_inserted_admission_to_sentry_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric_calls: list[tuple[str, int, dict[str, str]]] = []

    def record_count(
        name: str,
        value: int = 1,
        *,
        attributes: dict[str, str] | None = None,
    ) -> None:
        metric_calls.append((name, value, {} if attributes is None else attributes))

    monkeypatch.setattr(sentry_metrics_runtime, "count", record_count)
    service = _submission_service(inserted=True)

    receipt = service.submit(_submission_request())

    assert receipt.event_id == "evt-1"
    assert receipt.status == "queued"
    assert metric_calls == [
        (
            "ce.ingest.events_total",
            1,
            {"result": "inserted"},
        )
    ]


def test_submission_service_mirrors_duplicate_admission_to_sentry_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric_calls: list[tuple[str, int, dict[str, str]]] = []

    def record_count(
        name: str,
        value: int = 1,
        *,
        attributes: dict[str, str] | None = None,
    ) -> None:
        metric_calls.append((name, value, {} if attributes is None else attributes))

    monkeypatch.setattr(sentry_metrics_runtime, "count", record_count)
    service = _submission_service(inserted=False)

    receipt = service.submit(_submission_request())

    assert receipt.event_id == "evt-1"
    assert receipt.duplicate is True
    assert metric_calls == [
        (
            "ce.ingest.dedup_total",
            1,
            {"result": "duplicate"},
        )
    ]


def test_submission_service_does_not_export_request_taxonomy_to_sentry_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric_calls: list[tuple[str, int, dict[str, str]]] = []

    def record_count(
        name: str,
        value: int = 1,
        *,
        attributes: dict[str, str] | None = None,
    ) -> None:
        metric_calls.append((name, value, {} if attributes is None else attributes))

    monkeypatch.setattr(sentry_metrics_runtime, "count", record_count)
    service = _submission_service(inserted=True)

    service.submit(
        _submission_request(
            source_system="github:user@example.com",
            event_type="pull_request:secret-token-123",
            action="merged:/src/secrets.py",
        )
    )

    assert metric_calls == [
        (
            "ce.ingest.events_total",
            1,
            {"result": "inserted"},
        )
    ]


def _submission_service(*, inserted: bool) -> DefaultIngestionSubmissionService:
    settings = MagicMock()
    settings.is_enabled.return_value = True
    pots = MagicMock()
    pots.resolve_pot.return_value = single_github_repo_pot("pot-1", "o/r")
    reco = MagicMock()
    reco.append_event.return_value = ("evt-1", inserted)
    batches = MagicMock()
    batches.upsert_open_batch_for_pot.return_value = "batch-abc"
    jobs = MagicMock()
    events = MagicMock()
    events.get_event.return_value = _ingestion_event()
    return DefaultIngestionSubmissionService(
        settings=settings,
        pots=pots,
        reconciliation_agent=MagicMock(),
        reco_ledger=reco,
        events=events,
        batches=batches,
        jobs=jobs,
    )


def _submission_request(
    *,
    source_system: str = "github",
    event_type: str = "pull_request",
    action: str = "merged",
) -> IngestionSubmissionRequest:
    return IngestionSubmissionRequest(
        pot_id="pot-1",
        ingestion_kind=INGESTION_KIND_AGENT_RECONCILIATION,
        source_channel="webhook",
        source_system=source_system,
        event_type=event_type,
        action=action,
        source_id="pr_42_merged",
        repo_name="o/r",
        event_id="evt-1",
        payload={
            "batch_id": "batch-abc",
            "event_id": "evt-1",
            "repo_name": "o/r",
            "source_id": "pr_42_merged",
            "title": "Merge X",
        },
        metadata={"source_id": "pr_42_merged"},
        actor=Actor(
            user_id="user-1",
            surface="webhook",
            client_name="github-webhook",
            auth_method="webhook_signature",
        ),
    )


def _ingestion_event() -> IngestionEvent:
    return IngestionEvent(
        event_id="evt-1",
        pot_id="pot-1",
        ingestion_kind=INGESTION_KIND_AGENT_RECONCILIATION,
        source_channel="webhook",
        source_system="github",
        event_type="pull_request",
        action="merged",
        source_id="pr_42_merged",
        dedup_key=None,
        status="queued",
        stage="accepted",
        submitted_at=datetime(2026, 5, 6, 9, 0, tzinfo=timezone.utc),
        started_at=None,
        completed_at=None,
        error=None,
        payload={},
        repo_name="o/r",
    )

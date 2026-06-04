"""Tests for ``admit_event`` (event admission + enqueue)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from application.services.event_admission import admit_event
from domain.context_events import ContextEvent, EventScope
from domain.ingestion_kinds import INGESTION_KIND_AGENT_RECONCILIATION

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

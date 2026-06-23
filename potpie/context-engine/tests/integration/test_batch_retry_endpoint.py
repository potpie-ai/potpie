"""Behavior tests for POST /pots/{pot_id}/events/batch-retry.

The endpoint is bulk by contract: the whole validated set is marked for
retry, dropped into the pot's open batch, and enqueued in a fixed number
of statements regardless of event count. Coverage:

- happy path: one bulk mark, one bulk batch insert, one enqueue
- dedupes event_ids while preserving order
- rejects empty / >200 event_ids
- 404 on unknown event_id, 400 on cross-pot — both up-front, no partial apply
- a failing bulk step aborts the whole request (no half-applied retry)
- an enqueue failure after the writes committed is tolerated (still 202)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from context_engine.adapters.inbound.http.api.v1.context.router import create_context_router


def _ev(eid: str, pot_id: str = "pot-1") -> SimpleNamespace:
    return SimpleNamespace(event_id=eid, pot_id=pot_id, status="error")


def _build_container(event_rows: dict[str, SimpleNamespace]) -> MagicMock:
    container = MagicMock()

    events_store = MagicMock()
    events_store.get_event.side_effect = lambda eid: event_rows.get(eid)
    container.ingestion_event_store.return_value = events_store

    reco = MagicMock()
    container.reconciliation_ledger.return_value = reco

    batches = MagicMock()
    batches.add_events_to_open_batch_for_pot.return_value = "batch-1"
    container.batch_repository.return_value = batches

    jobs = MagicMock()
    container.jobs = jobs

    # Always-allow policy for these tests.
    decision = SimpleNamespace(allowed=True, status_code=202, reason=None, detail=None)
    container.policy.return_value.authorize.return_value = decision
    return container


def _client(
    container: MagicMock, *, raise_server_exceptions: bool = True
) -> TestClient:
    app = FastAPI()
    app.include_router(
        create_context_router(
            require_auth=lambda: None,
            get_container=lambda: container,
            get_db=lambda: MagicMock(),
            get_db_optional=lambda: MagicMock(),
        ),
        prefix="/api/v1/context",
    )
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def test_happy_path_bulk_marks_whole_set_and_enqueues_once() -> None:
    rows = {"e1": _ev("e1"), "e2": _ev("e2"), "e3": _ev("e3")}
    container = _build_container(rows)
    client = _client(container)

    r = client.post(
        "/api/v1/context/pots/pot-1/events/batch-retry",
        json={"event_ids": ["e1", "e2", "e3"]},
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["count"] == 3
    assert body["pot_id"] == "pot-1"
    assert body["batch_id"] == "batch-1"
    assert body["event_ids"] == ["e1", "e2", "e3"]

    reco = container.reconciliation_ledger.return_value
    batches = container.batch_repository.return_value
    # One bulk call each — no per-event loop.
    reco.mark_events_for_retry.assert_called_once_with(["e1", "e2", "e3"])
    reco.mark_events_queued.assert_called_once_with(["e1", "e2", "e3"])
    batches.add_events_to_open_batch_for_pot.assert_called_once_with(
        "pot-1", ["e1", "e2", "e3"]
    )
    # Single enqueue regardless of event count.
    container.jobs.enqueue_batch.assert_called_once_with("batch-1")


def test_dedupes_event_ids_while_preserving_order() -> None:
    rows = {"a": _ev("a"), "b": _ev("b")}
    container = _build_container(rows)
    client = _client(container)
    r = client.post(
        "/api/v1/context/pots/pot-1/events/batch-retry",
        json={"event_ids": ["a", "b", "a"]},
    )
    assert r.status_code == 202
    body = r.json()
    assert body["event_ids"] == ["a", "b"]
    assert body["count"] == 2

    reco = container.reconciliation_ledger.return_value
    batches = container.batch_repository.return_value
    reco.mark_events_for_retry.assert_called_once_with(["a", "b"])
    batches.add_events_to_open_batch_for_pot.assert_called_once_with(
        "pot-1", ["a", "b"]
    )


def test_empty_event_ids_is_rejected() -> None:
    container = _build_container({})
    client = _client(container)
    # min_length=1 → 422 from pydantic; the explicit guard would 400. We
    # only assert the error is structured rather than a 500.
    r = client.post(
        "/api/v1/context/pots/pot-1/events/batch-retry",
        json={"event_ids": []},
    )
    assert r.status_code in (400, 422)


def test_too_many_event_ids_is_rejected() -> None:
    container = _build_container({})
    client = _client(container)
    r = client.post(
        "/api/v1/context/pots/pot-1/events/batch-retry",
        json={"event_ids": [f"e{i}" for i in range(201)]},
    )
    assert r.status_code in (400, 422)


def test_unknown_event_id_is_404_with_no_partial_apply() -> None:
    rows = {"a": _ev("a")}
    container = _build_container(rows)
    client = _client(container)
    r = client.post(
        "/api/v1/context/pots/pot-1/events/batch-retry",
        json={"event_ids": ["a", "unknown"]},
    )
    assert r.status_code == 404
    # Validation happens up-front: nothing got marked or enqueued.
    reco = container.reconciliation_ledger.return_value
    batches = container.batch_repository.return_value
    assert reco.mark_events_for_retry.call_count == 0
    assert batches.add_events_to_open_batch_for_pot.call_count == 0
    assert container.jobs.enqueue_batch.call_count == 0


def test_cross_pot_event_id_is_400_with_no_partial_apply() -> None:
    rows = {"a": _ev("a"), "x": _ev("x", pot_id="pot-2")}
    container = _build_container(rows)
    client = _client(container)
    r = client.post(
        "/api/v1/context/pots/pot-1/events/batch-retry",
        json={"event_ids": ["a", "x"]},
    )
    assert r.status_code == 400
    reco = container.reconciliation_ledger.return_value
    batches = container.batch_repository.return_value
    assert reco.mark_events_for_retry.call_count == 0
    assert batches.add_events_to_open_batch_for_pot.call_count == 0
    assert container.jobs.enqueue_batch.call_count == 0


def test_failing_bulk_step_aborts_with_no_partial_apply() -> None:
    """The new atomic contract: a failed bulk step propagates as a 5xx and
    nothing downstream of it runs — no events get coalesced or enqueued
    while others are silently dropped."""
    rows = {"a": _ev("a"), "b": _ev("b")}
    container = _build_container(rows)
    reco = container.reconciliation_ledger.return_value
    reco.mark_events_for_retry.side_effect = RuntimeError("ledger down")

    client = _client(container, raise_server_exceptions=False)
    r = client.post(
        "/api/v1/context/pots/pot-1/events/batch-retry",
        json={"event_ids": ["a", "b"]},
    )
    assert r.status_code == 500
    batches = container.batch_repository.return_value
    assert batches.add_events_to_open_batch_for_pot.call_count == 0
    assert container.jobs.enqueue_batch.call_count == 0


def test_enqueue_failure_is_tolerated_and_still_202() -> None:
    """The writes committed; the batch is durable. A broker blip on the
    final enqueue must not fail the request."""
    rows = {"a": _ev("a"), "b": _ev("b")}
    container = _build_container(rows)
    container.jobs.enqueue_batch.side_effect = RuntimeError("broker blip")

    client = _client(container)
    r = client.post(
        "/api/v1/context/pots/pot-1/events/batch-retry",
        json={"event_ids": ["a", "b"]},
    )
    assert r.status_code == 202
    reco = container.reconciliation_ledger.return_value
    batches = container.batch_repository.return_value
    reco.mark_events_for_retry.assert_called_once_with(["a", "b"])
    batches.add_events_to_open_batch_for_pot.assert_called_once_with(
        "pot-1", ["a", "b"]
    )

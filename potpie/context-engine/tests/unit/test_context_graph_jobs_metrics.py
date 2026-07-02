from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from application.use_cases import context_graph_jobs
from bootstrap.ingestion_server import IngestionServerContainer
from domain.ports.observability import NoOpObservability

pytestmark = pytest.mark.unit


class _RecordingObservability(NoOpObservability):
    def __init__(self) -> None:
        self.counter_calls: list[tuple[str, int, dict[str, object]]] = []
        self.histogram_calls: list[tuple[str, float, dict[str, object]]] = []

    def counter(
        self, name: str, value: int = 1, *, attributes: dict[str, object] | None = None
    ) -> None:
        self.counter_calls.append((name, value, dict(attributes or {})))

    def histogram(
        self,
        name: str,
        value: float,
        *,
        attributes: dict[str, object] | None = None,
    ) -> None:
        self.histogram_calls.append((name, value, dict(attributes or {})))


def _batch() -> MagicMock:
    batch = MagicMock()
    batch.id = "batch-1"
    batch.pot_id = "pot-1"
    batch.created_at = datetime(2026, 5, 6, 9, 0, tzinfo=timezone.utc)
    batch.claimed_at = datetime(2026, 5, 6, 9, 0, 2, tzinfo=timezone.utc)
    return batch


def _container(
    *,
    reconciliation_agent: object | None,
    batch: MagicMock | None,
) -> IngestionServerContainer:
    container = MagicMock()
    container.reconciliation_agent = reconciliation_agent
    repo = MagicMock()
    repo.claim_batch_by_id.return_value = batch
    repo.list_events_for_batch.return_value = []
    container.batch_repository.return_value = repo
    container.reconciliation_ledger.return_value = MagicMock()
    container.agent_checkpoint_store.return_value = MagicMock()
    container.pots = MagicMock()
    container.policy.return_value = None
    container.event_stream_publisher = MagicMock()
    container.agent_execution_log.return_value = MagicMock()
    return container


def test_skipped_jobs_emit_finished_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs = _RecordingObservability()
    monkeypatch.setattr(context_graph_jobs, "get_observability", lambda: obs)

    no_agent = _container(reconciliation_agent=None, batch=_batch())
    not_pending = _container(reconciliation_agent=object(), batch=None)

    context_graph_jobs.handle_process_batch(
        MagicMock(),
        "batch-1",
        build_ingestion_server=lambda _db: no_agent,
    )
    context_graph_jobs.handle_process_batch(
        MagicMock(),
        "batch-1",
        build_ingestion_server=lambda _db: not_pending,
    )

    assert (
        "ce.batch.finished_total",
        1,
        {"result": "skipped_no_reconciliation_agent"},
    ) in obs.counter_calls
    assert (
        "ce.batch.finished_total",
        1,
        {"result": "skipped_not_pending"},
    ) in obs.counter_calls


def test_claimed_job_emits_batch_lifecycle_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs = _RecordingObservability()
    monkeypatch.setattr(context_graph_jobs, "get_observability", lambda: obs)
    monkeypatch.setattr(
        context_graph_jobs,
        "process_batch",
        lambda **_: MagicMock(
            batch_id="batch-1",
            ok=True,
            completed_event_ids=["e1"],
            tool_call_count=2,
            error=None,
        ),
    )

    container = _container(reconciliation_agent=object(), batch=_batch())

    context_graph_jobs.handle_process_batch(
        MagicMock(),
        "batch-1",
        build_ingestion_server=lambda _db: container,
    )

    assert (
        "ce.batch.started_total",
        1,
        {"pot_id": "pot-1", "result": "started"},
    ) in obs.counter_calls
    assert (
        "ce.batch.finished_total",
        1,
        {"pot_id": "pot-1", "result": "ok"},
    ) in obs.counter_calls
    assert (
        "ce.batch.time_in_pending_ms",
        2000.0,
        {"pot_id": "pot-1", "result": "started"},
    ) in obs.histogram_calls

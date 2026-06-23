from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from potpie.context_engine.application.use_cases import context_graph_jobs
from potpie.context_engine.bootstrap.ingestion_server import IngestionServerContainer
from potpie.context_engine.bootstrap import sentry_metrics_runtime

pytestmark = pytest.mark.unit


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


def test_skipped_jobs_mirror_finished_sentry_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts: list[tuple[str, int, dict[str, str]]] = []
    monkeypatch.setattr(
        sentry_metrics_runtime,
        "count",
        lambda name, value=1, *, attributes=None, unit=None: counts.append(
            (name, value, dict(attributes or {}))
        ),
    )

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
    ) in counts
    assert (
        "ce.batch.finished_total",
        1,
        {"result": "skipped_not_pending"},
    ) in counts


def test_claimed_job_mirrors_batch_lifecycle_sentry_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts: list[tuple[str, int, dict[str, str]]] = []
    distributions: list[tuple[str, float, str | None, dict[str, str]]] = []
    monkeypatch.setattr(
        sentry_metrics_runtime,
        "count",
        lambda name, value=1, *, attributes=None, unit=None: counts.append(
            (name, value, dict(attributes or {}))
        ),
    )
    monkeypatch.setattr(
        sentry_metrics_runtime,
        "distribution",
        lambda name, value, *, attributes=None, unit=None: distributions.append(
            (name, value, unit, dict(attributes or {}))
        ),
    )
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

    assert ("ce.batch.started_total", 1, {"result": "started"}) in counts
    assert ("ce.batch.finished_total", 1, {"result": "ok"}) in counts
    assert (
        "ce.batch.time_in_pending_ms",
        2000.0,
        "millisecond",
        {"result": "started"},
    ) in distributions

"""Behavior tests for windowed admission + flush.

Coverage:
- admit_event skips enqueue when the pot's mode is 'windowed'
- admit_event enqueues immediately for 'immediate' mode (legacy + opt-in)
- admit_event defaults to immediate when no config port is wired (back-compat)
- flush_ready_windowed_pots enqueues exactly the ready pots
- flush_ready_windowed_pots tolerates a missing open batch (race)
- force_flush_pot enqueues once or returns None when nothing pending
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from potpie.context_engine.application.services.event_admission import admit_event
from potpie.context_engine.application.use_cases.flush_windowed_batches import (
    flush_ready_windowed_pots,
    force_flush_pot,
)
from potpie.context_engine.domain.context_events import ContextEvent, EventScope
from potpie.context_engine.domain.ports.ingestion_config import (
    IngestionConfig,
    InMemoryIngestionConfig,
)

pytestmark = pytest.mark.unit


def _event(pot_id: str = "pot-1") -> ContextEvent:
    return ContextEvent(
        event_id="e1",
        source_system="test",
        event_type="x",
        action="y",
        pot_id=pot_id,
        provider="github",
        provider_host="github.com",
        repo_name="o/r",
        source_id="src-1",
        source_event_id=None,
        payload={},
        occurred_at=datetime.now(timezone.utc),
        received_at=None,
        ingestion_kind="agent_reconciliation",
    )


def _scope(pot_id: str = "pot-1") -> EventScope:
    return EventScope(
        pot_id=pot_id,
        provider="github",
        provider_host="github.com",
        repo_name="o/r",
    )


def _wiring() -> tuple[MagicMock, MagicMock, MagicMock]:
    reco = MagicMock()
    reco.append_event.return_value = ("e1", True)
    batches = MagicMock()
    batches.upsert_open_batch_for_pot.return_value = "batch-1"
    jobs = MagicMock()
    return reco, batches, jobs


class TestAdmitEventWindowed:
    def test_windowed_mode_does_not_enqueue(self) -> None:
        reco, batches, jobs = _wiring()
        cfg = InMemoryIngestionConfig(
            {
                "pot-1": IngestionConfig(
                    pot_id="pot-1",
                    mode="windowed",
                    window_minutes=5,
                    min_batch_size=None,
                )
            },
        )
        outcome = admit_event(
            reco,
            batches,
            jobs,
            _scope(),
            _event(),
            ingestion_config=cfg,
        )
        assert outcome.inserted is True
        assert outcome.enqueued is False
        assert outcome.batch_id == "batch-1"
        jobs.enqueue_batch.assert_not_called()
        # The event was still queued in the ledger so the flusher can find it.
        reco.mark_event_queued.assert_called_once_with("e1")

    def test_immediate_mode_enqueues_right_away(self) -> None:
        reco, batches, jobs = _wiring()
        cfg = InMemoryIngestionConfig(
            {
                "pot-1": IngestionConfig(
                    pot_id="pot-1",
                    mode="immediate",
                    window_minutes=5,
                    min_batch_size=None,
                )
            }
        )
        outcome = admit_event(
            reco,
            batches,
            jobs,
            _scope(),
            _event(),
            ingestion_config=cfg,
        )
        assert outcome.enqueued is True
        jobs.enqueue_batch.assert_called_once_with("batch-1")

    def test_default_when_no_config_port_is_immediate(self) -> None:
        # Back-compat: callers that don't pass a config port get the legacy
        # behaviour (enqueue right away).
        reco, batches, jobs = _wiring()
        outcome = admit_event(reco, batches, jobs, _scope(), _event())
        assert outcome.enqueued is True
        jobs.enqueue_batch.assert_called_once()

    def test_config_read_failure_falls_back_to_immediate(self) -> None:
        # If reading the config raises, we err toward "enqueue" — better to
        # over-ingest than to drop events on the floor.
        reco, batches, jobs = _wiring()
        cfg = MagicMock()
        cfg.get.side_effect = RuntimeError("db down")
        outcome = admit_event(
            reco,
            batches,
            jobs,
            _scope(),
            _event(),
            ingestion_config=cfg,
        )
        assert outcome.enqueued is True

    def test_duplicate_event_short_circuits_before_enqueue_decision(self) -> None:
        reco, batches, jobs = _wiring()
        reco.append_event.return_value = ("e1", False)  # duplicate
        outcome = admit_event(
            reco,
            batches,
            jobs,
            _scope(),
            _event(),
            ingestion_config=InMemoryIngestionConfig(),
        )
        assert outcome.inserted is False
        # Duplicate path doesn't touch the batch or queue at all.
        batches.upsert_open_batch_for_pot.assert_not_called()
        jobs.enqueue_batch.assert_not_called()


class TestFlushReadyWindowedPots:
    def test_enqueues_each_ready_pot_open_batch_once(self) -> None:
        cfg = InMemoryIngestionConfig(
            {
                "p1": IngestionConfig(
                    pot_id="p1",
                    mode="windowed",
                    window_minutes=5,
                    min_batch_size=None,
                ),
                "p2": IngestionConfig(
                    pot_id="p2",
                    mode="windowed",
                    window_minutes=5,
                    min_batch_size=None,
                ),
            },
            pending_batches_by_pot={
                "p1": [time.time() - 10 * 60],  # 10m old → ready
                "p2": [time.time() - 60],  # 1m old → not ready
            },
        )
        batches = MagicMock()
        batches.get_open_batch_id_for_pot.side_effect = lambda pid: (
            f"batch-{pid}" if pid == "p1" else None
        )
        jobs = MagicMock()
        outcome = flush_ready_windowed_pots(
            config=cfg,
            batches=batches,
            jobs=jobs,
            now_unix_seconds=time.time(),
        )
        assert outcome.pot_ids_flushed == ["p1"]
        assert outcome.batches_enqueued == 1
        assert outcome.errors == 0
        jobs.enqueue_batch.assert_called_once_with("batch-p1")

    def test_skips_pot_with_no_open_batch_without_failing(self) -> None:
        # Race: config says ready, but the batch was just claimed elsewhere.
        cfg = InMemoryIngestionConfig(
            {
                "p1": IngestionConfig(
                    pot_id="p1",
                    mode="windowed",
                    window_minutes=5,
                    min_batch_size=None,
                )
            },
            pending_batches_by_pot={"p1": [time.time() - 10 * 60]},
        )
        batches = MagicMock()
        batches.get_open_batch_id_for_pot.return_value = None
        jobs = MagicMock()
        outcome = flush_ready_windowed_pots(
            config=cfg,
            batches=batches,
            jobs=jobs,
        )
        assert outcome.batches_enqueued == 0
        assert outcome.errors == 0
        jobs.enqueue_batch.assert_not_called()

    def test_enqueue_failure_records_error_and_continues(self) -> None:
        cfg = InMemoryIngestionConfig(
            {
                "p1": IngestionConfig(
                    pot_id="p1",
                    mode="windowed",
                    window_minutes=5,
                    min_batch_size=None,
                ),
                "p2": IngestionConfig(
                    pot_id="p2",
                    mode="windowed",
                    window_minutes=5,
                    min_batch_size=None,
                ),
            },
            pending_batches_by_pot={
                "p1": [time.time() - 10 * 60],
                "p2": [time.time() - 10 * 60],
            },
        )
        batches = MagicMock()
        batches.get_open_batch_id_for_pot.side_effect = lambda pid: f"batch-{pid}"
        jobs = MagicMock()

        def maybe_fail(bid):
            if bid == "batch-p1":
                raise RuntimeError("broker down")

        jobs.enqueue_batch.side_effect = maybe_fail
        outcome = flush_ready_windowed_pots(
            config=cfg,
            batches=batches,
            jobs=jobs,
        )
        assert "p2" in outcome.pot_ids_flushed
        assert outcome.errors == 1


class TestForceFlushPot:
    def test_returns_batch_id_and_enqueues_when_open(self) -> None:
        batches = MagicMock()
        batches.get_open_batch_id_for_pot.return_value = "batch-1"
        jobs = MagicMock()
        out = force_flush_pot(pot_id="p1", batches=batches, jobs=jobs)
        assert out == "batch-1"
        jobs.enqueue_batch.assert_called_once_with("batch-1")

    def test_returns_none_when_no_pending_batch(self) -> None:
        batches = MagicMock()
        batches.get_open_batch_id_for_pot.return_value = None
        jobs = MagicMock()
        out = force_flush_pot(pot_id="p1", batches=batches, jobs=jobs)
        assert out is None
        jobs.enqueue_batch.assert_not_called()

    def test_enqueue_failure_still_returns_batch_id(self) -> None:
        # The DB batch is durable; the user sees the batch_id so they can
        # follow up. The error is logged for the operator.
        batches = MagicMock()
        batches.get_open_batch_id_for_pot.return_value = "batch-1"
        jobs = MagicMock()
        jobs.enqueue_batch.side_effect = RuntimeError("broker down")
        out = force_flush_pot(pot_id="p1", batches=batches, jobs=jobs)
        assert out == "batch-1"


class TestInMemoryIngestionConfigList:
    def test_only_returns_windowed_pots(self) -> None:
        cfg = InMemoryIngestionConfig(
            {
                "p1": IngestionConfig(
                    pot_id="p1",
                    mode="windowed",
                    window_minutes=5,
                    min_batch_size=None,
                ),
                "p2": IngestionConfig(
                    pot_id="p2",
                    mode="immediate",
                    window_minutes=5,
                    min_batch_size=None,
                ),
            },
            pending_batches_by_pot={
                "p1": [time.time() - 10 * 60],
                "p2": [time.time() - 10 * 60],
            },
        )
        ready = cfg.list_windowed_pots_ready_to_flush(
            as_of_unix_seconds=time.time(),
        )
        assert [c.pot_id for c in ready] == ["p1"]

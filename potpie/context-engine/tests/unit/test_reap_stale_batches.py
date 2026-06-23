"""Unit tests for the stale-batch reaper.

Covers the recovery the rest of the pipeline cannot do itself: a batch
whose worker died mid-run (no redelivery, never re-claimable) must be
driven to a terminal ``failed`` so its events stop being stuck forever —
without clobbering events an earlier partial attempt already completed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from application.use_cases.reap_stale_batches import reap_stale_batches
from bootstrap import sentry_metrics_runtime
from domain.reconciliation_batch import BatchEventRef, ReconciliationBatch


def _batch(bid: str, pot: str = "pot-1") -> ReconciliationBatch:
    return ReconciliationBatch(
        id=bid,
        pot_id=pot,
        status="running",
        attempt_count=1,
        created_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
        claimed_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
        completed_at=None,
        last_error=None,
    )


class TestReapStaleBatches:
    def test_no_stale_batches_is_a_noop(self) -> None:
        batches = MagicMock()
        batches.list_stale_in_flight_batches.return_value = []
        ledger = MagicMock()

        outcome = reap_stale_batches(
            batches=batches, reco_ledger=ledger, lease_seconds=6300
        )

        assert outcome == type(outcome)(0, 0, 0, [])
        batches.mark_batch_failed.assert_not_called()
        ledger.fail_inflight_events.assert_not_called()

    def test_reaps_each_stale_batch_events_first_then_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sentry_counts: list[tuple[str, int, dict[str, str]]] = []
        monkeypatch.setattr(
            sentry_metrics_runtime,
            "count",
            lambda name, value=1, *, attributes=None, unit=None: sentry_counts.append(
                (name, value, dict(attributes or {}))
            ),
        )
        batches = MagicMock()
        batches.list_stale_in_flight_batches.return_value = [
            _batch("b1"),
            _batch("b2"),
        ]
        batches.list_events_for_batch.side_effect = lambda bid: (
            [BatchEventRef(event_id=f"{bid}-e1", added_at=datetime.now(timezone.utc))]
        )
        ledger = MagicMock()
        ledger.fail_inflight_events.return_value = 1
        order: list[str] = []
        ledger.fail_inflight_events.side_effect = lambda ids, err: (
            order.append("events") or 1
        )
        batches.mark_batch_failed.side_effect = lambda bid, err: order.append("batch")

        outcome = reap_stale_batches(
            batches=batches, reco_ledger=ledger, lease_seconds=6300
        )

        assert outcome.batches_reaped == 2
        assert outcome.events_failed == 2
        assert outcome.errors == 0
        assert sorted(outcome.reaped_batch_ids) == ["b1", "b2"]
        # Events are failed before the batch — a crash between the two
        # leaves the batch in-flight (reaped again) rather than failed
        # with orphaned processing events.
        assert order == ["events", "batch", "events", "batch"]
        ledger.fail_inflight_events.assert_any_call(
            ["b1-e1"], ledger.fail_inflight_events.call_args_list[0].args[1]
        )
        assert sentry_counts == [
            ("ce.batch.reaped_total", 1, {"result": "reaped"}),
            ("ce.batch.reaped_total", 1, {"result": "reaped"}),
        ]

    def test_lease_is_passed_through_to_the_repo_query(self) -> None:
        batches = MagicMock()
        batches.list_stale_in_flight_batches.return_value = []
        reap_stale_batches(
            batches=batches, reco_ledger=MagicMock(), lease_seconds=1234.0
        )
        batches.list_stale_in_flight_batches.assert_called_once_with(1234.0)

    def test_one_failing_batch_does_not_abort_the_sweep(self) -> None:
        batches = MagicMock()
        batches.list_stale_in_flight_batches.return_value = [
            _batch("bad"),
            _batch("good"),
        ]
        batches.list_events_for_batch.return_value = [
            BatchEventRef(event_id="e", added_at=datetime.now(timezone.utc))
        ]
        ledger = MagicMock()
        ledger.fail_inflight_events.return_value = 1

        def fail_first(bid, err):
            if bid == "bad":
                raise RuntimeError("db blip")

        batches.mark_batch_failed.side_effect = fail_first

        outcome = reap_stale_batches(
            batches=batches, reco_ledger=ledger, lease_seconds=6300
        )

        assert outcome.errors == 1
        assert outcome.reaped_batch_ids == ["good"]
        assert outcome.batches_reaped == 1

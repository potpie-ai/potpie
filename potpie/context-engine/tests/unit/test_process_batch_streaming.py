"""Behavior tests for process_batch's live-streaming hooks.

The streaming side of ``process_batch`` is best-effort liveness: every
status transition (claimed → processing → done/failed) and terminal
``end`` is published, but a publish failure must not corrupt ingestion.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from potpie_context_engine.adapters.outbound.event_stream.inmemory_publisher import (
    InMemoryEventStreamPublisher,
)
from potpie_context_engine.adapters.outbound.reconciliation.noop_agent import NoOpReconciliationAgent
from potpie_context_engine.application.use_cases.process_batch import process_batch
from potpie_context_engine.domain.ports.reconciliation_ledger import ContextEventRow
from potpie_context_core.ports.pot_resolution import ResolvedPot, ResolvedPotRepo
from potpie_context_engine.domain.reconciliation_batch import (
    BATCH_STATUS_PENDING,
    BatchEventRef,
    ReconciliationBatch,
)

pytestmark = pytest.mark.unit


def _now() -> datetime:
    return datetime(2026, 5, 6, 9, 0, tzinfo=timezone.utc)


def _batch() -> ReconciliationBatch:
    return ReconciliationBatch(
        id="batch-1",
        pot_id="pot-1",
        status=BATCH_STATUS_PENDING,
        attempt_count=1,
        created_at=_now(),
        claimed_at=None,
        completed_at=None,
        last_error=None,
    )


def _event_row(eid: str) -> ContextEventRow:
    return ContextEventRow(
        id=eid,
        pot_id="pot-1",
        provider="github",
        provider_host="github.com",
        repo_name="o/r",
        source_system="test",
        event_type="x",
        action="y",
        source_id=f"src-{eid}",
        source_event_id=None,
        payload={},
        occurred_at=_now(),
        received_at=_now(),
        status="queued",
    )


def _pots() -> MagicMock:
    pots = MagicMock()
    pots.resolve_pot.return_value = ResolvedPot(
        pot_id="pot-1",
        name="p",
        repos=[
            ResolvedPotRepo(
                pot_id="pot-1",
                repo_id="r1",
                provider="github",
                provider_host="github.com",
                repo_name="o/r",
            )
        ],
    )
    return pots


def _wiring_for(event_ids: list[str]) -> tuple[MagicMock, MagicMock, MagicMock]:
    batches = MagicMock()
    batches.list_events_for_batch.return_value = [
        BatchEventRef(event_id=eid, added_at=_now()) for eid in event_ids
    ]
    ledger = MagicMock()
    ledger.get_event_by_id.side_effect = lambda eid: _event_row(eid)
    checkpoints = MagicMock()
    checkpoints.load.return_value = None
    return batches, ledger, checkpoints


class _RecordingExecutionLog:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def load_resume_state(self, batch_id):
        del batch_id
        return None

    def append(self, *, batch_id, seq, record_type, payload, event_id=None):
        self.records.append(
            {
                "batch_id": batch_id,
                "seq": seq,
                "record_type": record_type,
                "payload": payload,
                "event_id": event_id,
            }
        )

    def clear(self, batch_id):
        del batch_id


class TestStatusPublishing:
    def test_success_publishes_processing_then_done(self) -> None:
        batches, ledger, checkpoints = _wiring_for(["e1", "e2"])
        pub = InMemoryEventStreamPublisher()
        process_batch(
            batch=_batch(),
            agent=NoOpReconciliationAgent(),
            batches=batches,
            reco_ledger=ledger,
            checkpoints=checkpoints,
            pots=_pots(),
            stream_publisher=pub,
        )
        statuses = [(e["event_id"], e["status"]) for e in pub.status_events]
        # Must see "processing" for both events before "done" for both.
        proc_idx = [i for i, s in enumerate(statuses) if s[1] == "processing"]
        done_idx = [i for i, s in enumerate(statuses) if s[1] == "done"]
        assert proc_idx, "expected at least one processing status"
        assert done_idx, "expected at least one done status"
        assert max(proc_idx) < min(done_idx)
        # Both events get a terminal end marker.
        end_ids = {e["event_id"] for e in pub.end_events}
        assert end_ids == {"e1", "e2"}

    def test_processing_status_is_persisted_and_durable(self) -> None:
        batches, ledger, checkpoints = _wiring_for(["e1", "e2"])
        log = _RecordingExecutionLog()
        process_batch(
            batch=_batch(),
            agent=NoOpReconciliationAgent(),
            batches=batches,
            reco_ledger=ledger,
            checkpoints=checkpoints,
            pots=_pots(),
            execution_log=log,
        )

        claimed_ids = {
            c.args[0] for c in ledger.claim_event_for_processing.call_args_list
        }
        assert claimed_ids == {"e1", "e2"}
        status_records = [r for r in log.records if r["record_type"] == "status"]
        assert {r["event_id"] for r in status_records} == {"e1", "e2"}
        assert all(r["payload"]["status"] == "processing" for r in status_records)

    def test_agent_crash_publishes_failed_status_and_end(self) -> None:
        batches, ledger, checkpoints = _wiring_for(["e1"])
        agent = MagicMock()
        agent.run_batch.side_effect = RuntimeError("boom")
        agent.capability_metadata.return_value = {}

        pub = InMemoryEventStreamPublisher()
        out = process_batch(
            batch=_batch(),
            agent=agent,
            batches=batches,
            reco_ledger=ledger,
            checkpoints=checkpoints,
            pots=_pots(),
            stream_publisher=pub,
        )
        assert out.ok is False
        statuses = [(e["event_id"], e["status"]) for e in pub.status_events]
        # processing was published, then failed.
        assert ("e1", "processing") in statuses
        assert ("e1", "failed") in statuses
        # An end marker carries the error.
        ends = pub.end_events
        assert any(
            e["event_id"] == "e1" and e["status"] == "failed" and e["error"] == "boom"
            for e in ends
        )

    def test_publish_failures_do_not_break_processing(self) -> None:
        # If the publisher raises for every call, the batch must still
        # complete and return ok. Ingestion correctness is independent of
        # liveness.
        batches, ledger, checkpoints = _wiring_for(["e1"])
        pub = MagicMock()
        pub.publish_status.side_effect = RuntimeError("redis down")
        pub.publish_end.side_effect = RuntimeError("redis down")

        out = process_batch(
            batch=_batch(),
            agent=NoOpReconciliationAgent(),
            batches=batches,
            reco_ledger=ledger,
            checkpoints=checkpoints,
            pots=_pots(),
            stream_publisher=pub,
        )
        assert out.ok is True
        assert "e1" in out.completed_event_ids


class TestDefaultPublisherIsNoOp:
    def test_processing_works_without_publisher_arg(self) -> None:
        # Legacy call sites pass no publisher — must still work end-to-end.
        batches, ledger, checkpoints = _wiring_for(["e1"])
        out = process_batch(
            batch=_batch(),
            agent=NoOpReconciliationAgent(),
            batches=batches,
            reco_ledger=ledger,
            checkpoints=checkpoints,
            pots=_pots(),
        )
        assert out.ok is True

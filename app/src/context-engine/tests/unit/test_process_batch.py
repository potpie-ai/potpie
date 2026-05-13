"""Tests for ``process_batch`` (loads batch, runs agent, persists outcome)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from adapters.outbound.reconciliation.noop_agent import NoOpReconciliationAgent
from application.use_cases.process_batch import process_batch
from domain.ports.reconciliation_ledger import ContextEventRow
from domain.ports.pot_resolution import ResolvedPot, ResolvedPotRepo
from domain.reconciliation_batch import (
    BATCH_STATUS_DONE,
    BATCH_STATUS_PENDING,
    BatchEventRef,
    ReconciliationBatch,
)

pytestmark = pytest.mark.unit


def _now() -> datetime:
    return datetime(2026, 5, 6, 9, 0, tzinfo=timezone.utc)


def _batch(status: str = BATCH_STATUS_PENDING, attempt: int = 1) -> ReconciliationBatch:
    return ReconciliationBatch(
        id="batch-1",
        pot_id="pot-1",
        status=status,
        attempt_count=attempt,
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


def test_processes_pending_events_and_marks_batch_done() -> None:
    batches = MagicMock()
    batches.list_events_for_batch.return_value = [
        BatchEventRef(event_id="e1", added_at=_now()),
        BatchEventRef(event_id="e2", added_at=_now()),
    ]
    ledger = MagicMock()
    ledger.get_event_by_id.side_effect = lambda eid: _event_row(eid)
    checkpoints = MagicMock()
    checkpoints.load.return_value = None

    out = process_batch(
        batch=_batch(),
        agent=NoOpReconciliationAgent(),
        batches=batches,
        reco_ledger=ledger,
        checkpoints=checkpoints,
        pots=_pots(),
    )

    assert out.ok is True
    assert sorted(out.completed_event_ids) == ["e1", "e2"]
    batches.mark_batch_running.assert_called_once_with("batch-1")
    batches.mark_batch_done.assert_called_once()
    args, kwargs = batches.mark_batch_done.call_args
    assert args[0] == "batch-1"
    assert sorted(kwargs["completed_event_ids"]) == ["e1", "e2"]
    assert {c.args[0] for c in ledger.record_event_reconciled.call_args_list} == {"e1", "e2"}
    checkpoints.delete.assert_called_once_with("batch-1")


def test_skips_already_processed_events() -> None:
    batches = MagicMock()
    batches.list_events_for_batch.return_value = [
        BatchEventRef(event_id="e1", added_at=_now(), processed_at=_now()),
        BatchEventRef(event_id="e2", added_at=_now()),
    ]
    ledger = MagicMock()
    ledger.get_event_by_id.return_value = _event_row("e2")
    checkpoints = MagicMock()
    checkpoints.load.return_value = None

    out = process_batch(
        batch=_batch(),
        agent=NoOpReconciliationAgent(),
        batches=batches,
        reco_ledger=ledger,
        checkpoints=checkpoints,
        pots=_pots(),
    )

    assert out.ok is True
    # Only the unprocessed event was passed to the agent (via NoOp it's completed).
    assert out.completed_event_ids == ["e2"]


def test_empty_batch_closes_immediately() -> None:
    batches = MagicMock()
    batches.list_events_for_batch.return_value = []
    ledger = MagicMock()
    checkpoints = MagicMock()

    out = process_batch(
        batch=_batch(),
        agent=NoOpReconciliationAgent(),
        batches=batches,
        reco_ledger=ledger,
        checkpoints=checkpoints,
        pots=_pots(),
    )

    assert out.ok is True
    assert out.completed_event_ids == []
    batches.mark_batch_done.assert_called_once_with("batch-1", completed_event_ids=[])
    batches.mark_batch_running.assert_not_called()
    checkpoints.delete.assert_called_once_with("batch-1")


def test_done_batch_is_a_noop() -> None:
    batches = MagicMock()
    ledger = MagicMock()
    checkpoints = MagicMock()
    out = process_batch(
        batch=_batch(status=BATCH_STATUS_DONE),
        agent=NoOpReconciliationAgent(),
        batches=batches,
        reco_ledger=ledger,
        checkpoints=checkpoints,
        pots=_pots(),
    )
    assert out.ok is True
    batches.mark_batch_running.assert_not_called()
    batches.mark_batch_done.assert_not_called()


def test_agent_exception_marks_batch_failed_and_events_failed() -> None:
    batches = MagicMock()
    batches.list_events_for_batch.return_value = [
        BatchEventRef(event_id="e1", added_at=_now()),
    ]
    ledger = MagicMock()
    ledger.get_event_by_id.return_value = _event_row("e1")
    checkpoints = MagicMock()
    checkpoints.load.return_value = None

    class _Boom:
        def run_batch(self, ctx, *, checkpoints=None):
            del ctx, checkpoints
            raise RuntimeError("kaboom")

        def capability_metadata(self):
            return {}

    out = process_batch(
        batch=_batch(),
        agent=_Boom(),
        batches=batches,
        reco_ledger=ledger,
        checkpoints=checkpoints,
        pots=_pots(),
    )

    assert out.ok is False
    assert "kaboom" in (out.error or "")
    batches.mark_batch_failed.assert_called_once()
    ledger.record_event_failed.assert_called_once()
    args, _ = ledger.record_event_failed.call_args
    assert args[0] == "e1"


def test_opens_runs_and_fans_work_events_across_events() -> None:
    """Each pending event gets its own run, and the agent's trace is copied to all of them."""
    batches = MagicMock()
    batches.list_events_for_batch.return_value = [
        BatchEventRef(event_id="e1", added_at=_now()),
        BatchEventRef(event_id="e2", added_at=_now()),
    ]
    ledger = MagicMock()
    ledger.get_event_by_id.side_effect = lambda eid: _event_row(eid)
    ledger.next_attempt_number.side_effect = lambda _eid: 1
    ledger.start_reconciliation_run.side_effect = lambda _eid, **_: f"run-{_eid}"
    checkpoints = MagicMock()
    checkpoints.load.return_value = None

    history = [
        {"kind": "request", "parts": [{"part_kind": "user-prompt", "content": "go"}]},
        {
            "kind": "response",
            "parts": [
                {
                    "part_kind": "tool-call",
                    "tool_name": "context_search",
                    "args": {"q": "x"},
                    "tool_call_id": "c1",
                },
                {"part_kind": "text", "content": "done"},
            ],
        },
    ]

    class _Stub:
        def run_batch(self, ctx, *, checkpoints=None):
            del checkpoints
            from domain.reconciliation_batch import BatchAgentOutcome

            return BatchAgentOutcome(
                ok=True,
                completed_event_ids=[ev.event_id for ev in ctx.events],
                tool_call_count=1,
                prompt="go",
                agent_messages_json=history,
                final_response="done",
                agent_name="pydantic-deep",
                agent_version="0.0.0",
                toolset_version="batch-tools-v1",
            )

        def capability_metadata(self):
            return {
                "agent": "pydantic-deep",
                "version": "0.0.0",
                "toolset_version": "batch-tools-v1",
            }

    out = process_batch(
        batch=_batch(),
        agent=_Stub(),
        batches=batches,
        reco_ledger=ledger,
        checkpoints=checkpoints,
        pots=_pots(),
    )

    assert out.ok is True
    # One run row started per pending event.
    started_ids = {c.args[0] for c in ledger.start_reconciliation_run.call_args_list}
    assert started_ids == {"e1", "e2"}
    # The trace fanned into both runs (prompt + tool_call + plan_output = 3 work events × 2 runs).
    appended_run_ids = {
        c.args[0] for c in ledger.record_run_work_event.call_args_list
    }
    assert appended_run_ids == {"run-e1", "run-e2"}
    appended_kinds_per_run: dict[str, list[str]] = {}
    for c in ledger.record_run_work_event.call_args_list:
        appended_kinds_per_run.setdefault(c.args[0], []).append(c.kwargs["event_kind"])
    for kinds in appended_kinds_per_run.values():
        assert kinds == ["prompt", "tool_call", "plan_output"]
    # Both runs marked succeeded.
    success_ids = {c.args[0] for c in ledger.record_run_success.call_args_list}
    assert success_ids == {"run-e1", "run-e2"}


def test_failing_run_marks_run_failed_and_records_error_event() -> None:
    batches = MagicMock()
    batches.list_events_for_batch.return_value = [
        BatchEventRef(event_id="e1", added_at=_now()),
    ]
    ledger = MagicMock()
    ledger.get_event_by_id.side_effect = lambda eid: _event_row(eid)
    ledger.next_attempt_number.return_value = 1
    ledger.start_reconciliation_run.return_value = "run-e1"
    checkpoints = MagicMock()
    checkpoints.load.return_value = None

    class _Boom:
        def run_batch(self, ctx, *, checkpoints=None):
            del ctx, checkpoints
            raise RuntimeError("kaboom")

        def capability_metadata(self):
            return {"agent": "x", "version": "1", "toolset_version": "v1"}

    out = process_batch(
        batch=_batch(),
        agent=_Boom(),
        batches=batches,
        reco_ledger=ledger,
        checkpoints=checkpoints,
        pots=_pots(),
    )

    assert out.ok is False
    ledger.record_run_failure.assert_called_once()
    err_kinds = [
        c.kwargs["event_kind"] for c in ledger.record_run_work_event.call_args_list
    ]
    assert "error" in err_kinds


def test_resumes_with_prior_messages_when_checkpoint_exists() -> None:
    """When a checkpoint exists, ``BatchAgentContext.prior_messages_json`` is populated."""

    batches = MagicMock()
    batches.list_events_for_batch.return_value = [
        BatchEventRef(event_id="e1", added_at=_now()),
    ]
    ledger = MagicMock()
    ledger.get_event_by_id.return_value = _event_row("e1")
    checkpoints = MagicMock()

    class _Checkpoint:
        messages_json = [{"role": "assistant", "content": "prior"}]
        tool_call_count = 3

    checkpoints.load.return_value = _Checkpoint()

    captured = {}

    class _Capture:
        def run_batch(self, ctx, *, checkpoints=None):
            captured["prior"] = ctx.prior_messages_json
            captured["attempt"] = ctx.attempt_number
            from domain.reconciliation_batch import BatchAgentOutcome

            return BatchAgentOutcome(ok=True, completed_event_ids=["e1"])

        def capability_metadata(self):
            return {}

    process_batch(
        batch=_batch(attempt=2),
        agent=_Capture(),
        batches=batches,
        reco_ledger=ledger,
        checkpoints=checkpoints,
        pots=_pots(),
    )

    assert captured["prior"] == [{"role": "assistant", "content": "prior"}]
    assert captured["attempt"] == 2

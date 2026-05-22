"""CGT-7: ``process_batch`` failure-path and crash-resume invariants.

Pins behaviour the fake E2E harness does not cover:
- execution-log resume survives agent exceptions and ``ok=False`` outcomes;
- partial chunk completions are credited before failure;
- a retry resumes with prior message history and skips already-done events.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from application.use_cases.process_batch import process_batch
from domain.ports.agent_execution_log import ResumeState
from domain.ports.pot_resolution import ResolvedPot, ResolvedPotRepo
from domain.ports.reconciliation_ledger import ContextEventRow
from domain.reconciliation_batch import (
    BATCH_STATUS_PENDING,
    BatchAgentOutcome,
    BatchEventRef,
    ReconciliationBatch,
)

pytestmark = pytest.mark.unit


def _now() -> datetime:
    return datetime(2026, 5, 22, 12, 0, tzinfo=timezone.utc)


def _batch(*, attempt: int = 1) -> ReconciliationBatch:
    return ReconciliationBatch(
        id="batch-1",
        pot_id="pot-1",
        status=BATCH_STATUS_PENDING,
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


def _wiring(event_ids: list[str]) -> tuple[MagicMock, MagicMock]:
    batches = MagicMock()
    batches.list_events_for_batch.return_value = [
        BatchEventRef(event_id=eid, added_at=_now()) for eid in event_ids
    ]
    ledger = MagicMock()
    ledger.get_event_by_id.side_effect = lambda eid: _event_row(eid)
    ledger.start_reconciliation_run.side_effect = lambda eid, **_: f"run-{eid}"
    ledger.next_attempt_number.return_value = 1
    return batches, ledger


class _FakeExecutionLog:
    """Minimal durable log stub for resume contract tests."""

    def __init__(self, resume: ResumeState | None) -> None:
        self._resume = resume
        self.records: list[tuple[int, str, dict]] = []
        self.cleared = False

    def load_resume_state(self, batch_id: str) -> ResumeState | None:
        del batch_id
        return self._resume

    def append(
        self,
        *,
        batch_id: str,
        seq: int,
        record_type: str,
        payload: dict,
        event_id: str | None = None,
    ) -> None:
        del batch_id, event_id
        self.records.append((seq, record_type, payload))

    def clear(self, batch_id: str) -> None:
        del batch_id
        self.cleared = True


class _AgentRecorder:
    def __init__(self, outcomes: list[BatchAgentOutcome]) -> None:
        self.invocations: list[list[str]] = []
        self._outcomes = outcomes

    def capability_metadata(self) -> dict[str, str]:
        return {"agent": "t", "version": "1", "toolset_version": "t"}

    def run_batch(self, ctx, *, checkpoints=None, execution_log=None):
        del checkpoints, execution_log
        self.invocations.append([e.event_id for e in ctx.events])
        return self._outcomes[len(self.invocations) - 1]


class TestFailurePreservesResume:
    def test_agent_exception_does_not_clear_resume_checkpoint(self) -> None:
        batches, ledger = _wiring(["e1"])
        resume = ResumeState(
            batch_id="batch-1",
            messages_json=[{"prior": True}],
            tool_call_count=2,
            completed_event_ids=[],
            last_seq=7,
        )
        log = _FakeExecutionLog(resume)

        class _Boom:
            def capability_metadata(self):
                return {}

            def run_batch(self, ctx, *, checkpoints=None, execution_log=None):
                del ctx, checkpoints, execution_log
                raise RuntimeError("kaboom")

        out = process_batch(
            batch=_batch(),
            agent=_Boom(),
            batches=batches,
            reco_ledger=ledger,
            checkpoints=MagicMock(),
            pots=_pots(),
            execution_log=log,
        )
        assert out.ok is False
        assert log.cleared is False
        assert log.load_resume_state("batch-1") is resume

    def test_agent_ok_false_does_not_clear_resume_checkpoint(self) -> None:
        batches, ledger = _wiring(["e1"])
        resume = ResumeState(
            batch_id="batch-1",
            messages_json=[{"prior": True}],
            tool_call_count=1,
            completed_event_ids=[],
            last_seq=4,
        )
        log = _FakeExecutionLog(resume)
        agent = _AgentRecorder(
            [BatchAgentOutcome(ok=False, error="model refused", last_seq=6)]
        )
        out = process_batch(
            batch=_batch(),
            agent=agent,
            batches=batches,
            reco_ledger=ledger,
            checkpoints=MagicMock(),
            pots=_pots(),
            execution_log=log,
        )
        assert out.ok is False
        assert log.cleared is False
        assert [r[1] for r in log.records][-1] == "run_failed"

    def test_agent_exception_emits_run_failed_after_resume_seq(self) -> None:
        batches, ledger = _wiring(["e1"])
        resume = ResumeState(
            batch_id="batch-1",
            messages_json=[],
            tool_call_count=0,
            completed_event_ids=[],
            last_seq=11,
        )
        log = _FakeExecutionLog(resume)

        class _Boom:
            def capability_metadata(self):
                return {}

            def run_batch(self, ctx, *, checkpoints=None, execution_log=None):
                del ctx, checkpoints, execution_log
                raise RuntimeError("kaboom")

        process_batch(
            batch=_batch(),
            agent=_Boom(),
            batches=batches,
            reco_ledger=ledger,
            checkpoints=MagicMock(),
            pots=_pots(),
            execution_log=log,
        )
        terminal = [r for r in log.records if r[1] == "run_failed"]
        assert len(terminal) == 1
        assert terminal[0][0] > resume.last_seq


class TestRetryAfterPartialFailure:
    def test_retry_skips_events_already_credited_before_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Chunk 1 completes; chunk 2 fails; retry must not re-hand e1/e2."""
        monkeypatch.setenv("CONTEXT_ENGINE_MAX_CHUNK_EVENTS", "2")
        eids = ["e1", "e2", "e3", "e4"]
        batches, ledger = _wiring(eids)

        first_agent = _AgentRecorder(
            [
                BatchAgentOutcome(ok=True, completed_event_ids=["e1", "e2"]),
                BatchAgentOutcome(ok=False, error="boom"),
            ]
        )
        first = process_batch(
            batch=_batch(attempt=1),
            agent=first_agent,
            batches=batches,
            reco_ledger=ledger,
            checkpoints=MagicMock(),
            pots=_pots(),
        )
        assert first.ok is False
        assert sorted(first.completed_event_ids) == ["e1", "e2"]
        assert len(first_agent.invocations) == 2

        resume = ResumeState(
            batch_id="batch-1",
            messages_json=[{"role": "assistant", "content": "partial"}],
            tool_call_count=3,
            completed_event_ids=["e1", "e2"],
            last_seq=20,
            chunk_index=0,
        )
        retry_log = _FakeExecutionLog(resume)
        retry_agent = _AgentRecorder(
            [BatchAgentOutcome(ok=True, completed_event_ids=["e3", "e4"])]
        )
        second = process_batch(
            batch=_batch(attempt=2),
            agent=retry_agent,
            batches=batches,
            reco_ledger=ledger,
            checkpoints=MagicMock(),
            pots=_pots(),
            execution_log=retry_log,
        )
        assert second.ok is True
        assert sorted(second.completed_event_ids) == sorted(eids)
        assert len(retry_agent.invocations) == 1
        assert retry_agent.invocations[0] == ["e3", "e4"]
        assert retry_log.cleared is True

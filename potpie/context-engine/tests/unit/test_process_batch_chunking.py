"""Behavior tests for chunked execution of process_batch.

The single-chunk path is covered by ``test_process_batch.py``; this file
verifies the multi-chunk behavior introduced in Phase 5:

- batches above the threshold split into N chunks
- each chunk gets its own ``agent.run_batch`` invocation with fresh history
- per-chunk completions aggregate
- work events from chunk N fan only to chunk N's event runs
- a chunk failure stops the loop and partial completions still survive
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from context_engine.application.use_cases.process_batch import (
    _chunk_events,
    _chunk_size_default,
    process_batch,
)
from context_engine.domain.context_events import ContextEvent
from context_engine.domain.ports.pot_resolution import ResolvedPot, ResolvedPotRepo
from context_engine.domain.ports.reconciliation_ledger import ContextEventRow
from context_engine.domain.reconciliation_batch import (
    BATCH_STATUS_PENDING,
    BatchAgentOutcome,
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


def _wiring(event_ids: list[str]) -> tuple[MagicMock, MagicMock, MagicMock]:
    batches = MagicMock()
    batches.list_events_for_batch.return_value = [
        BatchEventRef(event_id=eid, added_at=_now()) for eid in event_ids
    ]
    ledger = MagicMock()
    ledger.get_event_by_id.side_effect = lambda eid: _event_row(eid)
    ledger.start_reconciliation_run.side_effect = lambda eid, **_: f"run-{eid}"
    ledger.next_attempt_number.return_value = 1
    checkpoints = MagicMock()
    checkpoints.load.return_value = None
    return batches, ledger, checkpoints


class _AgentRecorder:
    """Records each run_batch invocation and returns a configurable outcome."""

    def __init__(self, outcomes: list[BatchAgentOutcome]) -> None:
        self.invocations: list[list[ContextEvent]] = []
        self._outcomes = outcomes

    def capability_metadata(self) -> dict[str, str]:
        return {
            "agent": "test",
            "version": "1",
            "toolset_version": "t",
        }

    def run_batch(self, ctx, *, checkpoints=None, execution_log=None):  # noqa: D401
        del checkpoints, execution_log
        self.invocations.append(list(ctx.events))
        idx = len(self.invocations) - 1
        return self._outcomes[idx]


class TestChunkHelpers:
    def test_single_chunk_for_small_batch(self) -> None:
        # _chunk_events expects ContextEvent objects, but _event_row returns
        # ContextEventRow; just check the chunking math via a sentinel list.
        out = _chunk_events(["a", "b", "c"], 5)  # type: ignore[arg-type]
        assert len(out) == 1
        assert out[0] == ["a", "b", "c"]

    def test_splits_into_equal_chunks(self) -> None:
        items = list(range(10))
        out = _chunk_events(items, 4)  # type: ignore[arg-type]
        assert [len(c) for c in out] == [4, 4, 2]
        # Order preserved.
        assert [v for c in out for v in c] == items

    def test_default_chunk_size_falls_back_to_20(self, monkeypatch) -> None:
        monkeypatch.delenv("CONTEXT_ENGINE_MAX_CHUNK_EVENTS", raising=False)
        assert _chunk_size_default() == 20

    def test_chunk_size_env_overrides(self, monkeypatch) -> None:
        monkeypatch.setenv("CONTEXT_ENGINE_MAX_CHUNK_EVENTS", "5")
        assert _chunk_size_default() == 5

    def test_chunk_size_env_garbage_falls_back(self, monkeypatch) -> None:
        monkeypatch.setenv("CONTEXT_ENGINE_MAX_CHUNK_EVENTS", "not_a_number")
        assert _chunk_size_default() == 20


class TestChunkedProcessing:
    def test_large_batch_runs_one_agent_call_per_chunk(self, monkeypatch) -> None:
        monkeypatch.setenv("CONTEXT_ENGINE_MAX_CHUNK_EVENTS", "2")
        eids = ["e1", "e2", "e3", "e4", "e5"]
        batches, ledger, checkpoints = _wiring(eids)
        agent = _AgentRecorder(
            [
                BatchAgentOutcome(ok=True, completed_event_ids=["e1", "e2"]),
                BatchAgentOutcome(ok=True, completed_event_ids=["e3", "e4"]),
                BatchAgentOutcome(ok=True, completed_event_ids=["e5"]),
            ]
        )
        out = process_batch(
            batch=_batch(),
            agent=agent,
            batches=batches,
            reco_ledger=ledger,
            checkpoints=checkpoints,
            pots=_pots(),
        )
        # 5 events / 2 per chunk → 3 chunks
        assert len(agent.invocations) == 3
        assert [len(c) for c in agent.invocations] == [2, 2, 1]
        assert out.ok is True
        assert out.chunks_run == 3
        assert sorted(out.completed_event_ids) == sorted(eids)

    def test_each_chunk_runs_fresh_message_history_after_first(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("CONTEXT_ENGINE_MAX_CHUNK_EVENTS", "1")
        eids = ["e1", "e2", "e3"]
        batches, ledger, checkpoints = _wiring(eids)
        # No execution log wired → NoOp → no resume state, so every chunk
        # (including the first) starts with a fresh message history. The
        # execution-log-driven resume path is covered in
        # test_resume_from_execution_log_*.

        agent = _AgentRecorder(
            [BatchAgentOutcome(ok=True, completed_event_ids=[eid]) for eid in eids]
        )

        class _CtxCapturingAgent(_AgentRecorder):
            def __init__(self) -> None:
                super().__init__(
                    [
                        BatchAgentOutcome(ok=True, completed_event_ids=[eid])
                        for eid in eids
                    ]
                )
                self.priors: list[object] = []

            def run_batch(self, ctx, *, checkpoints=None, execution_log=None):  # noqa: D401
                self.priors.append(ctx.prior_messages_json)
                return super().run_batch(
                    ctx, checkpoints=checkpoints, execution_log=execution_log
                )

        agent = _CtxCapturingAgent()
        process_batch(
            batch=_batch(),
            agent=agent,
            batches=batches,
            reco_ledger=ledger,
            checkpoints=checkpoints,
            pots=_pots(),
        )
        # attempt_count=1 → first chunk also gets None (don't restore on
        # fresh attempts). Chunks 2+ always get None.
        assert agent.priors == [None, None, None]

    def test_chunk_failure_stops_loop_and_preserves_prior_chunk_completions(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("CONTEXT_ENGINE_MAX_CHUNK_EVENTS", "2")
        eids = ["e1", "e2", "e3", "e4"]
        batches, ledger, checkpoints = _wiring(eids)
        agent = _AgentRecorder(
            [
                BatchAgentOutcome(ok=True, completed_event_ids=["e1", "e2"]),
                # Chunk 2 fails — chunk 3 should NOT run.
                BatchAgentOutcome(ok=False, error="boom"),
                BatchAgentOutcome(ok=True, completed_event_ids=["e3", "e4"]),
            ]
        )

        out = process_batch(
            batch=_batch(),
            agent=agent,
            batches=batches,
            reco_ledger=ledger,
            checkpoints=checkpoints,
            pots=_pots(),
        )
        # Only the first two chunks ran.
        assert len(agent.invocations) == 2
        # Batch as a whole is marked failed, but the completed events from
        # chunk 1 should be reconciled.
        assert out.ok is False
        assert sorted(out.completed_event_ids) == ["e1", "e2"]
        completed: set[str] = set()
        for c in ledger.record_events_reconciled.call_args_list:
            completed.update(c.args[0])
        assert completed == {"e1", "e2"}
        # Chunk 2's events get marked failed.
        failed: set[str] = set()
        for c in ledger.record_events_failed.call_args_list:
            failed.update(c.args[0])
        assert failed == {"e3", "e4"}

    def test_work_events_fan_only_to_their_chunk(self, monkeypatch) -> None:
        # Phase 5 invariant: work events from chunk N only land on events
        # that were in chunk N — otherwise the user sees a mismatched trace
        # on an unrelated event.
        monkeypatch.setenv("CONTEXT_ENGINE_MAX_CHUNK_EVENTS", "2")
        eids = ["e1", "e2", "e3"]
        batches, ledger, checkpoints = _wiring(eids)
        agent = _AgentRecorder(
            [
                BatchAgentOutcome(
                    ok=True,
                    completed_event_ids=["e1", "e2"],
                    prompt="chunk1 prompt",
                    agent_messages_json=None,
                    final_response="chunk1 done",
                ),
                BatchAgentOutcome(
                    ok=True,
                    completed_event_ids=["e3"],
                    prompt="chunk2 prompt",
                    agent_messages_json=None,
                    final_response="chunk2 done",
                ),
            ]
        )
        process_batch(
            batch=_batch(),
            agent=agent,
            batches=batches,
            reco_ledger=ledger,
            checkpoints=checkpoints,
            pots=_pots(),
        )
        # record_run_work_event is called as (run_id, event_kind=..., ...).
        # Group call run_ids by which chunk they belong to.
        per_run: dict[str, list[str | None]] = {}
        for c in ledger.record_run_work_event.call_args_list:
            run_id = c.args[0]
            # ``body`` is the prompt body for prompt kind; we use it as a
            # proxy for "which chunk did this come from".
            per_run.setdefault(run_id, []).append(c.kwargs.get("body"))

        run_for = {eid: f"run-{eid}" for eid in eids}
        chunk1_runs = {run_for["e1"], run_for["e2"]}
        chunk2_runs = {run_for["e3"]}

        for run_id in chunk1_runs:
            bodies = per_run.get(run_id, [])
            # Any prompt body that's present should come from chunk1.
            assert all(b is None or "chunk1" in b for b in bodies), (
                f"chunk-1 run {run_id} got cross-chunk bodies {bodies}"
            )

        for run_id in chunk2_runs:
            bodies = per_run.get(run_id, [])
            assert all(b is None or "chunk2" in b for b in bodies), (
                f"chunk-2 run {run_id} got cross-chunk bodies {bodies}"
            )

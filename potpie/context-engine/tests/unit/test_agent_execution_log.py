"""Durable agent execution log: stream mapping, event-stream handler
translation, and process_batch resume contract.

These are pure / fake-backed unit tests — no Postgres. The Postgres
adapter's SQL is exercised by the integration suite; here we pin the
behaviour callers depend on.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from potpie_context_engine.adapters.outbound.postgres.agent_execution_log import _to_stream_event
from potpie_context_engine.adapters.outbound.reconciliation.pydantic_deep_agent import (
    _ExecutionLogSink,
    _make_event_stream_handler,
    _SeqAllocator,
)
from potpie_context_engine.application.use_cases.process_batch import process_batch
from potpie_context_engine.domain.ports.agent_execution_log import (
    NoOpAgentExecutionLog,
    ResumeState,
)
from potpie_context_engine.domain.ports.pot_resolution import ResolvedPot, ResolvedPotRepo
from potpie_context_engine.domain.ports.reconciliation_ledger import ContextEventRow
from potpie_context_engine.domain.reconciliation_batch import (
    BATCH_STATUS_PENDING,
    BatchAgentOutcome,
    BatchEventRef,
    ReconciliationBatch,
)

pytestmark = pytest.mark.unit


def _now() -> datetime:
    return datetime(2026, 5, 15, 9, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# _to_stream_event mapping
# --------------------------------------------------------------------------- #


def _row(**kw):
    base = dict(
        seq=1,
        record_type="text",
        event_id=None,
        part_id=None,
        done=True,
        payload={},
        created_at=_now(),
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestToStreamEvent:
    def test_text_part_maps_to_activity_with_part_id_and_done(self) -> None:
        ev = _to_stream_event(
            _row(
                seq=7,
                record_type="text",
                part_id="r1p0",
                done=False,
                payload={"content": "Look"},
            )
        )
        assert ev["type"] == "activity"
        assert ev["kind"] == "text"
        assert ev["stream_id"] == "7"
        assert ev["seq"] == 7
        assert ev["part_id"] == "r1p0"
        assert ev["done"] is False
        assert ev["body"] == "Look"

    def test_tool_call_carries_payload(self) -> None:
        ev = _to_stream_event(
            _row(
                seq=3,
                record_type="tool_call",
                payload={"tool_name": "context_search", "args": {"q": "x"}},
            )
        )
        assert ev["type"] == "activity"
        assert ev["kind"] == "tool_call"
        assert ev["payload"]["tool_name"] == "context_search"

    def test_mutation_applied_is_event_scoped(self) -> None:
        ev = _to_stream_event(
            _row(
                seq=9,
                record_type="mutation_applied",
                event_id="e1",
                payload={"counts": {"entity_upserts_applied": 3}},
            )
        )
        assert ev["kind"] == "mutation_applied"
        assert ev["event_id"] == "e1"
        assert ev["payload"]["counts"]["entity_upserts_applied"] == 3

    def test_run_finished_maps_to_terminal_end(self) -> None:
        ev = _to_stream_event(
            _row(record_type="run_finished", payload={"summary": "2 reconciled"})
        )
        assert ev["type"] == "end"
        assert ev["status"] == "done"

    def test_run_failed_maps_to_end_with_error(self) -> None:
        ev = _to_stream_event(_row(record_type="run_failed", payload={"error": "boom"}))
        assert ev["type"] == "end"
        assert ev["status"] == "failed"
        assert ev["error"] == "boom"


# --------------------------------------------------------------------------- #
# event_stream_handler translation
# --------------------------------------------------------------------------- #


class _RecordingLog:
    """Minimal AgentExecutionLogPort that records writes in memory."""

    def __init__(self) -> None:
        self.appended: list[dict] = []
        self.parts: list[dict] = []

    def append(self, *, batch_id, seq, record_type, payload, event_id=None):
        self.appended.append(
            {"seq": seq, "record_type": record_type, "payload": payload}
        )

    def upsert_part(
        self, *, batch_id, seq, record_type, part_id, content, done, event_id=None
    ):
        self.parts.append(
            {
                "seq": seq,
                "record_type": record_type,
                "part_id": part_id,
                "content": content,
                "done": done,
            }
        )


async def _drain(handler, events: list) -> None:
    async def _gen():
        for e in events:
            yield e

    await handler(SimpleNamespace(), _gen())


class TestEventStreamHandler:
    def test_text_and_thinking_parts_are_coalesced_and_finalized(self) -> None:
        from pydantic_ai.messages import (
            PartEndEvent,
            PartStartEvent,
            TextPart,
            ThinkingPart,
        )

        log = _RecordingLog()
        sink = _ExecutionLogSink(log, batch_id="b1", seq=_SeqAllocator(0))
        handler = _make_event_stream_handler(sink)

        events = [
            PartStartEvent(index=0, part=ThinkingPart(content="Plan: ")),
            PartEndEvent(index=0, part=ThinkingPart(content="Plan: do X")),
            PartStartEvent(index=1, part=TextPart(content="")),
            PartEndEvent(index=1, part=TextPart(content="Done.")),
        ]
        asyncio.run(_drain(handler, events))

        # Final state for each part is persisted with done=True.
        finals = {p["part_id"]: p for p in log.parts if p["done"]}
        thinking = [p for p in finals.values() if p["record_type"] == "thinking"]
        text = [p for p in finals.values() if p["record_type"] == "text"]
        assert thinking and thinking[0]["content"] == "Plan: do X"
        assert text and text[0]["content"] == "Done."
        # seqs are strictly increasing across all writes.
        seqs = [p["seq"] for p in log.parts]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)

    def test_tool_call_and_result_emit_discrete_records(self) -> None:
        from pydantic_ai.messages import (
            FunctionToolCallEvent,
            FunctionToolResultEvent,
            ToolCallPart,
            ToolReturnPart,
        )

        log = _RecordingLog()
        sink = _ExecutionLogSink(log, batch_id="b1", seq=_SeqAllocator(10))
        handler = _make_event_stream_handler(sink)

        call = ToolCallPart(
            tool_name="context_search", args={"q": "auth"}, tool_call_id="c1"
        )
        ret = ToolReturnPart(
            tool_name="context_search", content={"hits": 2}, tool_call_id="c1"
        )
        asyncio.run(
            _drain(
                handler,
                [
                    FunctionToolCallEvent(part=call),
                    FunctionToolResultEvent(result=ret),
                ],
            )
        )

        kinds = [a["record_type"] for a in log.appended]
        assert kinds == ["tool_call", "tool_result"]
        assert log.appended[0]["payload"]["tool_name"] == "context_search"
        assert log.appended[0]["payload"]["args"] == {"q": "auth"}
        assert log.appended[1]["payload"]["tool_name"] == "context_search"
        # seq continues from the allocator seed (10).
        assert log.appended[0]["seq"] == 11
        assert log.appended[1]["seq"] == 12

    def test_handler_swallows_per_event_errors(self) -> None:
        # A malformed event must not abort the stream.
        log = _RecordingLog()
        sink = _ExecutionLogSink(log, batch_id="b1", seq=_SeqAllocator(0))
        handler = _make_event_stream_handler(sink)
        asyncio.run(_drain(handler, [object()]))  # not a known event type
        assert log.appended == [] and log.parts == []


# --------------------------------------------------------------------------- #
# process_batch resume contract (fake execution log)
# --------------------------------------------------------------------------- #


class _FakeExecutionLog:
    def __init__(self, resume: ResumeState | None) -> None:
        self._resume = resume
        self.records: list[tuple[int, str, dict]] = []
        self.cleared = False

    def load_resume_state(self, batch_id):
        return self._resume

    def append(self, *, batch_id, seq, record_type, payload, event_id=None):
        self.records.append((seq, record_type, payload))

    def clear(self, batch_id):
        self.cleared = True


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


def _batch() -> ReconciliationBatch:
    return ReconciliationBatch(
        id="batch-1",
        pot_id="pot-1",
        status=BATCH_STATUS_PENDING,
        attempt_count=2,
        created_at=_now(),
        claimed_at=None,
        completed_at=None,
        last_error=None,
    )


class _CtxAgent:
    def __init__(self, outcome: BatchAgentOutcome) -> None:
        self.outcome = outcome
        self.seen_ctx = None

    def capability_metadata(self):
        return {"agent": "t", "version": "1", "toolset_version": "t"}

    def run_batch(self, ctx, *, checkpoints=None, execution_log=None):
        self.seen_ctx = ctx
        return self.outcome


class TestResumeContract:
    def _wiring(self, eids):
        batches = MagicMock()
        batches.list_events_for_batch.return_value = [
            BatchEventRef(event_id=eid, added_at=_now()) for eid in eids
        ]
        ledger = MagicMock()
        ledger.get_event_by_id.side_effect = lambda eid: _event_row(eid)
        ledger.start_reconciliation_run.side_effect = lambda eid, **_: f"run-{eid}"
        ledger.next_attempt_number.return_value = 1
        return batches, ledger

    def test_resume_restores_history_and_skips_done_events(self) -> None:
        batches, ledger = self._wiring(["e1", "e2", "e3"])
        resume = ResumeState(
            batch_id="batch-1",
            messages_json=[{"prior": True}],
            tool_call_count=4,
            completed_event_ids=["e1"],  # already finished pre-crash
            last_seq=12,
            chunk_index=0,
        )
        log = _FakeExecutionLog(resume)
        agent = _CtxAgent(
            BatchAgentOutcome(ok=True, completed_event_ids=["e2", "e3"], last_seq=20)
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
        # First chunk resumed with the durable message history.
        assert agent.seen_ctx.prior_messages_json == [{"prior": True}]
        # Already-done event was not handed to the agent…
        handed = {e.event_id for e in agent.seen_ctx.events}
        assert handed == {"e2", "e3"}
        # …but is still credited as completed.
        assert sorted(out.completed_event_ids) == ["e1", "e2", "e3"]
        # seq continues past the resume watermark; terminal record emitted.
        seqs = [r[0] for r in log.records]
        assert seqs == sorted(seqs)
        assert seqs[0] > 12
        kinds = [r[1] for r in log.records]
        assert kinds[:2] == ["status", "status"]
        assert "run_started" in kinds
        assert kinds[-1] == "run_finished"
        assert log.cleared is True

    def test_no_resume_runs_fresh_and_keeps_checkpoint_on_failure(self) -> None:
        batches, ledger = self._wiring(["e1"])
        log = _FakeExecutionLog(None)
        agent = _CtxAgent(BatchAgentOutcome(ok=False, error="boom", last_seq=3))
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
        assert agent.seen_ctx.prior_messages_json is None
        # Failure path keeps the resume checkpoint (retry resumes it).
        assert log.cleared is False
        assert [r[1] for r in log.records][-1] == "run_failed"


class TestNoOpExecutionLog:
    def test_noop_replay_yields_single_end(self) -> None:
        out = list(NoOpAgentExecutionLog().replay_and_tail(batch_id="b"))
        assert len(out) == 1 and out[0]["type"] == "end"

"""Unit tests for the backend-agnostic agent run-loop (run_agent_turn).

Uses a fake sink + fake chunk stream so it runs with no Celery/Hatchet/Redis/LLM.
This is the shared logic both the Celery and Hatchet backends drive.
"""

from app.modules.conversations.exceptions import GenerationCancelled
from app.modules.intelligence.agents.runtime.agent_execution_service import (
    run_agent_turn,
    serialize_tool_calls,
)
from app.modules.intelligence.agents.runtime.redis_sink import RedisStreamSink


class FakeSink:
    def __init__(self, cancel_after=None):
        self.events = []
        self.statuses = []
        self._cancel_after = cancel_after  # become cancelled after N is_cancelled() calls
        self._checks = 0

    def emit(self, event_type, payload):
        self.events.append((event_type, payload))

    def set_status(self, status):
        self.statuses.append(status)

    def is_cancelled(self):
        self._checks += 1
        return self._cancel_after is not None and self._checks > self._cancel_after

    def event_types(self):
        return [e[0] for e in self.events]


class FakeToolCall:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


class FakeChunk:
    def __init__(self, message="", citations=None, tool_calls=None, thinking=None):
        self.message = message
        self.citations = citations
        self.tool_calls = tool_calls
        self.thinking = thinking


async def _agen(items):
    for it in items:
        yield it


async def test_emits_start_then_chunks_then_returns_completed():
    sink = FakeSink()
    chunks = [
        FakeChunk(message="hello ", tool_calls=[FakeToolCall({"name": "search"})]),
        FakeChunk(message="world", citations=["c1"]),
    ]
    completed = await run_agent_turn(
        start_payload={"agent_id": "x", "status": "processing", "message": "start"},
        chunk_stream=_agen(chunks),
        sink=sink,
        flush_partial=lambda: None,
    )
    assert completed is True
    # No "end" here — the outer adapter emits the completed-end event.
    assert sink.event_types() == ["start", "chunk", "chunk"]
    first_chunk = sink.events[1][1]
    assert first_chunk["content"] == "hello "
    assert first_chunk["tool_calls_json"] == [{"name": "search"}]
    second_chunk = sink.events[2][1]
    assert second_chunk["citations_json"] == ["c1"]


async def test_chunk_emits_thinking_when_present():
    sink = FakeSink()
    completed = await run_agent_turn(
        start_payload={},
        chunk_stream=_agen([FakeChunk(message="hi", thinking="step 1")]),
        sink=sink,
        flush_partial=lambda: None,
    )
    assert completed is True
    assert sink.events[1][1]["thinking"] == "step 1"


def test_serialize_tool_calls_parses_json_strings():
    raw = '{"call_id": "c1", "tool_name": "search"}'
    assert serialize_tool_calls([raw]) == [
        {"call_id": "c1", "tool_name": "search"}
    ]


async def test_cancellation_mid_stream_flushes_and_emits_cancelled_end():
    sink = FakeSink(cancel_after=1)  # first chunk passes, cancelled before the second
    flushed = []
    completed = await run_agent_turn(
        start_payload={},
        chunk_stream=_agen([FakeChunk("a"), FakeChunk("b")]),
        sink=sink,
        flush_partial=lambda: flushed.append(True),
    )
    assert completed is False
    assert flushed == [True]
    assert sink.event_types() == ["start", "chunk", "end"]
    assert sink.events[-1][1] == {
        "status": "cancelled",
        "message": "Generation cancelled by user",
    }


async def test_cancellation_after_clean_stream_exit_is_treated_as_cancelled():
    """Cooperative cancel can let chunk_stream end cleanly (agent stops yielding)
    rather than raising. The post-loop check must still report cancelled, not
    completed. cancel_after=2 keeps the two per-chunk checks False and trips the
    final post-loop check True."""
    sink = FakeSink(cancel_after=2)
    flushed = []
    completed = await run_agent_turn(
        start_payload={},
        chunk_stream=_agen([FakeChunk("a"), FakeChunk("b")]),
        sink=sink,
        flush_partial=lambda: flushed.append(True),
    )
    assert completed is False
    assert flushed == [True]
    assert sink.event_types() == ["start", "chunk", "chunk", "end"]
    assert sink.events[-1][1] == {
        "status": "cancelled",
        "message": "Generation cancelled by user",
    }


async def test_generation_cancelled_exception_emits_custom_message():
    sink = FakeSink()

    async def raising():
        raise GenerationCancelled()
        yield  # pragma: no cover - makes this an async generator

    completed = await run_agent_turn(
        start_payload={},
        chunk_stream=raising(),
        sink=sink,
        flush_partial=lambda: None,
        cancel_message="Regeneration cancelled by user",
    )
    assert completed is False
    assert sink.events[-1] == (
        "end",
        {"status": "cancelled", "message": "Regeneration cancelled by user"},
    )


async def test_flush_failure_does_not_break_cancellation():
    sink = FakeSink(cancel_after=0)  # cancelled immediately

    def boom():
        raise RuntimeError("flush failed")

    completed = await run_agent_turn(
        start_payload={},
        chunk_stream=_agen([FakeChunk("a")]),
        sink=sink,
        flush_partial=boom,
    )
    assert completed is False
    assert sink.events[-1][0] == "end"


def test_redis_stream_sink_delegates_to_redis_manager():
    class FakeRedis:
        def __init__(self):
            self.calls = []

        def publish_event(self, c, r, t, p):
            self.calls.append(("publish", c, r, t, p))

        def set_task_status(self, c, r, s):
            self.calls.append(("status", c, r, s))

        def check_cancellation(self, c, r):
            self.calls.append(("cancel", c, r))
            return True

    fr = FakeRedis()
    sink = RedisStreamSink(fr, "conv", "run")
    sink.emit("chunk", {"x": 1})
    sink.set_status("running")
    assert sink.is_cancelled() is True
    assert fr.calls == [
        ("publish", "conv", "run", "chunk", {"x": 1}),
        ("status", "conv", "run", "running"),
        ("cancel", "conv", "run"),
    ]

"""Unit tests for the Hatchet agent worker.

Covers both wiring (`run_hatchet_agent_worker` task registration) and the worker
execution path (`run_agent_via_hatchet`, `_resolve_user_email`). All heavy
dependencies — Redis, DB sessions, ConversationService, the shared run-loop — are
faked so these run without infra.
"""

import pytest

from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.agents import hatchet_worker as hw
from app.modules.intelligence.agents.runtime.hatchet_backend import (
    AGENT_RUN_OPERATION_MESSAGE,
    AGENT_RUN_OPERATION_REGENERATE,
    EVENT_AGENT_RUN,
    AgentRunInput,
)


# ──────────────────────────────────────────────────────────────────────────
#   run_hatchet_agent_worker (registration)
# ──────────────────────────────────────────────────────────────────────────


class _FakeWorker:
    def __init__(self, name, slots, workflows):
        self.name = name
        self.slots = slots
        self.workflows = workflows
        self.started = False

    def start(self):
        self.started = True


class _FakeHatchet:
    def __init__(self):
        self.registered = []
        self.worker_obj = None

    def task(self, **kwargs):
        def deco(fn):
            self.registered.append((kwargs, fn))
            return fn

        return deco

    def worker(self, name, slots, workflows):
        self.worker_obj = _FakeWorker(name, slots, workflows)
        return self.worker_obj


def test_worker_registers_agent_task_on_event_and_starts():
    fake = _FakeHatchet()
    hw.run_hatchet_agent_worker(hatchet=fake)

    assert len(fake.registered) == 1
    kwargs, _fn = fake.registered[0]
    assert EVENT_AGENT_RUN in kwargs["on_events"]
    assert kwargs["input_validator"] is AgentRunInput

    assert fake.worker_obj is not None
    assert fake.worker_obj.started is True
    assert len(fake.worker_obj.workflows) == 1


# ──────────────────────────────────────────────────────────────────────────
#   _resolve_user_email (sync helper, has try/except fallback)
# ──────────────────────────────────────────────────────────────────────────


class _FakeUser:
    def __init__(self, email):
        self.email = email


def test_resolve_user_email_returns_email_when_user_found(monkeypatch):
    class _FakeUserService:
        def __init__(self, db):
            pass

        def get_user_by_uid(self, uid):
            return _FakeUser("alice@example.com")

    monkeypatch.setattr(
        "app.modules.users.user_service.UserService", _FakeUserService
    )
    assert hw._resolve_user_email(db=None, user_id="u1") == "alice@example.com"


def test_resolve_user_email_returns_empty_when_user_missing(monkeypatch):
    class _FakeUserService:
        def __init__(self, db):
            pass

        def get_user_by_uid(self, uid):
            return None

    monkeypatch.setattr(
        "app.modules.users.user_service.UserService", _FakeUserService
    )
    assert hw._resolve_user_email(db=None, user_id="u1") == ""


def test_resolve_user_email_returns_empty_on_exception(monkeypatch):
    class _BoomUserService:
        def __init__(self, db):
            pass

        def get_user_by_uid(self, uid):
            raise RuntimeError("db down")

    monkeypatch.setattr(
        "app.modules.users.user_service.UserService", _BoomUserService
    )
    assert hw._resolve_user_email(db=None, user_id="u1") == ""


# ──────────────────────────────────────────────────────────────────────────
#   _resolve_project_id (sync helper, has try/except fallback)
# ──────────────────────────────────────────────────────────────────────────


class _FakeConversation:
    def __init__(self, project_ids):
        self.project_ids = project_ids


class _FakeQuery:
    def __init__(self, result):
        self._result = result

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._result


class _FakeDb:
    def __init__(self, result):
        self._result = result

    def query(self, _model):
        return _FakeQuery(self._result)


def test_resolve_project_id_returns_first_project_when_present():
    db = _FakeDb(_FakeConversation(project_ids=["proj-1", "proj-2"]))
    assert hw._resolve_project_id(db, "c1") == "proj-1"


def test_resolve_project_id_returns_none_when_conversation_missing():
    db = _FakeDb(None)
    assert hw._resolve_project_id(db, "c1") is None


def test_resolve_project_id_returns_none_when_project_ids_empty():
    db = _FakeDb(_FakeConversation(project_ids=[]))
    assert hw._resolve_project_id(db, "c1") is None


def test_resolve_project_id_returns_none_on_exception():
    class _BoomDb:
        def query(self, _model):
            raise RuntimeError("db down")

    assert hw._resolve_project_id(_BoomDb(), "c1") is None


# ──────────────────────────────────────────────────────────────────────────
#   run_agent_via_hatchet (orchestration)
# ──────────────────────────────────────────────────────────────────────────


class _FakeSink:
    def __init__(self):
        self.events = []
        self.statuses = []

    def emit(self, event_type, payload):
        self.events.append((event_type, payload))

    def set_status(self, status):
        self.statuses.append(status)

    def is_cancelled(self):
        return False


class _FakeRedisManager:
    def check_cancellation(self, c, r):
        return False


class _FakeHistoryManager:
    def flush_message_buffer(self, cid, mtype):
        return None


class _FakeConversationService:
    """Records which method was called and returns a sentinel chunk_stream."""

    def __init__(self):
        self.history_manager = _FakeHistoryManager()
        self.store_message_calls = []
        self.regenerate_calls = []

    def store_message(self, *args, **kwargs):
        self.store_message_calls.append((args, kwargs))
        return "stream-sentinel-message"

    def regenerate_last_message_background(self, *args, **kwargs):
        self.regenerate_calls.append((args, kwargs))
        return "stream-sentinel-regenerate"


class _FakeSyncDb:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeAsyncSession:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class _FakeEngine:
    def __init__(self):
        self.disposed = False

    async def dispose(self):
        self.disposed = True


def _install_fakes(
    monkeypatch,
    *,
    run_turn_returns=True,
    run_turn_raises=None,
):
    """Patch all heavy deps and return (sink, service, sync_db, async_session, engine, run_turn_kwargs)."""
    sink = _FakeSink()
    service = _FakeConversationService()
    sync_db = _FakeSyncDb()
    async_session = _FakeAsyncSession()
    engine = _FakeEngine()
    run_turn_kwargs = {}

    monkeypatch.setattr(hw, "RedisStreamManager", lambda: _FakeRedisManager())
    monkeypatch.setattr(hw, "RedisStreamSink", lambda mgr, c, r: sink)
    monkeypatch.setattr(hw, "init_usage_context", lambda: None)
    monkeypatch.setattr(hw, "get_and_clear_usages", lambda: None)
    monkeypatch.setattr(hw, "_resolve_user_email", lambda db, uid: "u@example.com")

    async def _fake_run_agent_turn(**kwargs):
        run_turn_kwargs.update(kwargs)
        if run_turn_raises is not None:
            raise run_turn_raises
        return run_turn_returns

    monkeypatch.setattr(hw, "run_agent_turn", _fake_run_agent_turn)

    monkeypatch.setattr("app.core.database.SessionLocal", lambda: sync_db)
    monkeypatch.setattr(
        "app.core.database.create_celery_async_session",
        lambda: (async_session, engine),
    )
    import app.modules.conversations.conversation.conversation_service as cs_module

    monkeypatch.setattr(
        cs_module.ConversationService, "create", classmethod(lambda cls, **kw: service)
    )

    return sink, service, sync_db, async_session, engine, run_turn_kwargs


def _make_input(operation=AGENT_RUN_OPERATION_MESSAGE, **overrides):
    base = dict(
        conversation_id="c1",
        run_id="r1",
        user_id="u1",
        query="hello",
        agent_id="debugging_agent",
        operation=operation,
    )
    base.update(overrides)
    return AgentRunInput(**base)


async def test_run_agent_via_hatchet_message_op_calls_store_message(monkeypatch):
    sink, service, *_ = _install_fakes(monkeypatch)
    inp = _make_input(operation=AGENT_RUN_OPERATION_MESSAGE, query="hi there")

    completed = await hw.run_agent_via_hatchet(inp)

    assert completed is True
    assert len(service.store_message_calls) == 1
    assert len(service.regenerate_calls) == 0
    # First positional arg is conversation_id, MessageRequest content has the query
    args, kwargs = service.store_message_calls[0]
    assert args[0] == "c1"
    assert args[1].content == "hi there"
    assert kwargs["run_id"] == "r1"
    assert kwargs["stream"] is True


async def test_run_agent_via_hatchet_regenerate_op_calls_regenerate(monkeypatch):
    sink, service, *_ = _install_fakes(monkeypatch)
    inp = _make_input(
        operation=AGENT_RUN_OPERATION_REGENERATE,
        query="",
        node_ids=[{"node_id": "n1", "name": "N"}],
        attachment_ids=["a1"],
    )

    completed = await hw.run_agent_via_hatchet(inp)

    assert completed is True
    assert len(service.regenerate_calls) == 1
    assert len(service.store_message_calls) == 0
    args, kwargs = service.regenerate_calls[0]
    assert args[0] == "c1"
    # node ids are normalized to NodeContext before reaching the service
    assert args[1] == [NodeContext(node_id="n1", name="N")]
    assert args[2] == ["a1"]
    assert kwargs["run_id"] == "r1"


async def test_run_agent_via_hatchet_normalizes_string_node_ids_for_message(
    monkeypatch,
):
    """Plain string node ids from Hatchet payloads must be normalized to
    NodeContext before MessageRequest is built, otherwise validation fails."""
    sink, service, *_ = _install_fakes(monkeypatch)
    inp = _make_input(
        operation=AGENT_RUN_OPERATION_MESSAGE,
        query="hi",
        node_ids=["nodeA", "nodeB"],
    )

    completed = await hw.run_agent_via_hatchet(inp)

    assert completed is True
    args, _ = service.store_message_calls[0]
    message_request = args[1]
    assert message_request.node_ids == [
        NodeContext(node_id="nodeA", name="nodeA"),
        NodeContext(node_id="nodeB", name="nodeB"),
    ]


async def test_run_agent_via_hatchet_completed_emits_end_and_sets_completed_status(
    monkeypatch,
):
    sink, *_ = _install_fakes(monkeypatch, run_turn_returns=True)

    await hw.run_agent_via_hatchet(_make_input())

    assert sink.statuses == ["running", "completed"]
    end_events = [e for e in sink.events if e[0] == "end"]
    assert len(end_events) == 1
    assert end_events[0][1]["status"] == "completed"


async def test_run_agent_via_hatchet_cancelled_sets_cancelled_status(monkeypatch):
    sink, *_ = _install_fakes(monkeypatch, run_turn_returns=False)

    completed = await hw.run_agent_via_hatchet(_make_input())

    assert completed is False
    assert sink.statuses == ["running", "cancelled"]
    # The cancel "end" event is emitted by run_agent_turn (which is faked here),
    # so the outer wrapper must NOT emit a completed-end event on cancel.
    assert [e for e in sink.events if e[0] == "end"] == []


async def test_run_agent_via_hatchet_failure_sets_error_status_and_reraises(
    monkeypatch,
):
    sink, *_ = _install_fakes(
        monkeypatch, run_turn_raises=RuntimeError("agent exploded")
    )

    with pytest.raises(RuntimeError, match="agent exploded"):
        await hw.run_agent_via_hatchet(_make_input())

    assert sink.statuses == ["running", "error"]
    end_events = [e for e in sink.events if e[0] == "end"]
    assert len(end_events) == 1
    assert end_events[0][1]["status"] == "error"


async def test_run_agent_via_hatchet_closes_sessions_in_finally_on_success(
    monkeypatch,
):
    _, _, sync_db, async_session, engine, _ = _install_fakes(monkeypatch)

    await hw.run_agent_via_hatchet(_make_input())

    assert sync_db.closed is True
    assert async_session.closed is True
    assert engine.disposed is True


async def test_run_agent_via_hatchet_closes_sessions_in_finally_on_failure(
    monkeypatch,
):
    _, _, sync_db, async_session, engine, _ = _install_fakes(
        monkeypatch, run_turn_raises=RuntimeError("boom")
    )

    with pytest.raises(RuntimeError):
        await hw.run_agent_via_hatchet(_make_input())

    assert sync_db.closed is True
    assert async_session.closed is True
    assert engine.disposed is True


async def test_run_agent_via_hatchet_uses_correct_cancel_message_per_operation(
    monkeypatch,
):
    """run_agent_turn receives different cancel_message strings for message vs regenerate."""
    _, _, _, _, _, run_turn_kwargs = _install_fakes(monkeypatch)
    await hw.run_agent_via_hatchet(_make_input(operation=AGENT_RUN_OPERATION_MESSAGE))
    assert run_turn_kwargs["cancel_message"] == "Generation cancelled by user"

    _, _, _, _, _, run_turn_kwargs = _install_fakes(monkeypatch)
    await hw.run_agent_via_hatchet(
        _make_input(operation=AGENT_RUN_OPERATION_REGENERATE, query="")
    )
    assert run_turn_kwargs["cancel_message"] == "Regeneration cancelled by user"


async def test_run_agent_via_hatchet_wraps_run_in_logfire_trace_metadata(monkeypatch):
    """Verifies the wrapper enters logfire_trace_metadata with the routing IDs so spans
    inside the run carry user_id/conversation_id/run_id/agent_id/project_id."""
    _install_fakes(monkeypatch)

    captured = {}

    class _FakeContextManager:
        def __enter__(self_):
            return self_

        def __exit__(self_, *args):
            return False

    def _fake_trace(**kwargs):
        captured.update(kwargs)
        return _FakeContextManager()

    monkeypatch.setattr(hw, "logfire_trace_metadata", _fake_trace)

    await hw.run_agent_via_hatchet(_make_input(agent_id="debugging_agent"))

    assert captured["user_id"] == "u1"
    assert captured["conversation_id"] == "c1"
    assert captured["run_id"] == "r1"
    assert captured["agent_id"] == "debugging_agent"
    assert "project_id" in captured  # set, even if None


# ──────────────────────────────────────────────────────────────────────────
#   Defensive cleanup: finally must not UnboundLocalError or leak sessions
#   when setup raises before all DB resources are assigned.
# ──────────────────────────────────────────────────────────────────────────


async def test_run_agent_via_hatchet_no_unbound_local_when_session_local_raises(
    monkeypatch,
):
    """If SessionLocal() raises, the finally block must not crash with
    UnboundLocalError on async_session/engine/sync_db — they're all None."""
    sink, *_ = _install_fakes(monkeypatch)

    def _boom():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr("app.core.database.SessionLocal", _boom)

    with pytest.raises(RuntimeError, match="db unavailable"):
        await hw.run_agent_via_hatchet(_make_input())

    # The error path still fires sink lifecycle events
    assert sink.statuses == ["error"]


async def test_run_agent_via_hatchet_closes_sync_db_when_async_session_setup_raises(
    monkeypatch,
):
    """The reviewer's scenario: create_celery_async_session() raises AFTER sync_db
    is assigned — sync_db must still get closed, not leaked."""
    _, _, sync_db, _, _, _ = _install_fakes(monkeypatch)

    def _boom():
        raise RuntimeError("async session pool exhausted")

    monkeypatch.setattr("app.core.database.create_celery_async_session", _boom)

    with pytest.raises(RuntimeError, match="async session pool exhausted"):
        await hw.run_agent_via_hatchet(_make_input())

    assert sync_db.closed is True

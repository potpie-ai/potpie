"""Tests for the agent-run dispatch helper (celery vs hatchet routing, fail-closed).

Only the hatchet + failure branches are exercised here; the celery branch lazy-imports
the Celery task graph (heavy ML deps) which isn't importable in unit context.
"""

import pytest
from fastapi import HTTPException

from app.modules.conversations.utils import conversation_routing as cr


class _FakeAsyncRedis:
    def __init__(self):
        self.task_ids = []
        self.statuses = []
        self.events = []

    async def set_task_id(self, conversation_id, run_id, task_id):
        self.task_ids.append(task_id)

    async def set_task_status(self, conversation_id, run_id, status):
        self.statuses.append((conversation_id, run_id, status))

    async def publish_event(self, conversation_id, run_id, event_type, payload):
        self.events.append((conversation_id, run_id, event_type, payload))


async def test_dispatch_routes_allowlisted_agent_to_hatchet(monkeypatch):
    monkeypatch.setattr(cr, "select_backend", lambda agent_id: "hatchet")
    captured = {}
    monkeypatch.setattr(
        cr, "enqueue_agent_run", lambda inp, **kw: captured.update(inp=inp)
    )
    redis = _FakeAsyncRedis()

    result = await cr._dispatch_agent_run(
        conversation_id="c1",
        run_id="r1",
        user_id="u1",
        query="q",
        agent_id="debugging_agent",
        node_ids=None,
        attachment_ids=[],
        async_redis_manager=redis,
        local_mode=False,
        tunnel_url=None,
    )

    assert result is None  # hatchet runs have no celery task id
    assert captured["inp"].conversation_id == "c1"
    assert captured["inp"].agent_id == "debugging_agent"
    assert captured["inp"].operation == cr.AGENT_RUN_OPERATION_MESSAGE
    assert redis.task_ids == []  # no celery task id stored for hatchet


async def test_dispatch_routes_regenerate_to_hatchet_for_selected_agent(monkeypatch):
    monkeypatch.setattr(cr, "select_backend", lambda agent_id: "hatchet")
    captured = {}
    monkeypatch.setattr(
        cr, "enqueue_agent_run", lambda inp, **kw: captured.update(inp=inp)
    )
    redis = _FakeAsyncRedis()

    result = await cr._dispatch_agent_run(
        conversation_id="c1",
        run_id="r1",
        user_id="u1",
        query="",
        agent_id="debugging_agent",
        node_ids=[{"node_id": "n1", "name": "Node"}],
        attachment_ids=["a1"],
        async_redis_manager=redis,
        local_mode=False,
        tunnel_url=None,
        operation=cr.AGENT_RUN_OPERATION_REGENERATE,
    )

    assert result is None
    assert captured["inp"].operation == cr.AGENT_RUN_OPERATION_REGENERATE
    assert captured["inp"].query == ""
    assert captured["inp"].node_ids == [{"node_id": "n1", "name": "Node"}]
    assert captured["inp"].attachment_ids == ["a1"]
    assert redis.task_ids == []


async def test_dispatch_fails_closed_with_503_when_hatchet_enqueue_fails(monkeypatch):
    monkeypatch.setattr(cr, "select_backend", lambda agent_id: "hatchet")

    def _boom(inp, **kw):
        raise RuntimeError("hatchet unreachable")

    monkeypatch.setattr(cr, "enqueue_agent_run", _boom)

    redis = _FakeAsyncRedis()
    with pytest.raises(HTTPException) as excinfo:
        await cr._dispatch_agent_run(
            conversation_id="c1",
            run_id="r1",
            user_id="u1",
            query="q",
            agent_id="debugging_agent",
            node_ids=None,
            attachment_ids=[],
            async_redis_manager=redis,
            local_mode=False,
            tunnel_url=None,
        )
    assert excinfo.value.status_code == 503
    # Caller publishes "queued" before dispatch; 503 path must transition to
    # a terminal state so subscribers don't sit on a stranded queued event.
    assert redis.statuses == [("c1", "r1", "error")]
    assert len(redis.events) == 1
    conv_id, run_id, event_type, payload = redis.events[0]
    assert (conv_id, run_id, event_type) == ("c1", "r1", "end")
    assert payload["status"] == "error"


def _patch_store(monkeypatch, conv):
    class _FakeStore:
        def __init__(self, db, async_db):
            pass

        async def get_by_id(self, conversation_id):
            return conv

    monkeypatch.setattr(
        "app.modules.conversations.conversation.conversation_store.ConversationStore",
        _FakeStore,
    )


async def test_resolve_conversation_agent_id_reads_agent_ids(monkeypatch):
    class _FakeConv:
        agent_ids = ["debugging_agent"]

    _patch_store(monkeypatch, _FakeConv())
    assert await cr.resolve_conversation_agent_id("c1", None, None) == "debugging_agent"


async def test_resolve_conversation_agent_id_none_when_missing(monkeypatch):
    _patch_store(monkeypatch, None)
    assert await cr.resolve_conversation_agent_id("c1", None, None) is None

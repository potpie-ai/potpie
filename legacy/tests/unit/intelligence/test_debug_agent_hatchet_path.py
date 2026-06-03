"""Verify debugging_agent + local_mode works through the Hatchet execution path."""

from __future__ import annotations

import asyncio
from contextvars import copy_context
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.modules.conversations.utils import conversation_routing as cr
from app.modules.intelligence.agents.runtime.backend_selection import select_backend
from app.modules.intelligence.agents.runtime.hatchet_backend import AgentRunInput

pytestmark = pytest.mark.unit


class _FakeAsyncRedis:
    def __init__(self):
        self.task_ids = []

    async def set_task_id(self, conversation_id, run_id, task_id):
        self.task_ids.append(task_id)


@pytest.mark.asyncio
async def test_debugging_agent_routes_to_hatchet_with_local_mode(monkeypatch):
    """VS Code local_mode must survive enqueue → Hatchet worker → agent context."""
    monkeypatch.setenv("AGENT_TASK_BACKEND", "hatchet")
    monkeypatch.setenv("HATCHET_AGENT_ALLOWLIST", "debugging_agent")
    assert select_backend("debugging_agent") == "hatchet"

    captured = {}

    def _capture(inp, **kw):
        captured["inp"] = inp

    with patch.object(cr, "select_backend", return_value="hatchet"), patch.object(
        cr, "enqueue_agent_run", side_effect=_capture
    ):
        result = await cr._dispatch_agent_run(
            conversation_id="conv-1",
            run_id="run-1",
            user_id="user-1",
            query="debug this",
            agent_id="debugging_agent",
            node_ids=None,
            attachment_ids=[],
            async_redis_manager=_FakeAsyncRedis(),
            local_mode=True,
            tunnel_url="socket://abc123",
        )

    assert result is None
    inp = captured["inp"]
    assert isinstance(inp, AgentRunInput)
    assert inp.local_mode is True
    assert inp.tunnel_url == "socket://abc123"
    assert inp.agent_id == "debugging_agent"


def test_pydantic_deep_debug_initializes_tunnel_context_in_local_mode():
    """Hatchet runs PydanticDeepDebugAgent which must set ContextVars before tools."""
    from app.modules.intelligence.agents.chat_agent import ChatContext
    from app.modules.intelligence.agents.chat_agents.pydantic_deep_debug_agent import (
        PydanticDeepDebugAgent,
    )
    from app.modules.intelligence.agents.chat_agents.agent_config import AgentConfig
    from app.modules.intelligence.tools.code_changes_manager import (
        _get_local_mode,
        _get_repository,
        _get_user_id,
    )

    ctx = ChatContext(
        project_id="proj-1",
        project_name="valkey-io/valkey",
        curr_agent_id="debugging_agent",
        history=[],
        query="test",
        conversation_id="conv-1",
        user_id="user-1",
        local_mode=True,
        repository="valkey-io/valkey",
        branch="unstable",
    )
    agent = PydanticDeepDebugAgent(MagicMock(), AgentConfig(role="r", goal="g", backstory="b", tasks=[]), [])

    # _set_sandbox_context mutates process-local ContextVars. Run it (and the
    # assertions that read those vars) inside a copied context so the mutations
    # don't leak into other tests on the same worker.
    def _check():
        agent._set_sandbox_context(ctx)
        assert _get_local_mode() is True
        assert _get_user_id() == "user-1"
        assert _get_repository() == "valkey-io/valkey"

    copy_context().run(_check)


def test_socket_sync_rpc_reuses_event_loop_on_same_thread():
    """Hatchet worker must not close the RPC loop between terminal/DAP tool calls."""
    from app.modules.tunnel.socket_service import WorkspaceSocketService

    svc = WorkspaceSocketService()
    loops = []

    async def _fake_execute(self, workspace_id, endpoint, payload, timeout):
        loops.append(asyncio.get_running_loop())
        return {"success": True, "result": {"ok": True}}

    with patch.object(
        WorkspaceSocketService,
        "_execute_tool_call_with_timeout",
        _fake_execute,
    ):
        svc.execute_tool_call_sync("wid1", "/api/terminal/execute", {"command": "echo 1"})
        svc.execute_tool_call_sync("wid1", "/api/terminal/execute", {"command": "echo 2"})

    assert len(loops) == 2
    assert loops[0] is loops[1]
    assert not loops[0].is_closed()

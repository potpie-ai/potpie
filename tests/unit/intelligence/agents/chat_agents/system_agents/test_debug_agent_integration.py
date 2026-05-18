"""Mock integration: DebugAgent wires debugger tools only in local_mode."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.modules.intelligence.agents.chat_agent import ChatContext
from app.modules.intelligence.agents.chat_agents.system_agents.debug_agent import (
    DebugAgent,
)
from app.modules.intelligence.tools.debug_tools.debug_tunnel_utils import (
    route_debug_command,
)


def _capture_get_tools():  # type: ignore[no-untyped-def]
    captured = []

    def get_tools_capture(names, exclude_embedding_tools=False):
        captured.append((list(names), exclude_embedding_tools))
        return [SimpleNamespace(name=n) for n in names]

    return captured, get_tools_capture


def test_debug_agent_build_non_local_skips_debug_tools() -> None:
    captured, get_tools_capture = _capture_get_tools()
    mock_tools = MagicMock()
    mock_tools.get_tools = get_tools_capture

    llm = MagicMock()
    llm.supports_pydantic.return_value = True
    prompts = MagicMock()

    ctx = ChatContext(
        project_id="p1",
        project_name="Pn",
        curr_agent_id="debugging_agent",
        history=[],
        query="why",
        project_status=None,
    )

    with patch(
        "app.modules.intelligence.agents.chat_agents.system_agents.debug_agent.MultiAgentConfig.should_use_multi_agent",
        return_value=False,
    ):
        agent = DebugAgent(llm, mock_tools, prompts)
        agent._build_agent(ctx, local_mode=False)

    assert len(captured) == 1
    assert "debug_start" not in captured[0][0]


def test_debug_agent_build_local_extends_tools_with_debug_bundle() -> None:
    captured, get_tools_capture = _capture_get_tools()
    mock_tools = MagicMock()
    mock_tools.get_tools = get_tools_capture

    llm = MagicMock()
    llm.supports_pydantic.return_value = True
    prompts = MagicMock()

    ctx = ChatContext(
        project_id="p1",
        project_name="Pn",
        curr_agent_id="debugging_agent",
        history=[],
        query="trace bug",
        project_status=None,
    )

    with patch(
        "app.modules.intelligence.agents.chat_agents.system_agents.debug_agent.MultiAgentConfig.should_use_multi_agent",
        return_value=False,
    ):
        agent = DebugAgent(llm, mock_tools, prompts)
        agent._build_agent(ctx, local_mode=True)

    assert len(captured) == 2
    extra_names = captured[1][0]
    expected = [
        "debug_start",
        "debug_stop",
        "debug_set_breakpoints",
        "debug_snapshot",
        "debug_step_into",
        "debug_step_out",
        "debug_step_over",
        "debug_continue",
        "debug_select_frame",
        "debug_list_sessions",
        "execute_terminal_command",
        "search_text",
        "search_files",
    ]
    assert extra_names == expected


def test_mock_debug_session_flow_via_route_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chained Socket.IO payloads: start → breakpoints → snapshot."""
    monkeypatch.setenv("FORCE_TUNNEL", "true")

    mock_ts = MagicMock()
    mock_ts.get_tunnel_url.return_value = "socket://workspace-1"

    payloads = iter(
        [
            {"success": True, "result": {"session_id": "sid-1", "status": "initialized"}},
            {
                "success": True,
                "result": {
                    "file": "app.py",
                    "breakpoints": [{"line": 10, "verified": True}],
                },
            },
            {
                "success": True,
                "result": {
                    "paused_at": {"file": "app.py", "line": 10, "function": "f"},
                    "call_stack": [
                        {"file": "app.py", "line": 10, "function": "f", "frame_id": 1},
                    ],
                    "locals": {"x": "1"},
                    "expression_results": [],
                    "session_id": "sid-1",
                    "status": "paused",
                },
            },
        ]
    )

    def next_sock_payload(*args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            return next(payloads)
        except StopIteration as exc:
            raise AssertionError("unexpected extra socket call") from exc

    with (
        patch(
            "app.modules.intelligence.tools.code_changes_manager._get_tunnel_url",
            return_value=None,
        ),
        patch(
            "app.modules.intelligence.tools.code_changes_manager._get_repository",
            return_value="repo",
        ),
        patch(
            "app.modules.intelligence.tools.code_changes_manager._get_branch",
            return_value="main",
        ),
        patch(
            "app.modules.tunnel.tunnel_service.get_tunnel_service",
            return_value=mock_ts,
        ),
        patch(
            "app.modules.intelligence.tools.debug_tools.debug_tunnel_utils._try_http_local_debug",
            return_value=None,
        ),
        patch(
            "app.modules.intelligence.tools.debug_tools.debug_tunnel_utils._execute_via_socket_full_response",
            side_effect=next_sock_payload,
        ),
    ):
        s1 = route_debug_command(
            "debug_start",
            {"program": "app.py"},
            "u1",
            "c1",
        )
        s2 = route_debug_command(
            "debug_set_breakpoints",
            {"file": "app.py", "lines": [10]},
            "u1",
            "c1",
        )
        s3 = route_debug_command("debug_snapshot", {"wait": True}, "u1", "c1")

    assert "sid-1" in s1
    assert "Breakpoints" in s2 or "verified" in s2.lower()
    assert "Paused at" in s3
    assert "x = 1" in s3

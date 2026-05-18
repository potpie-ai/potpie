"""Per-tool StructuredTool tests — each forwards to route_debug_command with correct operation/payload."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from app.modules.intelligence.tools.debug_tools import create_debug_tools
from app.modules.intelligence.tools.registry.definitions import DEBUG_TOOLS


@pytest.fixture
def tools_by_name():
    mapping = {}
    for t in create_debug_tools():
        mapping[t.name] = t
    return mapping


def _run_debug_tool(tool: Any, raw: Dict[str, Any]) -> str:
    """Execute the tool's sync body with a validated Pydantic input (matches agent binding)."""
    if tool.args_schema is None:
        raise AssertionError(f"{tool.name} missing args_schema")
    model = tool.args_schema.model_validate(raw)
    return tool.func(model)


@pytest.mark.parametrize(
    ("tool_name", "module", "invoke_arg", "expected_payload"),
    [
        (
            "debug_start",
            "app.modules.intelligence.tools.debug_tools.debug_start_tool",
            {"program": "src/app.py"},
            {
                "program": "src/app.py",
                "language": "python",
                "mode": "launch",
                "args": {},
            },
        ),
        (
            "debug_stop",
            "app.modules.intelligence.tools.debug_tools.debug_stop_tool",
            {},
            {},
        ),
        (
            "debug_set_breakpoints",
            "app.modules.intelligence.tools.debug_tools.debug_set_breakpoints_tool",
            {"file": "src/a.py", "lines": [10, 20]},
            {"file": "src/a.py", "lines": [10, 20]},
        ),
        (
            "debug_snapshot",
            "app.modules.intelligence.tools.debug_tools.debug_snapshot_tool",
            {},
            {"wait": True, "timeout": 30.0},
        ),
        (
            "debug_step_into",
            "app.modules.intelligence.tools.debug_tools.debug_step_into_tool",
            {},
            {},
        ),
        (
            "debug_step_out",
            "app.modules.intelligence.tools.debug_tools.debug_step_out_tool",
            {},
            {},
        ),
        (
            "debug_step_over",
            "app.modules.intelligence.tools.debug_tools.debug_step_over_tool",
            {},
            {},
        ),
        (
            "debug_continue",
            "app.modules.intelligence.tools.debug_tools.debug_continue_tool",
            {},
            {},
        ),
        (
            "debug_select_frame",
            "app.modules.intelligence.tools.debug_tools.debug_select_frame_tool",
            {"frame_index": 2},
            {"frame_index": 2},
        ),
        (
            "debug_list_sessions",
            "app.modules.intelligence.tools.debug_tools.debug_list_sessions_tool",
            {},
            {},
        ),
    ],
)
def test_structured_tool_invokes_route_debug_command(
    tools_by_name: Dict[str, Any],
    tool_name: str,
    module: str,
    invoke_arg: Dict[str, Any],
    expected_payload: Dict[str, Any],
) -> None:
    tool = tools_by_name[tool_name]
    with patch(f"{module}.route_debug_command", return_value="ok") as mock_route:
        with patch(f"{module}.get_context_vars", return_value=("user-1", "conv-1")):
            result = _run_debug_tool(tool, invoke_arg)
    assert result == "ok"
    mock_route.assert_called_once()
    args, kw = mock_route.call_args
    assert kw == {}
    op, data, user_id, conv_id = args
    assert op == tool_name
    assert data == expected_payload
    assert user_id == "user-1"
    assert conv_id == "conv-1"


@pytest.mark.parametrize("tool_name", DEBUG_TOOLS)
def test_tool_tunnel_unauthorized_user_returns_message(
    tools_by_name: Dict[str, Any], tool_name: str
) -> None:
    """When user_id from context is None, extension route is unreachable — clear error."""
    tool = tools_by_name[tool_name]
    invoke_arg = _minimal_invoke_args(tool_name)
    module = _tool_module(tool_name)

    # Real route_debug_command; no authenticated user_id from context vars
    with patch(f"{module}.get_context_vars", return_value=(None, None)):
        out = _run_debug_tool(tool, invoke_arg)
    assert "authenticated user" in out.lower() or "❌" in out


@pytest.mark.parametrize("tool_name", DEBUG_TOOLS)
def test_tool_tunnel_missing_returns_message(
    tools_by_name: Dict[str, Any], tool_name: str
) -> None:
    """With user_id present but tunnel URL unavailable, tools surface tunnel error."""
    mock_ts = MagicMock()
    mock_ts.get_tunnel_url.return_value = None
    invoke_arg = _minimal_invoke_args(tool_name)
    module = _tool_module(tool_name)

    with (
        patch(f"{module}.get_context_vars", return_value=("user-1", "conv-1")),
        patch(
            "app.modules.intelligence.tools.code_changes_manager._get_tunnel_url",
            return_value=None,
        ),
        patch(
            "app.modules.intelligence.tools.code_changes_manager._get_repository",
            return_value=None,
        ),
        patch(
            "app.modules.intelligence.tools.code_changes_manager._get_branch",
            return_value=None,
        ),
        patch(
            "app.modules.tunnel.tunnel_service.get_tunnel_service",
            return_value=mock_ts,
        ),
        patch(
            "app.modules.intelligence.tools.debug_tools.debug_tunnel_utils._try_http_local_debug",
            return_value=None,
        ),
    ):
        out = _run_debug_tool(tools_by_name[tool_name], invoke_arg)

    assert "tunnel" in out.lower()


def _tool_module(tool_name: str) -> str:
    return f"app.modules.intelligence.tools.debug_tools.{tool_name}_tool"


def _minimal_invoke_args(tool_name: str) -> Dict[str, Any]:
    if tool_name == "debug_start":
        return {"program": "x.py"}
    if tool_name == "debug_set_breakpoints":
        return {"file": "x.py", "lines": [1]}
    if tool_name == "debug_select_frame":
        return {"frame_index": 0}
    return {}

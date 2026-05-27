"""Unit tests for get_workspace_debug_context tool (A4).

Covers all required scenarios from the task spec:
  1. Success: dispatch returns a populated payload → populated fields, available=True, message=None.
  2. Partial success: only launch_configs populated → other lists empty, available=True.
  3. Tunnel unavailable (no_tunnel) → available=False, message mentions no tunnel.
  4. Unknown route (E4 not yet implemented) → available=False, message states handler missing.
  5. Timeout → available=False, message mentions timeout.
  6. Exception during dispatch → caught, returns available=False with error message.
  7. focus_path forwarded into RPC payload (mock call inspection).
  8. Schema round-trip: build WorkspaceDebugContext with all fields, dump → validate.
  9. Pydantic-level schema: LaunchConfig requires name, type, request; program optional.
  10. Pydantic-level schema: InferredCommand requires label, command, source.
  11. Tool registered in tool_service.py (source check).
  12. "get_workspace_debug_context" appears in DebugAgent's get_tools([...]) list.
"""

from __future__ import annotations

import os

# Set mandatory env vars before any app module is imported.
os.environ.setdefault("POSTGRES_SERVER", "postgresql://test:test@localhost:5432/testdb")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

import pytest
from unittest.mock import patch

from app.modules.intelligence.tools.get_workspace_debug_context_tool import (
    LaunchConfig,
    InferredCommand,
    RecentChange,
    WorkspaceDebugContext,
    get_workspace_debug_context,
    get_workspace_debug_context_tool,
    _MSG_NO_TUNNEL,
    _MSG_TIMEOUT,
    _MSG_UNKNOWN_ROUTE,
    _MSG_TUNNEL_UNREACHABLE,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Patch targets — mirror the _PATCH_TARGET style from test_run_validation_tool.py
# ---------------------------------------------------------------------------

_PATCH_TARGET = (
    "app.modules.intelligence.tools.get_workspace_debug_context_tool.route_workspace_debug_context"
)
_CONTEXT_PATCH = (
    "app.modules.intelligence.tools.get_workspace_debug_context_tool.get_context_vars"
)

# ---------------------------------------------------------------------------
# Fixtures / payload builders
# ---------------------------------------------------------------------------


def _full_payload() -> dict:
    """Return a fully-populated extension response payload."""
    return {
        "launch_configs": [
            {"name": "Python: Flask", "type": "python", "request": "launch", "program": "app.py"},
            {"name": "Node: Attach", "type": "node", "request": "attach", "program": None},
        ],
        "debug_adapters": ["python", "node"],
        "recent_changes": [
            {
                "file": "src/server.py",
                "commit_sha": "abc1234",
                "commit_message": "fix: handle timeout properly",
                "relative_time": "3 hours ago",
            }
        ],
        "related_tests": ["tests/test_server.py", "tests/test_utils.py"],
        "inferred_commands": [
            {
                "label": "pytest",
                "command": "pytest tests/",
                "source": "pyproject.toml",
            }
        ],
    }


def _launch_only_payload() -> dict:
    """Return a partial response — only launch_configs populated."""
    return {
        "launch_configs": [
            {"name": "Run App", "type": "go", "request": "launch", "program": "main.go"},
        ],
    }


# ---------------------------------------------------------------------------
# 1. Success: fully populated payload → available=True, all fields populated
# ---------------------------------------------------------------------------


def test_success_available_true():
    with patch(_PATCH_TARGET, return_value=(_full_payload(), None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    assert result["available"] is True


def test_success_message_is_none():
    with patch(_PATCH_TARGET, return_value=(_full_payload(), None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    assert result["message"] is None


def test_success_launch_configs_populated():
    with patch(_PATCH_TARGET, return_value=(_full_payload(), None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    assert len(result["launch_configs"]) == 2
    names = [lc["name"] for lc in result["launch_configs"]]
    assert "Python: Flask" in names
    assert "Node: Attach" in names


def test_success_debug_adapters_populated():
    with patch(_PATCH_TARGET, return_value=(_full_payload(), None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    assert result["debug_adapters"] == ["python", "node"]


def test_success_recent_changes_populated():
    with patch(_PATCH_TARGET, return_value=(_full_payload(), None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    assert len(result["recent_changes"]) == 1
    assert result["recent_changes"][0]["file"] == "src/server.py"
    assert result["recent_changes"][0]["commit_sha"] == "abc1234"
    assert result["recent_changes"][0]["relative_time"] == "3 hours ago"


def test_success_related_tests_populated():
    with patch(_PATCH_TARGET, return_value=(_full_payload(), None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    assert "tests/test_server.py" in result["related_tests"]


def test_success_inferred_commands_populated():
    with patch(_PATCH_TARGET, return_value=(_full_payload(), None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    assert len(result["inferred_commands"]) == 1
    cmd = result["inferred_commands"][0]
    assert cmd["label"] == "pytest"
    assert cmd["command"] == "pytest tests/"
    assert cmd["source"] == "pyproject.toml"


# ---------------------------------------------------------------------------
# 2. Partial success: only launch_configs in payload → other lists empty
# ---------------------------------------------------------------------------


def test_partial_success_available_true():
    with patch(_PATCH_TARGET, return_value=(_launch_only_payload(), None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    assert result["available"] is True


def test_partial_success_launch_configs_has_data():
    with patch(_PATCH_TARGET, return_value=(_launch_only_payload(), None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    assert len(result["launch_configs"]) == 1
    assert result["launch_configs"][0]["name"] == "Run App"


def test_partial_success_other_lists_empty():
    with patch(_PATCH_TARGET, return_value=(_launch_only_payload(), None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    assert result["debug_adapters"] == []
    assert result["recent_changes"] == []
    assert result["related_tests"] == []
    assert result["inferred_commands"] == []


def test_partial_success_message_is_none():
    with patch(_PATCH_TARGET, return_value=(_launch_only_payload(), None)), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    assert result["message"] is None


# ---------------------------------------------------------------------------
# 3. Tunnel unavailable: dispatch returns (None, "no_tunnel")
# ---------------------------------------------------------------------------


def test_no_tunnel_available_false():
    with patch(_PATCH_TARGET, return_value=(None, "no_tunnel")), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    assert result["available"] is False


def test_no_tunnel_message_mentions_no_tunnel():
    with patch(_PATCH_TARGET, return_value=(None, "no_tunnel")), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    msg = result["message"] or ""
    # Should say something about the extension not being connected
    assert "not connected" in msg.lower() or "no tunnel" in msg.lower() or "extension" in msg.lower()


def test_no_tunnel_all_lists_empty():
    with patch(_PATCH_TARGET, return_value=(None, "no_tunnel")), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    assert result["launch_configs"] == []
    assert result["debug_adapters"] == []
    assert result["recent_changes"] == []
    assert result["related_tests"] == []
    assert result["inferred_commands"] == []


# ---------------------------------------------------------------------------
# 4. Unknown route — extension has not implemented E4 yet
# ---------------------------------------------------------------------------


def test_unknown_route_available_false():
    with patch(_PATCH_TARGET, return_value=(None, "unknown_route")), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    assert result["available"] is False


def test_unknown_route_message_states_handler_missing():
    with patch(_PATCH_TARGET, return_value=(None, "unknown_route")), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    msg = result["message"] or ""
    # Must clearly indicate the handler is not yet implemented on the extension
    assert (
        "not yet implemented" in msg.lower()
        or "e4" in msg.lower()
        or "handler" in msg.lower()
        or "pending" in msg.lower()
    )


def test_unknown_route_all_lists_empty():
    with patch(_PATCH_TARGET, return_value=(None, "unknown_route")), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    assert result["launch_configs"] == []
    assert result["recent_changes"] == []
    assert result["related_tests"] == []


# ---------------------------------------------------------------------------
# 5. Timeout: dispatch returns (None, "timeout")
# ---------------------------------------------------------------------------


def test_timeout_available_false():
    with patch(_PATCH_TARGET, return_value=(None, "timeout")), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    assert result["available"] is False


def test_timeout_message_mentions_timeout():
    with patch(_PATCH_TARGET, return_value=(None, "timeout")), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    msg = result["message"] or ""
    assert "timeout" in msg.lower() or "timed out" in msg.lower()


def test_timeout_all_lists_empty():
    with patch(_PATCH_TARGET, return_value=(None, "timeout")), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    assert result["launch_configs"] == []
    assert result["inferred_commands"] == []


# ---------------------------------------------------------------------------
# 6. Exception during dispatch → caught, available=False with error message
# ---------------------------------------------------------------------------


def test_exception_available_false():
    with patch(_PATCH_TARGET, side_effect=RuntimeError("socket exploded")), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    assert result["available"] is False


def test_exception_message_contains_error_text():
    with patch(_PATCH_TARGET, side_effect=RuntimeError("socket exploded")), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    msg = result["message"] or ""
    assert "socket exploded" in msg


def test_exception_does_not_raise():
    """The tool must never propagate exceptions to the agent."""
    with patch(_PATCH_TARGET, side_effect=Exception("catastrophic failure")), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        # Should not raise
        result = get_workspace_debug_context()

    assert result["available"] is False


# ---------------------------------------------------------------------------
# 7. focus_path forwarded into the RPC payload
# ---------------------------------------------------------------------------


def test_focus_path_forwarded_to_dispatch():
    with patch(_PATCH_TARGET, return_value=(None, "no_tunnel")) as mock_rpc, \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        get_workspace_debug_context(focus_path="src/payments/service.py")

    mock_rpc.assert_called_once()
    kwargs = mock_rpc.call_args
    # focus_path must appear in the call (either positional or keyword)
    all_args = list(kwargs.args) + list(kwargs.kwargs.values())
    assert "src/payments/service.py" in all_args or kwargs.kwargs.get("focus_path") == "src/payments/service.py"


def test_focus_path_none_when_not_provided():
    with patch(_PATCH_TARGET, return_value=(None, "no_tunnel")) as mock_rpc, \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        get_workspace_debug_context()

    mock_rpc.assert_called_once()
    # focus_path should be None (default)
    kwargs = mock_rpc.call_args.kwargs
    assert kwargs.get("focus_path") is None or "focus_path" not in kwargs


def test_focus_path_forwarded_on_success():
    """focus_path must be forwarded regardless of dispatch outcome."""
    with patch(_PATCH_TARGET, return_value=(_full_payload(), None)) as mock_rpc, \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        get_workspace_debug_context(focus_path="app/models.py")

    mock_rpc.assert_called_once()
    kwargs = mock_rpc.call_args.kwargs
    assert kwargs.get("focus_path") == "app/models.py"


# ---------------------------------------------------------------------------
# 8. Schema round-trip
# ---------------------------------------------------------------------------


def test_schema_round_trip_full():
    ctx = WorkspaceDebugContext(
        launch_configs=[
            LaunchConfig(name="Flask Debug", type="python", request="launch", program="run.py"),
        ],
        debug_adapters=["python", "node", "go"],
        recent_changes=[
            RecentChange(
                file="app/views.py",
                commit_sha="deadbee",
                commit_message="refactor: split view logic",
                relative_time="5 hours ago",
            )
        ],
        related_tests=["tests/test_views.py"],
        inferred_commands=[
            InferredCommand(label="pytest", command="pytest -x tests/", source="pyproject.toml")
        ],
        available=True,
        message=None,
    )
    json_str = ctx.model_dump_json()
    restored = WorkspaceDebugContext.model_validate_json(json_str)

    assert restored.available is True
    assert restored.message is None
    assert len(restored.launch_configs) == 1
    assert restored.launch_configs[0].name == "Flask Debug"
    assert restored.launch_configs[0].program == "run.py"
    assert restored.debug_adapters == ["python", "node", "go"]
    assert len(restored.recent_changes) == 1
    assert restored.recent_changes[0].file == "app/views.py"
    assert restored.recent_changes[0].relative_time == "5 hours ago"
    assert "tests/test_views.py" in restored.related_tests
    assert len(restored.inferred_commands) == 1
    assert restored.inferred_commands[0].source == "pyproject.toml"


def test_schema_round_trip_unavailable():
    ctx = WorkspaceDebugContext(
        available=False,
        message="VS Code extension not connected.",
    )
    json_str = ctx.model_dump_json()
    restored = WorkspaceDebugContext.model_validate_json(json_str)

    assert restored.available is False
    assert restored.message == "VS Code extension not connected."
    assert restored.launch_configs == []
    assert restored.debug_adapters == []
    assert restored.recent_changes == []
    assert restored.related_tests == []
    assert restored.inferred_commands == []


# ---------------------------------------------------------------------------
# 9. Pydantic-level schema: LaunchConfig field requirements
# ---------------------------------------------------------------------------


def test_launch_config_requires_name():
    with pytest.raises(Exception):
        LaunchConfig(type="python", request="launch")  # type: ignore[call-arg]


def test_launch_config_requires_type():
    with pytest.raises(Exception):
        LaunchConfig(name="Debug", request="launch")  # type: ignore[call-arg]


def test_launch_config_requires_request():
    with pytest.raises(Exception):
        LaunchConfig(name="Debug", type="python")  # type: ignore[call-arg]


def test_launch_config_program_is_optional():
    lc = LaunchConfig(name="Debug", type="python", request="launch")
    assert lc.program is None


def test_launch_config_program_can_be_set():
    lc = LaunchConfig(name="Debug", type="python", request="launch", program="main.py")
    assert lc.program == "main.py"


def test_launch_config_valid_minimal():
    lc = LaunchConfig(name="My Launch", type="node", request="attach")
    assert lc.name == "My Launch"
    assert lc.type == "node"
    assert lc.request == "attach"


# ---------------------------------------------------------------------------
# 10. Pydantic-level schema: InferredCommand field requirements
# ---------------------------------------------------------------------------


def test_inferred_command_requires_label():
    with pytest.raises(Exception):
        InferredCommand(command="pytest", source="pyproject.toml")  # type: ignore[call-arg]


def test_inferred_command_requires_command():
    with pytest.raises(Exception):
        InferredCommand(label="pytest", source="pyproject.toml")  # type: ignore[call-arg]


def test_inferred_command_requires_source():
    with pytest.raises(Exception):
        InferredCommand(label="pytest", command="pytest tests/")  # type: ignore[call-arg]


def test_inferred_command_valid():
    ic = InferredCommand(label="npm test", command="npm test", source="package.json scripts.test")
    assert ic.label == "npm test"
    assert ic.command == "npm test"
    assert ic.source == "package.json scripts.test"


# ---------------------------------------------------------------------------
# 11. Tool registered in tool_service.py (source check)
# ---------------------------------------------------------------------------


def test_tool_registered_in_tool_service():
    import pathlib

    tools_dir = (
        pathlib.Path(__file__).parents[3]
        / "app"
        / "modules"
        / "intelligence"
        / "tools"
    )
    source = (tools_dir / "tool_service.py").read_text(encoding="utf-8")

    assert "get_workspace_debug_context_tool" in source, (
        "tool_service.py must import get_workspace_debug_context_tool"
    )
    assert (
        '"get_workspace_debug_context"' in source
        or "'get_workspace_debug_context'" in source
    ), "tool_service.py must register the tool under the key 'get_workspace_debug_context'"


# ---------------------------------------------------------------------------
# 12. DebugAgent tool list contains get_workspace_debug_context
# ---------------------------------------------------------------------------


def test_get_workspace_debug_context_in_debug_agent_tool_list():
    from app.modules.intelligence.agents.chat_agents.system_agents.debug_agent import (
        DEBUG_AGENT_BASE_TOOLS,
        DEBUG_AGENT_DAP_TOOLS,
        DEBUG_AGENT_TERMINAL_TOOLS,
    )

    tool_list = (
        list(DEBUG_AGENT_BASE_TOOLS)
        + list(DEBUG_AGENT_DAP_TOOLS)
        + list(DEBUG_AGENT_TERMINAL_TOOLS)
    )
    assert "get_workspace_debug_context" in tool_list, (
        f"'get_workspace_debug_context' not found in DebugAgent's tool allow-list. "
        f"Found: {tool_list}"
    )


# ---------------------------------------------------------------------------
# Bonus: tunnel_unreachable error type
# ---------------------------------------------------------------------------


def test_tunnel_unreachable_available_false():
    with patch(_PATCH_TARGET, return_value=(None, "tunnel_unreachable")), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    assert result["available"] is False


def test_tunnel_unreachable_message_mentions_tunnel():
    with patch(_PATCH_TARGET, return_value=(None, "tunnel_unreachable")), \
         patch(_CONTEXT_PATCH, return_value=("user1", "conv1")):
        result = get_workspace_debug_context()

    msg = result["message"] or ""
    assert "tunnel" in msg.lower() or "extension" in msg.lower()


# ---------------------------------------------------------------------------
# Bonus: StructuredTool factory sanity check
# ---------------------------------------------------------------------------


def test_tool_factory_returns_structured_tool():
    from langchain_core.tools import StructuredTool

    tool = get_workspace_debug_context_tool()
    assert isinstance(tool, StructuredTool)
    assert tool.name == "get_workspace_debug_context"


def test_tool_factory_has_args_schema():
    from app.modules.intelligence.tools.get_workspace_debug_context_tool import (
        GetWorkspaceDebugContextInput,
    )

    tool = get_workspace_debug_context_tool()
    assert tool.args_schema is GetWorkspaceDebugContextInput


# ---------------------------------------------------------------------------
# Bonus: message constants are non-empty strings (regression guard)
# ---------------------------------------------------------------------------


def test_message_constants_are_non_empty():
    assert _MSG_NO_TUNNEL
    assert _MSG_TIMEOUT
    assert _MSG_UNKNOWN_ROUTE
    assert _MSG_TUNNEL_UNREACHABLE


# ---------------------------------------------------------------------------
# Dispatcher-level: HTTP timeout path returns (None, "timeout")
# ---------------------------------------------------------------------------


def test_route_workspace_debug_context_http_timeout_returns_timeout_code():
    """An httpx.TimeoutException raised during HTTP dispatch must surface as
    (None, "timeout") from route_workspace_debug_context — NOT "unknown_error".

    Strategy:
    - Make get_tunnel_service().get_tunnel_url() return an http:// URL so the
      HTTP branch (not the socket branch) is taken.
    - Make get_tunnel_service().get_workspace_id() return None (unused in this
      path, but avoids AttributeError if accessed).
    - Stub _get_tunnel_url / _get_repository / _get_branch so no real context
      lookups fire.
    - Patch httpx.Client.post to raise httpx.TimeoutException.
    """
    import httpx
    from unittest.mock import MagicMock, patch as _patch

    from app.modules.intelligence.tools.local_search_tools.tunnel_utils import (
        route_workspace_debug_context,
    )

    http_tunnel_url = "http://localhost:49152"

    mock_tunnel_service = MagicMock()
    mock_tunnel_service.get_tunnel_url.return_value = http_tunnel_url
    mock_tunnel_service.get_workspace_id.return_value = None

    with _patch(
        "app.modules.tunnel.tunnel_service.get_tunnel_service",
        return_value=mock_tunnel_service,
    ), _patch(
        "app.modules.intelligence.tools.code_changes_manager._get_tunnel_url",
        return_value=None,
    ), _patch(
        "app.modules.intelligence.tools.code_changes_manager._get_repository",
        return_value=None,
    ), _patch(
        "app.modules.intelligence.tools.code_changes_manager._get_branch",
        return_value=None,
    ), _patch.object(
        httpx.Client,
        "post",
        side_effect=httpx.TimeoutException("timed out"),
    ):
        result, err = route_workspace_debug_context(
            focus_path=None,
            user_id="u1",
            conversation_id="c1",
            timeout=1.0,
        )

    assert result is None
    assert err == "timeout"


# ---------------------------------------------------------------------------
# Dispatcher-level: HTTP ConnectError path returns (None, "tunnel_unreachable")
# ---------------------------------------------------------------------------


def test_route_workspace_debug_context_http_connect_error_returns_tunnel_unreachable():
    """An httpx.ConnectError raised during HTTP dispatch must surface as
    (None, "tunnel_unreachable") from route_workspace_debug_context — NOT "unknown_error".

    Strategy mirrors the HTTP-timeout test directly above:
    - Make get_tunnel_service().get_tunnel_url() return an http:// URL so the
      HTTP branch (not the socket branch) is taken.
    - Patch httpx.Client.post to raise httpx.ConnectError (DNS / refused).
    """
    import httpx
    from unittest.mock import MagicMock, patch as _patch

    from app.modules.intelligence.tools.local_search_tools.tunnel_utils import (
        route_workspace_debug_context,
    )

    http_tunnel_url = "http://localhost:49152"

    mock_tunnel_service = MagicMock()
    mock_tunnel_service.get_tunnel_url.return_value = http_tunnel_url
    mock_tunnel_service.get_workspace_id.return_value = None

    with _patch(
        "app.modules.tunnel.tunnel_service.get_tunnel_service",
        return_value=mock_tunnel_service,
    ), _patch(
        "app.modules.intelligence.tools.code_changes_manager._get_tunnel_url",
        return_value=None,
    ), _patch(
        "app.modules.intelligence.tools.code_changes_manager._get_repository",
        return_value=None,
    ), _patch(
        "app.modules.intelligence.tools.code_changes_manager._get_branch",
        return_value=None,
    ), _patch.object(
        httpx.Client,
        "post",
        side_effect=httpx.ConnectError("dns failure"),
    ):
        result, err = route_workspace_debug_context(
            focus_path=None,
            user_id="u1",
            conversation_id="c1",
            timeout=1.0,
        )

    assert result is None
    assert err == "tunnel_unreachable"

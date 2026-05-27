"""Regression tests for Socket.IO timeout error classification."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("POSTGRES_SERVER", "postgresql://test:test@localhost:5432/testdb")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

pytestmark = pytest.mark.unit


def _socket_tunnel_service() -> MagicMock:
    svc = MagicMock()
    svc.get_tunnel_url.return_value = "socket://workspace-1"
    svc.get_workspace_id.return_value = "workspace-1"
    return svc


def _route_patches(socket_return):
    return (
        patch(
            "app.modules.tunnel.tunnel_service.get_tunnel_service",
            return_value=_socket_tunnel_service(),
        ),
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
            "app.modules.intelligence.tools.local_search_tools.tunnel_utils._execute_via_socket",
            return_value=socket_return,
        ),
    )


def test_execute_via_socket_normalizes_extension_handler_timeout():
    from app.modules.intelligence.tools.local_search_tools.tunnel_utils import (
        _execute_via_socket,
    )

    svc = MagicMock()
    svc.execute_tool_call_with_fallback.return_value = {
        "success": False,
        "error": "Handler timed out after 29500ms",
    }

    with patch(
        "app.modules.tunnel.tunnel_service.get_tunnel_service",
        return_value=svc,
    ):
        result, error_type = _execute_via_socket(
            user_id="u1",
            conversation_id="c1",
            endpoint="/api/debug/start-session",
            payload={},
            tunnel_url="socket://workspace-1",
            timeout=30.0,
            return_error=True,
        )

    assert result is None
    assert error_type == "timeout"


def test_route_terminal_command_maps_handler_timeout_to_timeout():
    from app.modules.intelligence.tools.local_search_tools.tunnel_utils import (
        route_terminal_command,
    )

    patches = _route_patches((None, "Handler timed out after 24500ms"))
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result, error_type = route_terminal_command(
            command="sleep 60",
            timeout=20_000,
            user_id="u1",
            conversation_id="c1",
        )

    assert result is None
    assert error_type == "timeout"


def test_route_workspace_debug_context_maps_handler_timeout_to_timeout():
    from app.modules.intelligence.tools.local_search_tools.tunnel_utils import (
        route_workspace_debug_context,
    )

    patches = _route_patches((None, "Handler timed out after 29500ms"))
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result, error_type = route_workspace_debug_context(
            user_id="u1",
            conversation_id="c1",
        )

    assert result is None
    assert error_type == "timeout"


def test_route_dap_command_maps_handler_timeout_to_timeout():
    from app.modules.intelligence.tools.local_search_tools.tunnel_utils import (
        route_dap_command,
    )

    patches = _route_patches((None, "Handler timed out after 29500ms"))
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result, error_type = route_dap_command(
            method="start_session",
            payload={"program": "a.py"},
            user_id="u1",
            conversation_id="c1",
        )

    assert result is None
    assert error_type == "timeout"

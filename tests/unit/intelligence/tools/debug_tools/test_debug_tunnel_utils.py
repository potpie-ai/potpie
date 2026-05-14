"""Tests for debug tunnel formatting and routing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.modules.intelligence.tools.debug_tools.debug_tunnel_utils import (
    format_debug_result,
    route_debug_command,
)


class TestFormatDebugResult:
    def test_none_result(self) -> None:
        assert "completed" in format_debug_result("debug_continue", None)

    def test_snapshot_like_with_stack_and_locals(self) -> None:
        text = format_debug_result(
            "debug_snapshot",
            {
                "paused_at": {"file": "/proj/foo.py", "line": 10, "function": "bar"},
                "call_stack": [
                    {"file": "/proj/foo.py", "line": 10, "function": "bar", "frame_id": "f1"},
                ],
                "locals": {"x": 1},
                "session_id": "s1",
                "status": "stopped",
            },
        )
        assert "Paused at foo.py:10 in bar()" in text
        assert "Call Stack:" in text
        assert "bar()" in text
        assert "Local Variables:" in text
        assert "x = 1" in text
        assert "session_id: s1" in text

    def test_breakpoints_format(self) -> None:
        text = format_debug_result(
            "debug_set_breakpoints",
            {
                "file": "/proj/app.py",
                "breakpoints": [
                    {"line": 5, "verified": True},
                    {"line": 20, "verified": False, "message": "no code"},
                ],
            },
        )
        assert "Breakpoints in app.py:" in text
        assert "Line 5: verified" in text
        assert "Line 20: not verified" in text

    def test_sessions_empty_and_nonempty(self) -> None:
        empty = format_debug_result("debug_list_sessions", {"sessions": []})
        assert "(none)" in empty

        full = format_debug_result(
            "debug_list_sessions",
            {
                "sessions": [
                    {
                        "session_id": "abc123",
                        "program": "/proj/main.py",
                        "language": "python",
                        "status": "running",
                    },
                ],
            },
        )
        assert "Active debug sessions:" in full
        assert "main.py" in full

    def test_generic_truncates_large_dict(self) -> None:
        big = {f"k{i}": i for i in range(500)}
        text = format_debug_result("debug_start", big)
        assert len(text) <= 8000


class TestRouteDebugCommand:
    def test_unknown_operation(self) -> None:
        out = route_debug_command("debug_nope", {}, "u1", "c1")
        assert "Unknown debug operation" in out

    def test_requires_user_id(self) -> None:
        out = route_debug_command("debug_continue", {}, None, "c1")
        assert "authenticated user" in out.lower()

    @pytest.mark.parametrize(
        ("operation", "minimal_payload"),
        [
            ("debug_start", {"program": "m.py"}),
            ("debug_stop", {}),
            ("debug_set_breakpoints", {"file": "m.py", "lines": [1]}),
            ("debug_snapshot", {}),
            ("debug_step_into", {}),
            ("debug_step_out", {}),
            ("debug_step_over", {}),
            ("debug_continue", {}),
            ("debug_select_frame", {"frame_index": 0}),
            ("debug_list_sessions", {}),
        ],
    )
    def test_no_tunnel_url_every_operation(
        self, operation: str, minimal_payload: dict
    ) -> None:
        mock_ts = MagicMock()
        mock_ts.get_tunnel_url.return_value = None
        with (
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
            out = route_debug_command(
                operation, minimal_payload, "user-1", "conv-1"
            )
        assert "tunnel" in out.lower()

    def test_socket_success_formats_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORCE_TUNNEL", "true")
        mock_ts = MagicMock()
        mock_ts.get_tunnel_url.return_value = "socket://workspace-1"

        sock_payload = {
            "success": True,
            "result": {"session_id": "sid", "status": "started"},
        }

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
                return_value=sock_payload,
            ),
        ):
            out = route_debug_command("debug_start", {"program": "main.py"}, "u1", "c1")

        assert "session_id: sid" in out
        assert "status: started" in out

    def test_socket_extension_error_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORCE_TUNNEL", "true")
        mock_ts = MagicMock()
        mock_ts.get_tunnel_url.return_value = "socket://workspace-1"

        with (
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
            patch(
                "app.modules.intelligence.tools.debug_tools.debug_tunnel_utils._execute_via_socket_full_response",
                return_value={"success": False, "error": "no active session"},
            ),
        ):
            out = route_debug_command("debug_continue", {}, "u1", "c1")

        assert "failed" in out.lower()
        assert "no active session" in out

    def test_legacy_http_tunnel_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORCE_TUNNEL", "true")
        mock_ts = MagicMock()
        mock_ts.get_tunnel_url.return_value = "http://127.0.0.1:7777/tool"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "result": {"sessions": []},
        }
        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response

        mock_http_class = MagicMock()
        mock_http_class.return_value.__enter__.return_value = mock_client_instance

        with (
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
            patch(
                "app.modules.intelligence.tools.debug_tools.debug_tunnel_utils.httpx.Client",
                mock_http_class,
            ),
        ):
            out = route_debug_command("debug_list_sessions", {}, "u1", "c1")

        assert "(none)" in out or "sessions" in out.lower()
        mock_client_instance.post.assert_called_once()

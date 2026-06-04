"""Unit tests for the DAP tool family (A3).

Coverage:
  - A3.1: Schema round-trips, optional-field defaults, Literal enum rejection
  - A3.2: route_dap_command — success, HTTP timeout, HTTP ConnectError
  - A3.3: Each of the 10 tools — success path, no_tunnel failure, unknown_route failure,
           and one arg-forwarding assertion (method= and payload=)
  - The 4 step_* tools are parametrized together (same shape)
"""

from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_SERVER", "postgresql://test:test@localhost:5432/testdb")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

import pytest
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_DISPATCH = "app.modules.intelligence.tools.dap_tools.route_dap_command"
_CTX = "app.modules.intelligence.tools.dap_tools.get_context_vars"
_CTX_RETURN = ("user1", "conv1")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snapshot_payload(**extra) -> dict:
    return {
        "session_id": "sess-abc",
        "status": "paused",
        "paused_at": {"file": "/app/main.py", "line": 10, "function": "run"},
        "call_stack": [
            {"frame_id": 1, "function": "run", "file": "/app/main.py", "line": 10}
        ],
        "locals": {"x": "42"},
        "expression_results": [],
        **extra,
    }


# ===========================================================================
# A3.1 — Schema tests
# ===========================================================================


class TestDebugCallFrameSchema:
    def test_round_trip(self):
        from app.modules.intelligence.tools.dap_schemas import DebugCallFrame

        frame = DebugCallFrame(frame_id=1, name="main", source_path="/a.py", line=5, column=3)
        restored = DebugCallFrame.model_validate_json(frame.model_dump_json())
        assert restored.frame_id == 1
        assert restored.name == "main"
        assert restored.source_path == "/a.py"
        assert restored.line == 5
        assert restored.column == 3

    def test_optional_fields_default_none(self):
        from app.modules.intelligence.tools.dap_schemas import DebugCallFrame

        frame = DebugCallFrame(frame_id=0, name="<top>")
        assert frame.source_path is None
        assert frame.line is None
        assert frame.column is None

    def test_debug_call_frame_accepts_ts_field_names_via_aliases(self):
        from app.modules.intelligence.tools.dap_schemas import DebugCallFrame

        frame = DebugCallFrame.model_validate({
            "frame_id": 1,
            "function": "createOrder",
            "file": "/src/checkout/createOrder.ts",
            "line": 88,
        })
        assert frame.name == "createOrder"
        assert frame.source_path == "/src/checkout/createOrder.ts"

    def test_debug_call_frame_accepts_python_field_names(self):
        from app.modules.intelligence.tools.dap_schemas import DebugCallFrame

        frame = DebugCallFrame(
            frame_id=1,
            name="createOrder",
            source_path="/src/checkout/createOrder.ts",
            line=88,
        )
        assert frame.name == "createOrder"


class TestDebugSnapshotSchema:
    def test_round_trip_full(self):
        from app.modules.intelligence.tools.dap_schemas import DebugSnapshot, DebugCallFrame

        snap = DebugSnapshot(
            session_id="s1",
            status="paused",
            paused_at={"file": "/a.py", "line": 1, "function": "f"},
            call_stack=[DebugCallFrame(frame_id=1, name="f", source_path="/a.py", line=1)],
            locals={"x": "1"},
            expression_results=[{"expression": "x+1", "result": "2"}],
            output="hello",
        )
        restored = DebugSnapshot.model_validate_json(snap.model_dump_json())
        assert restored.session_id == "s1"
        assert restored.status == "paused"
        assert restored.locals == {"x": "1"}
        assert len(restored.call_stack) == 1
        assert restored.output == "hello"

    def test_defaults(self):
        from app.modules.intelligence.tools.dap_schemas import DebugSnapshot

        snap = DebugSnapshot(session_id="s", status="running")
        assert snap.call_stack == []
        assert snap.locals == {}
        assert snap.expression_results == []
        assert snap.paused_at is None
        assert snap.output is None


class TestStartSessionResultSchema:
    def test_round_trip(self):
        from app.modules.intelligence.tools.dap_schemas import StartSessionResult

        r = StartSessionResult(session_id="s1", program="/a.py", language="python", status="initialized")
        restored = StartSessionResult.model_validate_json(r.model_dump_json())
        assert restored.session_id == "s1"
        assert restored.language == "python"

    def test_optional_fields_default_none(self):
        from app.modules.intelligence.tools.dap_schemas import StartSessionResult

        r = StartSessionResult(session_id="s1", status="failed")
        assert r.program is None
        assert r.language is None
        assert r.message is None


class TestStartDebugSessionInputSchema:
    def test_args_is_launch_config_object(self):
        from app.modules.intelligence.tools.dap_tools import StartDebugSessionInput

        inp = StartDebugSessionInput(
            program="/a.py",
            args={"args": ["--flag"], "stopOnEntry": True},
        )
        assert inp.args == {"args": ["--flag"], "stopOnEntry": True}

    def test_args_rejects_raw_list_in_tool_schema(self):
        from app.modules.intelligence.tools.dap_tools import StartDebugSessionInput

        with pytest.raises(ValidationError):
            StartDebugSessionInput(program="/a.py", args=["--flag"])  # type: ignore[arg-type]

    @pytest.mark.parametrize("lang", ["c", "cpp", "c++", "lldb", "cppdbg", "lldb-dap"])
    def test_native_language_values_accepted(self, lang):
        from app.modules.intelligence.tools.dap_tools import StartDebugSessionInput

        inp = StartDebugSessionInput(program="/bin/a.out", language=lang)
        assert inp.language == lang


class TestSetBreakpointsResultSchema:
    def test_round_trip(self):
        from app.modules.intelligence.tools.dap_schemas import SetBreakpointsResult

        r = SetBreakpointsResult(
            session_id="s1",
            file="/a.py",
            breakpoints=[{"line": 10, "verified": True}],
        )
        restored = SetBreakpointsResult.model_validate_json(r.model_dump_json())
        assert restored.file == "/a.py"
        assert restored.breakpoints[0]["verified"] is True

    def test_optional_fields_default(self):
        from app.modules.intelligence.tools.dap_schemas import SetBreakpointsResult

        r = SetBreakpointsResult(file="/b.py")
        assert r.breakpoints == []
        assert r.session_id is None
        assert r.message is None


class TestTrackedDebugSessionSchema:
    def test_round_trip(self):
        from app.modules.intelligence.tools.dap_schemas import TrackedDebugSession

        s = TrackedDebugSession(session_id="s1", program="/a.py", language="python", status="paused")
        restored = TrackedDebugSession.model_validate_json(s.model_dump_json())
        assert restored.status == "paused"

    def test_optional_fields_default_none(self):
        from app.modules.intelligence.tools.dap_schemas import TrackedDebugSession

        s = TrackedDebugSession(session_id="s1", status="running")
        assert s.program is None
        assert s.language is None
        assert s.created_at is None

    def test_tracked_debug_session_accepts_ts_field_names_via_alias(self):
        from app.modules.intelligence.tools.dap_schemas import TrackedDebugSession

        s = TrackedDebugSession.model_validate({
            "session_id": "sess_1",
            "status": "running",
            "createdAt": "2026-05-19T10:00:00Z",
        })
        assert s.created_at == "2026-05-19T10:00:00Z"

    def test_tracked_debug_session_accepts_python_field_name(self):
        from app.modules.intelligence.tools.dap_schemas import TrackedDebugSession

        s = TrackedDebugSession(session_id="sess_1", status="running", created_at="2026-05-19T10:00:00Z")
        assert s.created_at == "2026-05-19T10:00:00Z"


class TestListSessionsResultSchema:
    def test_round_trip_empty(self):
        from app.modules.intelligence.tools.dap_schemas import ListSessionsResult

        r = ListSessionsResult()
        restored = ListSessionsResult.model_validate_json(r.model_dump_json())
        assert restored.sessions == []

    def test_round_trip_with_sessions(self):
        from app.modules.intelligence.tools.dap_schemas import ListSessionsResult, TrackedDebugSession

        r = ListSessionsResult(
            sessions=[TrackedDebugSession(session_id="s1", status="paused")]
        )
        restored = ListSessionsResult.model_validate_json(r.model_dump_json())
        assert len(restored.sessions) == 1
        assert restored.sessions[0].session_id == "s1"


class TestEvaluateResultSchema:
    def test_round_trip(self):
        from app.modules.intelligence.tools.dap_schemas import EvaluateResult

        r = EvaluateResult(session_id="s1", expression="x+1", value="42", type="int")
        restored = EvaluateResult.model_validate_json(r.model_dump_json())
        assert restored.value == "42"
        assert restored.type == "int"

    def test_type_optional(self):
        from app.modules.intelligence.tools.dap_schemas import EvaluateResult

        r = EvaluateResult(session_id="s1", expression="x", value="1")
        assert r.type is None


class TestStopSessionResultSchema:
    def test_round_trip_terminated(self):
        from app.modules.intelligence.tools.dap_schemas import StopSessionResult

        r = StopSessionResult(session_id="s1", status="terminated")
        restored = StopSessionResult.model_validate_json(r.model_dump_json())
        assert restored.status == "terminated"

    def test_invalid_status_raises(self):
        from app.modules.intelligence.tools.dap_schemas import StopSessionResult

        with pytest.raises(ValidationError):
            StopSessionResult(session_id="s1", status="running")  # type: ignore

    def test_message_optional(self):
        from app.modules.intelligence.tools.dap_schemas import StopSessionResult

        r = StopSessionResult(session_id="s1", status="not_found")
        assert r.message is None


class TestDapErrorSchema:
    def test_round_trip(self):
        from app.modules.intelligence.tools.dap_schemas import DapError

        e = DapError(available=False, error="no_tunnel", error_type="no_tunnel", message="not connected")
        restored = DapError.model_validate_json(e.model_dump_json())
        assert restored.error_type == "no_tunnel"
        assert restored.available is False

    def test_available_default_false(self):
        from app.modules.intelligence.tools.dap_schemas import DapError

        e = DapError(error="x", error_type="unknown_error", message="oops")
        assert e.available is False

    def test_invalid_error_type_raises(self):
        from app.modules.intelligence.tools.dap_schemas import DapError

        with pytest.raises(ValidationError):
            DapError(error="x", error_type="bad_type", message="oops")  # type: ignore

    @pytest.mark.parametrize("etype", [
        "no_tunnel", "unknown_route", "timeout",
        "tunnel_unreachable", "backend_socket_error", "extension_error",
        "no_user_id", "unknown_error",
    ])
    def test_all_valid_error_types(self, etype):
        from app.modules.intelligence.tools.dap_schemas import DapError

        e = DapError(error=etype, error_type=etype, message="msg")
        assert e.error_type == etype

    def test_no_session_is_rejected(self):
        from app.modules.intelligence.tools.dap_schemas import DapError

        with pytest.raises(ValidationError):
            DapError(error="no_session", error_type="no_session", message="msg")  # type: ignore


class TestMakeDapError:
    def test_dap_error_no_user_id_message_distinct_from_no_tunnel(self):
        from app.modules.intelligence.tools.dap_tools import _make_dap_error, _NO_TUNNEL_MSG

        err = _make_dap_error(error_type="no_user_id", context="test").model_dump(mode="json")
        assert err["error_type"] == "no_user_id"
        assert "user" in err["message"].lower()
        assert err["message"] != _NO_TUNNEL_MSG

    def test_dap_error_backend_loop_mismatch_not_reported_as_tunnel_drop(self):
        from app.modules.intelligence.tools.dap_tools import _make_dap_error

        err = _make_dap_error(error_type="backend_loop_mismatch", context="snapshot").model_dump(mode="json")

        assert err["error"] == "backend_loop_mismatch"
        assert err["error_type"] == "backend_socket_error"
        assert "not evidence" in err["message"]
        assert "disconnected" in err["message"]

    def test_dap_error_extension_error_not_reported_as_tunnel_drop(self):
        from app.modules.intelligence.tools.dap_tools import _make_dap_error

        err = _make_dap_error(error_type="debug adapter not available", context="start_session").model_dump(mode="json")

        assert err["error"] == "debug adapter not available"
        assert err["error_type"] == "extension_error"
        assert "debug adapter" in err["message"]
        assert "NOT a tunnel/connection problem" in err["message"]
        assert "Reconnect and retry" not in err["message"]

    def test_dap_error_timeout_not_reported_as_tunnel_drop(self):
        from app.modules.intelligence.tools.dap_tools import _make_dap_error

        err = _make_dap_error(error_type="timeout", context="start_session").model_dump(mode="json")

        assert err["error_type"] == "timeout"
        assert "not evidence" in err["message"]
        assert "Reconnect and retry" not in err["message"]

    def test_dap_error_debug_adapter_unavailable_includes_install_guidance(self):
        from app.modules.intelligence.tools.dap_tools import _make_dap_error

        err = _make_dap_error(
            error_type="debug_adapter_unavailable: No registered debug adapter",
            context="start_session",
        ).model_dump(mode="json")

        assert err["error_type"] == "extension_error"
        assert "not installed" in err["message"].lower()


# ===========================================================================
# A3.2 — route_dap_command dispatcher tests
# ===========================================================================


class TestRouteDapCommand:
    """Tests for the generic HTTP path of route_dap_command.

    Uses the same mocking strategy as test_get_workspace_debug_context_tool.py.
    """

    _http_tunnel_url = "http://localhost:49152"

    def _mock_tunnel_service(self):
        svc = MagicMock()
        svc.get_tunnel_url.return_value = self._http_tunnel_url
        svc.get_workspace_id.return_value = None
        return svc

    def test_http_success_returns_result(self):
        import httpx
        from unittest.mock import patch as _patch, MagicMock
        from app.modules.intelligence.tools.local_search_tools.tunnel_utils import route_dap_command

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"session_id": "s1", "status": "paused"}

        with _patch(
            "app.modules.tunnel.tunnel_service.get_tunnel_service",
            return_value=self._mock_tunnel_service(),
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
            return_value=mock_resp,
        ):
            result, err = route_dap_command(
                method="snapshot",
                payload={},
                user_id="u1",
                conversation_id="c1",
                timeout=5.0,
            )

        assert err is None
        assert result is not None
        assert result["session_id"] == "s1"

    def test_http_timeout_returns_timeout_code(self):
        import httpx
        from unittest.mock import patch as _patch
        from app.modules.intelligence.tools.local_search_tools.tunnel_utils import route_dap_command

        with _patch(
            "app.modules.tunnel.tunnel_service.get_tunnel_service",
            return_value=self._mock_tunnel_service(),
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
            result, err = route_dap_command(
                method="start_session",
                payload={"program": "a.py"},
                user_id="u1",
                conversation_id="c1",
                timeout=1.0,
            )

        assert result is None
        assert err == "timeout"

    def test_http_connect_error_returns_tunnel_unreachable(self):
        import httpx
        from unittest.mock import patch as _patch
        from app.modules.intelligence.tools.local_search_tools.tunnel_utils import route_dap_command

        with _patch(
            "app.modules.tunnel.tunnel_service.get_tunnel_service",
            return_value=self._mock_tunnel_service(),
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
            result, err = route_dap_command(
                method="list_sessions",
                payload={},
                user_id="u1",
                conversation_id="c1",
                timeout=1.0,
            )

        assert result is None
        assert err == "tunnel_unreachable"

    def test_no_user_id_returns_no_user_id(self):
        from app.modules.intelligence.tools.local_search_tools.tunnel_utils import route_dap_command

        result, err = route_dap_command(method="snapshot", payload={}, user_id="")
        assert result is None
        assert err == "no_user_id"

    def test_no_tunnel_returns_no_tunnel(self):
        from unittest.mock import patch as _patch
        from app.modules.intelligence.tools.local_search_tools.tunnel_utils import route_dap_command

        mock_svc = MagicMock()
        mock_svc.get_tunnel_url.return_value = None
        mock_svc.get_workspace_id.return_value = None

        with _patch(
            "app.modules.tunnel.tunnel_service.get_tunnel_service",
            return_value=mock_svc,
        ), _patch(
            "app.modules.intelligence.tools.code_changes_manager._get_tunnel_url",
            return_value=None,
        ), _patch(
            "app.modules.intelligence.tools.code_changes_manager._get_repository",
            return_value=None,
        ), _patch(
            "app.modules.intelligence.tools.code_changes_manager._get_branch",
            return_value=None,
        ):
            result, err = route_dap_command(method="snapshot", payload={}, user_id="u1")

        assert result is None
        assert err == "no_tunnel"

    def test_http_404_returns_unknown_route(self):
        import httpx
        from unittest.mock import patch as _patch, MagicMock
        from app.modules.intelligence.tools.local_search_tools.tunnel_utils import route_dap_command

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not Found"

        with _patch(
            "app.modules.tunnel.tunnel_service.get_tunnel_service",
            return_value=self._mock_tunnel_service(),
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
            return_value=mock_resp,
        ):
            result, err = route_dap_command(
                method="snapshot",
                payload={},
                user_id="u1",
                conversation_id="c1",
                timeout=5.0,
            )

        assert result is None
        assert err == "unknown_route"

    def test_http_200_domain_failure_returns_extension_error(self):
        import httpx
        from unittest.mock import patch as _patch, MagicMock
        from app.modules.intelligence.tools.local_search_tools.tunnel_utils import route_dap_command

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "success": False,
            "error": "debug_adapter_unavailable",
        }

        with _patch(
            "app.modules.tunnel.tunnel_service.get_tunnel_service",
            return_value=self._mock_tunnel_service(),
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
            return_value=mock_resp,
        ):
            result, err = route_dap_command(
                method="start_session",
                payload={"program": "a.py"},
                user_id="u1",
                conversation_id="c1",
            )

        assert result is None
        assert err == "debug_adapter_unavailable"

    def test_http_422_returns_domain_error_not_tunnel_unreachable(self):
        import httpx
        from unittest.mock import patch as _patch, MagicMock
        from app.modules.intelligence.tools.local_search_tools.tunnel_utils import route_dap_command

        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.json.return_value = {
            "error": "debug_adapter_unavailable: adapter missing",
            "adapter_type": "cppdbg",
        }
        mock_resp.text = "debug_adapter_unavailable: adapter missing"

        with _patch(
            "app.modules.tunnel.tunnel_service.get_tunnel_service",
            return_value=self._mock_tunnel_service(),
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
            return_value=mock_resp,
        ):
            result, err = route_dap_command(
                method="start_session",
                payload={"program": "a.py"},
                user_id="u1",
                conversation_id="c1",
            )

        assert result is None
        assert err == "debug_adapter_unavailable: adapter missing"
        assert err != "tunnel_unreachable"

    def test_socket_route_hyphenizes_method_endpoint(self):
        from unittest.mock import patch as _patch
        from app.modules.intelligence.tools.local_search_tools.tunnel_utils import route_dap_command

        mock_svc = MagicMock()
        mock_svc.get_tunnel_url.return_value = "socket://workspace-1"
        mock_svc.get_workspace_id.return_value = "workspace-1"

        with _patch(
            "app.modules.tunnel.tunnel_service.get_tunnel_service",
            return_value=mock_svc,
        ), _patch(
            "app.modules.intelligence.tools.code_changes_manager._get_tunnel_url",
            return_value=None,
        ), _patch(
            "app.modules.intelligence.tools.code_changes_manager._get_repository",
            return_value=None,
        ), _patch(
            "app.modules.intelligence.tools.code_changes_manager._get_branch",
            return_value=None,
        ), _patch(
            "app.modules.intelligence.tools.local_search_tools.tunnel_utils._execute_via_socket",
            return_value={"session_id": "s1", "status": "paused"},
        ) as mock_socket:
            result, err = route_dap_command(
                method="continue_execution",
                payload={},
                user_id="u1",
                conversation_id="c1",
            )

        assert err is None
        assert result["session_id"] == "s1"
        assert mock_socket.call_args.kwargs["endpoint"] == "/api/debug/continue-execution"

    def test_socket_timeout_returns_timeout_code(self):
        from unittest.mock import patch as _patch
        from app.modules.intelligence.tools.local_search_tools.tunnel_utils import route_dap_command

        mock_svc = MagicMock()
        mock_svc.get_tunnel_url.return_value = "socket://workspace-1"
        mock_svc.get_workspace_id.return_value = "workspace-1"

        with _patch(
            "app.modules.tunnel.tunnel_service.get_tunnel_service",
            return_value=mock_svc,
        ), _patch(
            "app.modules.intelligence.tools.code_changes_manager._get_tunnel_url",
            return_value=None,
        ), _patch(
            "app.modules.intelligence.tools.code_changes_manager._get_repository",
            return_value=None,
        ), _patch(
            "app.modules.intelligence.tools.code_changes_manager._get_branch",
            return_value=None,
        ), _patch(
            "app.modules.intelligence.tools.local_search_tools.tunnel_utils._execute_via_socket",
            return_value=(None, "timeout"),
        ):
            result, err = route_dap_command(
                method="start_session",
                payload={"program": "a.py"},
                user_id="u1",
                conversation_id="c1",
            )

        assert result is None
        assert err == "timeout"

    def test_socket_extension_error_is_preserved(self):
        from unittest.mock import patch as _patch
        from app.modules.intelligence.tools.local_search_tools.tunnel_utils import route_dap_command

        mock_svc = MagicMock()
        mock_svc.get_tunnel_url.return_value = "socket://workspace-1"
        mock_svc.get_workspace_id.return_value = "workspace-1"

        with _patch(
            "app.modules.tunnel.tunnel_service.get_tunnel_service",
            return_value=mock_svc,
        ), _patch(
            "app.modules.intelligence.tools.code_changes_manager._get_tunnel_url",
            return_value=None,
        ), _patch(
            "app.modules.intelligence.tools.code_changes_manager._get_repository",
            return_value=None,
        ), _patch(
            "app.modules.intelligence.tools.code_changes_manager._get_branch",
            return_value=None,
        ), _patch(
            "app.modules.intelligence.tools.local_search_tools.tunnel_utils._execute_via_socket",
            return_value=(None, "debug adapter not available"),
        ):
            result, err = route_dap_command(
                method="start_session",
                payload={"program": "a.py"},
                user_id="u1",
                conversation_id="c1",
            )

        assert result is None
        assert err == "debug adapter not available"

    def test_socket_domain_failure_body_returns_extension_error(self):
        from unittest.mock import patch as _patch
        from app.modules.intelligence.tools.local_search_tools.tunnel_utils import route_dap_command

        mock_svc = MagicMock()
        mock_svc.get_tunnel_url.return_value = "socket://workspace-1"
        mock_svc.get_workspace_id.return_value = "workspace-1"

        with _patch(
            "app.modules.tunnel.tunnel_service.get_tunnel_service",
            return_value=mock_svc,
        ), _patch(
            "app.modules.intelligence.tools.code_changes_manager._get_tunnel_url",
            return_value=None,
        ), _patch(
            "app.modules.intelligence.tools.code_changes_manager._get_repository",
            return_value=None,
        ), _patch(
            "app.modules.intelligence.tools.code_changes_manager._get_branch",
            return_value=None,
        ), _patch(
            "app.modules.intelligence.tools.local_search_tools.tunnel_utils._execute_via_socket",
            return_value=({"success": False, "error": "debug_adapter_unavailable"}, None),
        ):
            result, err = route_dap_command(
                method="start_session",
                payload={"program": "a.py"},
                user_id="u1",
                conversation_id="c1",
            )

        assert result is None
        assert err == "debug_adapter_unavailable"


# ===========================================================================
# A3.3 — Per-tool tests
# ===========================================================================


class TestStartDebugSession:
    def test_success_returns_start_session_result(self):
        from app.modules.intelligence.tools.dap_tools import start_debug_session

        payload = {"session_id": "s1", "program": "/a.py", "language": "python", "status": "initialized"}
        with patch(_DISPATCH, return_value=(payload, None)), patch(_CTX, return_value=_CTX_RETURN):
            result = start_debug_session(program="/a.py")
        assert result["session_id"] == "s1"
        assert result["status"] == "initialized"

    def test_no_tunnel_returns_dap_error(self):
        from app.modules.intelligence.tools.dap_tools import start_debug_session

        with patch(_DISPATCH, return_value=(None, "no_tunnel")), patch(_CTX, return_value=_CTX_RETURN):
            result = start_debug_session(program="/a.py")
        assert result["error_type"] == "no_tunnel"
        assert result["available"] is False

    def test_unknown_route_returns_dap_error(self):
        from app.modules.intelligence.tools.dap_tools import start_debug_session

        with patch(_DISPATCH, return_value=(None, "unknown_route")), patch(_CTX, return_value=_CTX_RETURN):
            result = start_debug_session(program="/a.py")
        assert result["error_type"] == "unknown_route"

    def test_forwards_correct_method_and_payload(self):
        from app.modules.intelligence.tools.dap_tools import start_debug_session

        with patch(_DISPATCH, return_value=(None, "no_tunnel")) as mock_d, \
             patch(_CTX, return_value=_CTX_RETURN):
            start_debug_session(
                program="/a.py",
                language="python",
                mode="launch",
                port=5678,
                args={"stopOnEntry": True},
            )

        mock_d.assert_called_once()
        kwargs = mock_d.call_args.kwargs if mock_d.call_args.kwargs else {}
        args = mock_d.call_args.args if mock_d.call_args.args else ()
        # method should be "start_session"
        method_val = kwargs.get("method") or (args[0] if args else None)
        assert method_val == "start_session"
        # payload should contain program and port
        payload_val = kwargs.get("payload") or (args[1] if len(args) > 1 else None)
        assert payload_val["program"] == "/a.py"
        assert payload_val["port"] == 5678
        assert payload_val["args"] == {"stopOnEntry": True}
        assert kwargs.get("timeout") == 65.0

    def test_empty_session_id_returns_extension_error(self):
        from app.modules.intelligence.tools.dap_tools import start_debug_session

        with patch(_DISPATCH, return_value=({"success": False, "error": "launch_failed"}, None)), \
             patch(_CTX, return_value=_CTX_RETURN):
            result = start_debug_session(program="/a.py")
        assert result["error_type"] == "extension_error"
        assert "session_id" in result["message"]

    def test_domain_failure_body_returns_extension_error(self):
        from app.modules.intelligence.tools.dap_tools import start_debug_session

        with patch(
            _DISPATCH,
            return_value=(None, "debug_adapter_unavailable"),
        ), patch(_CTX, return_value=_CTX_RETURN):
            result = start_debug_session(program="/a.py")
        assert result["error_type"] == "extension_error"
        assert "not installed" in result["message"].lower()

    def test_legacy_argv_list_is_wrapped_as_launch_config_args(self):
        from app.modules.intelligence.tools.dap_tools import start_debug_session

        with patch(_DISPATCH, return_value=(None, "no_tunnel")) as mock_d, \
             patch(_CTX, return_value=_CTX_RETURN):
            start_debug_session(program="/a.py", args=["--flag"])

        payload = mock_d.call_args.kwargs["payload"]
        assert payload["args"] == {"args": ["--flag"]}

    def test_does_not_raise_on_exception(self):
        from app.modules.intelligence.tools.dap_tools import start_debug_session

        with patch(_DISPATCH, side_effect=RuntimeError("boom")), patch(_CTX, return_value=_CTX_RETURN):
            result = start_debug_session(program="/a.py")
        assert result["available"] is False
        assert result["error_type"] == "unknown_error"


class TestSetBreakpoints:
    def test_success_returns_set_breakpoints_result(self):
        from app.modules.intelligence.tools.dap_tools import set_breakpoints

        payload = {"file": "/a.py", "breakpoints": [{"line": 5, "verified": True}]}
        with patch(_DISPATCH, return_value=(payload, None)), patch(_CTX, return_value=_CTX_RETURN):
            result = set_breakpoints(file="/a.py", lines=[5])
        assert result["file"] == "/a.py"
        assert result["breakpoints"][0]["verified"] is True

    def test_no_tunnel_returns_dap_error(self):
        from app.modules.intelligence.tools.dap_tools import set_breakpoints

        with patch(_DISPATCH, return_value=(None, "no_tunnel")), patch(_CTX, return_value=_CTX_RETURN):
            result = set_breakpoints(file="/a.py", lines=[5])
        assert result["error_type"] == "no_tunnel"

    def test_unknown_route_returns_dap_error(self):
        from app.modules.intelligence.tools.dap_tools import set_breakpoints

        with patch(_DISPATCH, return_value=(None, "unknown_route")), patch(_CTX, return_value=_CTX_RETURN):
            result = set_breakpoints(file="/a.py", lines=[5])
        assert result["error_type"] == "unknown_route"

    def test_forwards_correct_method_and_payload(self):
        from app.modules.intelligence.tools.dap_tools import set_breakpoints

        with patch(_DISPATCH, return_value=(None, "no_tunnel")) as mock_d, \
             patch(_CTX, return_value=_CTX_RETURN):
            set_breakpoints(file="/a.py", lines=[10, 20], condition="x > 5")

        kwargs = mock_d.call_args.kwargs
        assert kwargs["method"] == "set_breakpoints"
        assert kwargs["payload"]["file"] == "/a.py"
        assert kwargs["payload"]["lines"] == [10, 20]
        assert kwargs["payload"]["condition"] == "x > 5"


class TestTakeDebugSnapshot:
    def test_success_returns_debug_snapshot(self):
        from app.modules.intelligence.tools.dap_tools import take_debug_snapshot

        with patch(_DISPATCH, return_value=(_snapshot_payload(), None)), \
             patch(_CTX, return_value=_CTX_RETURN):
            result = take_debug_snapshot()
        assert result["session_id"] == "sess-abc"
        assert result["status"] == "paused"
        assert result["locals"] == {"x": "42"}

    def test_no_tunnel_returns_dap_error(self):
        from app.modules.intelligence.tools.dap_tools import take_debug_snapshot

        with patch(_DISPATCH, return_value=(None, "no_tunnel")), patch(_CTX, return_value=_CTX_RETURN):
            result = take_debug_snapshot()
        assert result["error_type"] == "no_tunnel"

    def test_unknown_route_returns_dap_error(self):
        from app.modules.intelligence.tools.dap_tools import take_debug_snapshot

        with patch(_DISPATCH, return_value=(None, "unknown_route")), patch(_CTX, return_value=_CTX_RETURN):
            result = take_debug_snapshot()
        assert result["error_type"] == "unknown_route"

    def test_forwards_correct_method_and_payload(self):
        from app.modules.intelligence.tools.dap_tools import take_debug_snapshot

        with patch(_DISPATCH, return_value=(None, "no_tunnel")) as mock_d, \
             patch(_CTX, return_value=_CTX_RETURN):
            take_debug_snapshot(session_id="s1", expressions=["x+1"], wait_for_stop=True)

        kwargs = mock_d.call_args.kwargs
        assert kwargs["method"] == "snapshot"
        assert kwargs["payload"]["session_id"] == "s1"
        assert kwargs["payload"]["expressions"] == ["x+1"]
        assert kwargs["payload"]["wait"] is True


# ---------------------------------------------------------------------------
# Parametrized step_* tests
# ---------------------------------------------------------------------------

_STEP_TOOLS = [
    ("step_over", "step_over"),
    ("step_into", "step_into"),
    ("step_out", "step_out"),
]


@pytest.mark.parametrize("tool_name,expected_method", _STEP_TOOLS)
class TestStepTools:
    def _get_tool_fn(self, tool_name):
        import app.modules.intelligence.tools.dap_tools as dap
        return getattr(dap, tool_name)

    def test_success_returns_debug_snapshot(self, tool_name, expected_method):
        fn = self._get_tool_fn(tool_name)
        with patch(_DISPATCH, return_value=(_snapshot_payload(), None)), \
             patch(_CTX, return_value=_CTX_RETURN):
            result = fn()
        assert result["session_id"] == "sess-abc"
        assert result["status"] == "paused"

    def test_no_tunnel_returns_dap_error(self, tool_name, expected_method):
        fn = self._get_tool_fn(tool_name)
        with patch(_DISPATCH, return_value=(None, "no_tunnel")), patch(_CTX, return_value=_CTX_RETURN):
            result = fn()
        assert result["error_type"] == "no_tunnel"

    def test_unknown_route_returns_dap_error(self, tool_name, expected_method):
        fn = self._get_tool_fn(tool_name)
        with patch(_DISPATCH, return_value=(None, "unknown_route")), patch(_CTX, return_value=_CTX_RETURN):
            result = fn()
        assert result["error_type"] == "unknown_route"

    def test_forwards_correct_method(self, tool_name, expected_method):
        fn = self._get_tool_fn(tool_name)
        with patch(_DISPATCH, return_value=(None, "no_tunnel")) as mock_d, \
             patch(_CTX, return_value=_CTX_RETURN):
            fn(session_id="s1", expressions=["x"])
        kwargs = mock_d.call_args.kwargs
        assert kwargs["method"] == expected_method
        assert kwargs["payload"].get("session_id") == "s1"
        assert kwargs["payload"].get("expressions") == ["x"]
        assert kwargs.get("timeout") == 40.0


class TestContinueExecution:
    def test_success_returns_running_snapshot(self):
        from app.modules.intelligence.tools.dap_tools import continue_execution

        with patch(_DISPATCH, return_value=({"session_id": "s1", "status": "running"}, None)), \
             patch(_CTX, return_value=_CTX_RETURN):
            result = continue_execution()
        assert result["status"] == "running"
        assert result["session_id"] == "s1"

    def test_no_tunnel_returns_dap_error(self):
        from app.modules.intelligence.tools.dap_tools import continue_execution

        with patch(_DISPATCH, return_value=(None, "no_tunnel")), patch(_CTX, return_value=_CTX_RETURN):
            result = continue_execution()
        assert result["error_type"] == "no_tunnel"

    def test_unknown_route_returns_dap_error(self):
        from app.modules.intelligence.tools.dap_tools import continue_execution

        with patch(_DISPATCH, return_value=(None, "unknown_route")), patch(_CTX, return_value=_CTX_RETURN):
            result = continue_execution()
        assert result["error_type"] == "unknown_route"

    def test_forwards_correct_method_and_payload(self):
        from app.modules.intelligence.tools.dap_tools import continue_execution

        with patch(_DISPATCH, return_value=(None, "no_tunnel")) as mock_d, \
             patch(_CTX, return_value=_CTX_RETURN):
            continue_execution(session_id="s1")
        kwargs = mock_d.call_args.kwargs
        assert kwargs["method"] == "continue_execution"
        assert kwargs["payload"].get("session_id") == "s1"


class TestEvaluateExpression:
    def test_success_returns_evaluate_result(self):
        from app.modules.intelligence.tools.dap_tools import evaluate_expression

        with patch(_DISPATCH, return_value=({"session_id": "s1", "expression": "x+1", "result": "43", "type": "int"}, None)), \
             patch(_CTX, return_value=_CTX_RETURN):
            result = evaluate_expression(expression="x+1")
        assert result["value"] == "43"
        assert result["type"] == "int"
        assert result["expression"] == "x+1"

    def test_preserves_falsy_extension_result_values(self):
        from app.modules.intelligence.tools.dap_tools import evaluate_expression

        with patch(_DISPATCH, return_value=({"session_id": "s1", "expression": "x", "result": 0}, None)), \
             patch(_CTX, return_value=_CTX_RETURN):
            result = evaluate_expression(expression="x")
        assert result["value"] == "0"

    def test_no_tunnel_returns_dap_error(self):
        from app.modules.intelligence.tools.dap_tools import evaluate_expression

        with patch(_DISPATCH, return_value=(None, "no_tunnel")), patch(_CTX, return_value=_CTX_RETURN):
            result = evaluate_expression(expression="x")
        assert result["error_type"] == "no_tunnel"

    def test_unknown_route_returns_dap_error(self):
        from app.modules.intelligence.tools.dap_tools import evaluate_expression

        with patch(_DISPATCH, return_value=(None, "unknown_route")), patch(_CTX, return_value=_CTX_RETURN):
            result = evaluate_expression(expression="x")
        assert result["error_type"] == "unknown_route"

    def test_forwards_correct_method_and_payload(self):
        from app.modules.intelligence.tools.dap_tools import evaluate_expression

        with patch(_DISPATCH, return_value=(None, "no_tunnel")) as mock_d, \
             patch(_CTX, return_value=_CTX_RETURN):
            evaluate_expression(expression="y*2", session_id="s1", frame_id=3)
        kwargs = mock_d.call_args.kwargs
        assert kwargs["method"] == "evaluate"
        assert kwargs["payload"]["expression"] == "y*2"
        assert kwargs["payload"]["frame_id"] == 3
        assert kwargs["payload"]["session_id"] == "s1"


class TestListDebugSessions:
    def test_success_returns_list_sessions_result(self):
        from app.modules.intelligence.tools.dap_tools import list_debug_sessions

        raw = [{"session_id": "s1", "program": "/a.py", "language": "python", "status": "paused"}]
        with patch(_DISPATCH, return_value=(raw, None)), patch(_CTX, return_value=_CTX_RETURN):
            result = list_debug_sessions()
        assert "sessions" in result
        assert result["sessions"][0]["session_id"] == "s1"

    def test_success_with_sessions_key_in_result(self):
        from app.modules.intelligence.tools.dap_tools import list_debug_sessions

        raw = {"sessions": [{"session_id": "s2", "status": "running"}]}
        with patch(_DISPATCH, return_value=(raw, None)), patch(_CTX, return_value=_CTX_RETURN):
            result = list_debug_sessions()
        assert result["sessions"][0]["session_id"] == "s2"

    def test_no_tunnel_returns_dap_error(self):
        from app.modules.intelligence.tools.dap_tools import list_debug_sessions

        with patch(_DISPATCH, return_value=(None, "no_tunnel")), patch(_CTX, return_value=_CTX_RETURN):
            result = list_debug_sessions()
        assert result["error_type"] == "no_tunnel"

    def test_unknown_route_returns_dap_error(self):
        from app.modules.intelligence.tools.dap_tools import list_debug_sessions

        with patch(_DISPATCH, return_value=(None, "unknown_route")), patch(_CTX, return_value=_CTX_RETURN):
            result = list_debug_sessions()
        assert result["error_type"] == "unknown_route"

    def test_forwards_correct_method(self):
        from app.modules.intelligence.tools.dap_tools import list_debug_sessions

        with patch(_DISPATCH, return_value=(None, "no_tunnel")) as mock_d, \
             patch(_CTX, return_value=_CTX_RETURN):
            list_debug_sessions()
        kwargs = mock_d.call_args.kwargs
        assert kwargs["method"] == "list_sessions"


class TestStopDebugSession:
    def test_success_returns_stop_session_result(self):
        from app.modules.intelligence.tools.dap_tools import stop_debug_session

        with patch(_DISPATCH, return_value=({"session_id": "s1", "stopped": True}, None)), \
             patch(_CTX, return_value=_CTX_RETURN):
            result = stop_debug_session(session_id="s1")
        assert result["session_id"] == "s1"
        assert result["status"] == "terminated"

    def test_not_found_maps_to_not_found(self):
        from app.modules.intelligence.tools.dap_tools import stop_debug_session

        with patch(_DISPATCH, return_value=({"session_id": "s1", "stopped": False}, None)), \
             patch(_CTX, return_value=_CTX_RETURN):
            result = stop_debug_session(session_id="s1")
        assert result["status"] == "not_found"

    def test_no_tunnel_returns_dap_error(self):
        from app.modules.intelligence.tools.dap_tools import stop_debug_session

        with patch(_DISPATCH, return_value=(None, "no_tunnel")), patch(_CTX, return_value=_CTX_RETURN):
            result = stop_debug_session()
        assert result["error_type"] == "no_tunnel"

    def test_unknown_route_returns_dap_error(self):
        from app.modules.intelligence.tools.dap_tools import stop_debug_session

        with patch(_DISPATCH, return_value=(None, "unknown_route")), patch(_CTX, return_value=_CTX_RETURN):
            result = stop_debug_session()
        assert result["error_type"] == "unknown_route"

    def test_forwards_correct_method_and_payload(self):
        from app.modules.intelligence.tools.dap_tools import stop_debug_session

        with patch(_DISPATCH, return_value=(None, "no_tunnel")) as mock_d, \
             patch(_CTX, return_value=_CTX_RETURN):
            stop_debug_session(session_id="s1")
        kwargs = mock_d.call_args.kwargs
        assert kwargs["method"] == "stop_session"
        assert kwargs["payload"]["session_id"] == "s1"


# ===========================================================================
# A3.6 — Extension contract parsing with stubbed responder shapes
# ===========================================================================


class TestExtensionContractShapes:
    def test_each_dap_tool_parses_extension_json_shapes(self):
        import app.modules.intelligence.tools.dap_tools as dap

        def fake_dispatch(method, payload, user_id, conversation_id=None, timeout=30.0):
            assert user_id == _CTX_RETURN[0]
            if method == "start_session":
                return {
                    "session_id": "sess-1",
                    "program": payload["program"],
                    "language": payload["language"],
                    "status": "initialized",
                }, None
            if method == "set_breakpoints":
                return {
                    "session_id": "sess-1",
                    "file": payload["file"],
                    "breakpoints": [{"line": 25, "verified": True}],
                }, None
            if method in {"snapshot", "step_over", "step_into", "step_out"}:
                return {
                    "session_id": "sess-1",
                    "status": "paused",
                    "paused_at": {"file": "/repo/app.py", "line": 25, "function": "main"},
                    "call_stack": [
                        {"frame_id": 1, "function": "main", "file": "/repo/app.py", "line": 25}
                    ],
                    "locals": {"x": "10", "y": "20"},
                    "expression_results": [
                        {"expression": "x + y", "result": "30"}
                    ],
                }, None
            if method == "continue_execution":
                return {"session_id": "sess-1", "status": "running"}, None
            if method == "evaluate":
                return {
                    "session_id": "sess-1",
                    "expression": payload["expression"],
                    "result": "30",
                }, None
            if method == "list_sessions":
                return {
                    "sessions": [
                        {
                            "session_id": "sess-1",
                            "program": "/repo/app.py",
                            "language": "python",
                            "status": "paused",
                            "createdAt": "2026-05-21T00:00:00Z",
                        }
                    ]
                }, None
            if method == "stop_session":
                return {"session_id": "sess-1", "stopped": True}, None
            raise AssertionError(f"Unexpected method: {method}")

        with patch(_DISPATCH, side_effect=fake_dispatch), patch(_CTX, return_value=_CTX_RETURN):
            start = dap.start_debug_session(
                program="/repo/app.py",
                language="python",
                args={"stopOnEntry": True},
            )
            bps = dap.set_breakpoints(file="/repo/app.py", lines=[25])
            snapshot = dap.take_debug_snapshot(
                session_id="sess-1",
                expressions=["x + y"],
                wait_for_stop=True,
            )
            step = dap.step_over(session_id="sess-1", expressions=["x + y"])
            continued = dap.continue_execution(session_id="sess-1")
            evaluated = dap.evaluate_expression(
                expression="x + y",
                session_id="sess-1",
                frame_id=1,
            )
            sessions = dap.list_debug_sessions()
            stopped = dap.stop_debug_session(session_id="sess-1")

        assert start["status"] == "initialized"
        assert bps["breakpoints"][0]["verified"] is True
        assert snapshot["call_stack"][0]["name"] == "main"
        assert snapshot["call_stack"][0]["source_path"] == "/repo/app.py"
        assert snapshot["locals"] == {"x": "10", "y": "20"}
        assert snapshot["expression_results"][0]["result"] == "30"
        assert step["expression_results"][0]["expression"] == "x + y"
        assert continued["status"] == "running"
        assert evaluated["value"] == "30"
        assert sessions["sessions"][0]["created_at"] == "2026-05-21T00:00:00Z"
        assert stopped["status"] == "terminated"


# ===========================================================================
# A3.5 — Registration checks (source text inspection, same pattern as A4 tests)
# ===========================================================================


class TestRegistration:
    def test_all_dap_tools_registered_in_tool_service(self):
        import pathlib

        tools_dir = (
            pathlib.Path(__file__).parents[3]
            / "app"
            / "modules"
            / "intelligence"
            / "tools"
        )
        source = (tools_dir / "tool_service.py").read_text(encoding="utf-8")

        expected_keys = [
            "start_debug_session",
            "set_breakpoints",
            "take_debug_snapshot",
            "step_over",
            "step_into",
            "step_out",
            "continue_execution",
            "evaluate_expression",
            "list_debug_sessions",
            "stop_debug_session",
        ]
        for key in expected_keys:
            assert f'"{key}"' in source or f"'{key}'" in source, (
                f"tool_service.py is missing registration for '{key}'"
            )

    def test_all_dap_tools_in_debug_agent_get_tools(self):
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

        expected_keys = [
            "start_debug_session",
            "set_breakpoints",
            "take_debug_snapshot",
            "step_over",
            "step_into",
            "step_out",
            "continue_execution",
            "evaluate_expression",
            "list_debug_sessions",
            "stop_debug_session",
        ]
        for key in expected_keys:
            assert key in tool_list, (
                f"'{key}' not found in DebugAgent's tool allow-list. Found: {tool_list}"
            )

    def test_dap_tools_not_in_embedding_dependent_tools(self):
        """Verify none of the DAP tools appear in EMBEDDING_DEPENDENT_TOOLS via source text."""
        import pathlib

        tools_dir = (
            pathlib.Path(__file__).parents[3]
            / "app"
            / "modules"
            / "intelligence"
            / "tools"
        )
        source = (tools_dir / "tool_service.py").read_text(encoding="utf-8")

        # Find the EMBEDDING_DEPENDENT_TOOLS block
        start = source.find("EMBEDDING_DEPENDENT_TOOLS")
        assert start != -1, "Could not find EMBEDDING_DEPENDENT_TOOLS in tool_service.py"
        # Extract the set literal (up to its closing brace)
        block_start = source.find("{", start)
        block_end = source.find("}", block_start)
        assert block_start != -1 and block_end != -1
        embedding_block = source[block_start : block_end + 1]

        dap_tools = [
            "start_debug_session", "set_breakpoints", "take_debug_snapshot",
            "step_over", "step_into", "step_out", "continue_execution",
            "evaluate_expression", "list_debug_sessions", "stop_debug_session",
        ]
        for tool in dap_tools:
            assert tool not in embedding_block, (
                f"DAP tool '{tool}' should NOT be in EMBEDDING_DEPENDENT_TOOLS"
            )


# ===========================================================================
# A3.5 — StructuredTool factory sanity checks
# ===========================================================================


class TestToolFactories:
    @pytest.mark.parametrize("factory_name,expected_tool_name", [
        ("start_debug_session_tool", "start_debug_session"),
        ("set_breakpoints_tool", "set_breakpoints"),
        ("take_debug_snapshot_tool", "take_debug_snapshot"),
        ("step_over_tool", "step_over"),
        ("step_into_tool", "step_into"),
        ("step_out_tool", "step_out"),
        ("continue_execution_tool", "continue_execution"),
        ("evaluate_expression_tool", "evaluate_expression"),
        ("list_debug_sessions_tool", "list_debug_sessions"),
        ("stop_debug_session_tool", "stop_debug_session"),
    ])
    def test_factory_returns_structured_tool_with_correct_name(self, factory_name, expected_tool_name):
        from langchain_core.tools import StructuredTool
        import app.modules.intelligence.tools.dap_tools as dap

        factory = getattr(dap, factory_name)
        tool = factory()
        assert isinstance(tool, StructuredTool)
        assert tool.name == expected_tool_name

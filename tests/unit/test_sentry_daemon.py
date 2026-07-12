from __future__ import annotations

import json

import pytest
from fastapi import HTTPException
from typer.testing import CliRunner

from potpie.cli import host_cli
from potpie.cli.commands import _common
from potpie.cli.telemetry.context import current_telemetry_context
from potpie.daemon import main as daemon_main
from potpie.daemon.rpc import dispatch_rpc
from potpie.runtime.contracts import CapabilityNotImplemented


class _CrashingDaemon:
    in_process = False
    home = None

    def status(self) -> dict[str, str]:
        raise RuntimeError("daemon crashed with /Users/dsantra/private/path")

    def stop(self) -> dict[str, str]:
        raise CapabilityNotImplemented(
            "potpie.daemon.lifecycle.stop",
            recommended_next_action="host runs in-process; nothing to stop",
        )


class _FakeHost:
    daemon = _CrashingDaemon()


def test_daemon_unexpected_failure_is_captured_with_session_id(
    monkeypatch,
    tmp_path,
) -> None:
    captured: list[tuple[str, str, str | None]] = []
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    monkeypatch.setattr(
        "potpie.cli.telemetry.sentry_runtime.capture_unexpected_cli_error",
        lambda exc, *, error_code, error_kind: captured.append(
            (error_code, error_kind, current_telemetry_context().daemon_session_id)
        ),
    )
    _common.set_host(_FakeHost())

    result = CliRunner().invoke(host_cli.app, ["--json", "daemon", "status"])

    payload = json.loads(result.stdout)["error"]
    assert result.exit_code == _common.EXIT_INTERNAL
    assert payload["code"] == "unexpected_cli_error"
    assert captured == [
        (
            "unexpected_cli_error",
            "unexpected",
            current_telemetry_context().daemon_session_id,
        )
    ]


def test_daemon_expected_not_implemented_is_not_captured(
    monkeypatch,
    tmp_path,
) -> None:
    captured: list[BaseException] = []
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    monkeypatch.setattr(
        "potpie.cli.telemetry.sentry_runtime.capture_unexpected_cli_error",
        lambda exc, *, error_code, error_kind: captured.append(exc),
    )
    _common.set_host(_FakeHost())

    result = CliRunner().invoke(host_cli.app, ["--json", "daemon", "stop"])

    payload = json.loads(result.stdout)["error"]
    assert result.exit_code == _common.EXIT_OPERATION
    assert payload["code"] == "not_implemented"
    assert captured == []


@pytest.mark.asyncio
async def test_daemon_rpc_unexpected_failure_is_captured(monkeypatch) -> None:
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "potpie.daemon.telemetry.sentry_runtime.capture_unexpected_daemon_error",
        lambda exc, *, error_code, error_kind: captured.append(
            (error_code, error_kind)
        ),
    )

    class Context:
        async def status(self, _request):
            raise RuntimeError("daemon exploded")

    class Engine:
        context = Context()

    payload = await dispatch_rpc(
        Engine(),
        {
            "protocol_version": "1",
            "request_id": "boom",
            "method": "engine.context.status",
            "params": {},
        },
    )

    assert payload["error"]["code"] == "ENGINE_INTERNAL_ERROR"
    assert captured == [("ENGINE_INTERNAL_ERROR", "unexpected")]


@pytest.mark.asyncio
async def test_daemon_rpc_expected_error_is_not_captured(monkeypatch) -> None:
    captured: list[BaseException] = []
    monkeypatch.setattr(
        "potpie.daemon.telemetry.sentry_runtime.capture_unexpected_daemon_error",
        lambda exc, *, error_code, error_kind: captured.append(exc),
    )

    class Context:
        async def status(self, _request):
            raise CapabilityNotImplemented("graph.snapshot.export")

    class Engine:
        context = Context()

    payload = await dispatch_rpc(
        Engine(),
        {
            "protocol_version": "1",
            "request_id": "expected",
            "method": "engine.context.status",
            "params": {},
        },
    )

    assert payload["error"]["code"] == "ENGINE_CAPABILITY_NOT_IMPLEMENTED"
    assert captured == []


def test_daemon_rpc_authorize_uses_compare_digest(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def _compare_digest(left: str, right: str) -> bool:
        calls.append((left, right))
        return left == right

    monkeypatch.setattr(daemon_main.secrets, "compare_digest", _compare_digest)

    daemon_main._authorize("Bearer token", "token")

    assert calls == [("Bearer token", "Bearer token")]
    with pytest.raises(HTTPException) as exc_info:
        daemon_main._authorize(None, "token")
    assert exc_info.value.status_code == 401

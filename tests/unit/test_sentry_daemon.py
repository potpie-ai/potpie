from __future__ import annotations

import json

import pytest
from fastapi import HTTPException
from typer.testing import CliRunner

from potpie.cli import main as host_cli
from potpie.cli.commands import _common
from potpie.cli.telemetry.context import current_telemetry_context
from potpie_context_engine.domain.errors import CapabilityNotImplemented
from potpie.daemon import main as daemon_main


class _CrashingDaemon:
    in_process = False
    home = None

    def status(self) -> dict[str, str]:
        raise RuntimeError("daemon crashed with /Users/dsantra/private/path")

    def stop(self) -> dict[str, str]:
        raise CapabilityNotImplemented(
            "host.daemon.stop",
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
    runner = CliRunner()

    result = runner.invoke(host_cli.app, ["--json", "daemon", "status"])

    payload = json.loads(result.stdout)
    assert result.exit_code == _common.EXIT_VALIDATION
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
    runner = CliRunner()

    result = runner.invoke(host_cli.app, ["--json", "daemon", "stop"])

    payload = json.loads(result.stdout)
    assert result.exit_code == _common.EXIT_UNAVAILABLE
    assert payload["code"] == "not_implemented"
    assert captured == []


def test_daemon_rpc_unexpected_failure_is_captured(monkeypatch) -> None:
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "potpie.cli.sentry_runtime.capture_unexpected_daemon_error",
        lambda exc, *, error_code, error_kind: captured.append(
            (error_code, error_kind)
        ),
    )

    payload = daemon_main._error_payload(RuntimeError("daemon exploded"))

    assert payload["ok"] is False
    assert payload["error"]["code"] == "daemon_error"
    assert captured == [("daemon_error", "unexpected")]


def test_daemon_rpc_expected_error_is_not_captured(monkeypatch) -> None:
    captured: list[BaseException] = []
    monkeypatch.setattr(
        "potpie.cli.sentry_runtime.capture_unexpected_daemon_error",
        lambda exc, *, error_code, error_kind: captured.append(exc),
    )

    payload = daemon_main._error_payload(
        CapabilityNotImplemented("graph.neo4j.snapshot.export")
    )

    assert payload["ok"] is False
    assert payload["error"]["code"] == "not_implemented"
    assert captured == []


def test_daemon_rpc_validation_error_guidance_round_trips() -> None:
    # UnknownGraphViewError's did_you_mean must survive the daemon RPC
    # boundary so remote reads error exactly like in-process ones.
    from potpie_context_engine.domain.graph_views import UnknownGraphViewError, include_guess_guidance
    from potpie.daemon import daemon_client

    guidance = include_guess_guidance("docs", "relevant")
    payload = daemon_main._error_payload(
        UnknownGraphViewError(
            "unknown graph view 'docs.relevant'",
            did_you_mean=guidance,
            recommended_next_action=guidance["read_command"],
        )
    )

    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["detail"] == {"did_you_mean": guidance}
    assert payload["error"]["recommended_next_action"] == guidance["read_command"]

    with pytest.raises(ValueError) as exc_info:
        daemon_client._raise_remote_error(payload)
    assert getattr(exc_info.value, "detail") == {"did_you_mean": guidance}
    assert (
        getattr(exc_info.value, "recommended_next_action") == (guidance["read_command"])
    )


def test_daemon_rpc_plain_validation_error_has_no_guidance() -> None:
    from potpie.daemon import daemon_client

    payload = daemon_main._error_payload(ValueError("--subgraph is required"))

    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["detail"] is None
    assert payload["error"]["recommended_next_action"] is None

    with pytest.raises(ValueError) as exc_info:
        daemon_client._raise_remote_error(payload)
    assert not hasattr(exc_info.value, "detail")


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


def test_daemon_rpc_rejects_private_targets() -> None:
    with pytest.raises(ValueError, match="invalid RPC member"):
        daemon_main._validate_rpc_target("backend", "_profile")

    with pytest.raises(ValueError, match="invalid RPC surface"):
        daemon_main._validate_rpc_target("backend.__class__", "profile")

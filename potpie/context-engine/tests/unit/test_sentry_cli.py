from __future__ import annotations

import json

import pytest
import typer
from typer.testing import CliRunner

from adapters.inbound.cli import host_cli
from adapters.inbound.cli.commands import _common
from adapters.inbound.cli.telemetry_context import (
    current_telemetry_context,
    load_anonymous_install_id,
)


def test_in_process_cli_invocations_share_install_and_daemon_session_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    runner = CliRunner()

    first = runner.invoke(host_cli.app, ["--json", "daemon", "status"])
    first_ctx = current_telemetry_context()
    second = runner.invoke(host_cli.app, ["--json", "daemon", "status"])
    second_ctx = current_telemetry_context()

    assert first.exit_code == 0, first.stdout
    assert second.exit_code == 0, second.stdout
    assert first_ctx is not None
    assert second_ctx is not None
    assert first_ctx.anonymous_install_id == second_ctx.anonymous_install_id
    assert first_ctx.invocation_id != second_ctx.invocation_id
    in_process_daemon_session_id = first_ctx.daemon_session_id
    assert second_ctx.daemon_session_id == in_process_daemon_session_id
    assert first_ctx.command == "daemon"
    assert load_anonymous_install_id(tmp_path) == first_ctx.anonymous_install_id


def test_expected_contract_error_does_not_capture_sentry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[BaseException] = []
    monkeypatch.setattr(
        "adapters.inbound.cli.sentry_runtime.capture_unexpected_cli_error",
        lambda exc, *, error_code, error_kind: captured.append(exc),
    )

    with pytest.raises(typer.Exit) as exc_info:
        with _common.contract():
            raise ValueError("bad input")

    assert exc_info.value.exit_code == _common.EXIT_VALIDATION
    assert captured == []


def test_contract_preserves_expected_typer_exit_from_fail(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[BaseException] = []
    monkeypatch.setattr(
        "adapters.inbound.cli.sentry_runtime.capture_unexpected_cli_error",
        lambda exc, *, error_code, error_kind: captured.append(exc),
    )
    _common.set_json(True)

    with pytest.raises(typer.Exit) as exc_info:
        with _common.contract():
            _common.fail(
                code="no_active_pot",
                message="No active pot.",
                next_action="run 'potpie setup'",
            )

    payload = json.loads(capsys.readouterr().out)
    assert exc_info.value.exit_code == _common.EXIT_VALIDATION
    assert payload["code"] == "no_active_pot"
    assert captured == []


def test_unexpected_contract_error_is_captured_and_rendered_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        "adapters.inbound.cli.sentry_runtime.capture_unexpected_cli_error",
        lambda exc, *, error_code, error_kind: captured.append(
            (type(exc).__name__, error_code, error_kind)
        ),
    )
    _common.set_json(True)

    with pytest.raises(typer.Exit) as exc_info:
        with _common.contract():
            raise RuntimeError("boom with /Users/dsantra/private/path")

    payload = json.loads(capsys.readouterr().out)
    assert exc_info.value.exit_code == _common.EXIT_VALIDATION
    assert payload["code"] == "unexpected_cli_error"
    assert payload["message"] == "Unexpected internal error."
    assert captured == [("RuntimeError", "unexpected_cli_error", "unexpected")]


def test_cli_telemetry_identity_write_failure_is_nonfatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    home_file = tmp_path / "not-a-directory"
    home_file.write_text("not a directory", encoding="utf-8")
    install_id = load_anonymous_install_id(home_file)
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(home_file))
    runner = CliRunner()

    result = runner.invoke(host_cli.app, ["--json", "daemon", "status"])

    assert install_id.startswith("install_")
    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.stdout)
    assert payload["code"] == "unexpected_cli_error"
    assert "NotADirectoryError" not in result.output

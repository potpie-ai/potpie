from __future__ import annotations

import pytest
import typer

from adapters.inbound.cli.auth import _login_impl, github_commands
from adapters.inbound.cli.commands import _common
from adapters.outbound.cli_auth.github import GitHubDeviceFlowError
from tests._auth_fakes import InMemoryCredentialStore


def test_expected_auth_failure_is_not_captured(monkeypatch) -> None:
    captured: list[BaseException] = []
    monkeypatch.setattr(
        "adapters.inbound.cli.telemetry.sentry_runtime.capture_unexpected_cli_error",
        lambda exc, *, error_code, error_kind: captured.append(exc),
    )
    monkeypatch.setattr(
        github_commands,
        "request_device_code",
        lambda: (_ for _ in ()).throw(GitHubDeviceFlowError("authorization denied")),
    )
    _common.set_store(InMemoryCredentialStore())
    _common.set_json(True)

    with pytest.raises(typer.Exit) as exc_info:
        github_commands.github_login_impl()

    assert exc_info.value.exit_code == _common.EXIT_AUTH
    assert captured == []


def test_unexpected_auth_failure_is_captured(monkeypatch) -> None:
    captured: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        "adapters.inbound.cli.telemetry.sentry_runtime.capture_unexpected_cli_error",
        lambda exc, *, error_code, error_kind: captured.append(
            (type(exc).__name__, error_code, error_kind)
        ),
    )
    monkeypatch.setattr(
        github_commands,
        "request_device_code",
        lambda: (_ for _ in ()).throw(RuntimeError("sdk exploded with token=secret")),
    )
    _common.set_store(InMemoryCredentialStore())
    _common.set_json(True)

    with pytest.raises(typer.Exit) as exc_info:
        github_commands.github_login_impl()

    assert exc_info.value.exit_code == _common.EXIT_AUTH
    assert captured == [("RuntimeError", "unexpected_cli_error", "unexpected")]


def test_expected_typer_exit_in_github_login_is_not_recaptured(monkeypatch) -> None:
    captured: list[BaseException] = []
    monkeypatch.setattr(
        "adapters.inbound.cli.telemetry.sentry_runtime.capture_unexpected_cli_error",
        lambda exc, *, error_code, error_kind: captured.append(exc),
    )
    monkeypatch.setattr(
        github_commands,
        "request_device_code",
        lambda: (_ for _ in ()).throw(typer.Exit(code=_common.EXIT_AUTH)),
    )
    _common.set_store(InMemoryCredentialStore())
    _common.set_json(True)

    with pytest.raises(typer.Exit) as exc_info:
        github_commands.github_login_impl()

    assert exc_info.value.exit_code == _common.EXIT_AUTH
    assert captured == []


def test_unexpected_github_repo_credential_lookup_failure_is_captured(
    monkeypatch,
) -> None:
    captured: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        "adapters.inbound.cli.telemetry.sentry_runtime.capture_unexpected_cli_error",
        lambda exc, *, error_code, error_kind: captured.append(
            (type(exc).__name__, error_code, error_kind)
        ),
    )
    monkeypatch.setattr(
        github_commands,
        "get_store",
        lambda: (_ for _ in ()).throw(RuntimeError("credential store exploded")),
    )
    _common.set_json(True)

    with pytest.raises(typer.Exit) as exc_info:
        github_commands.github_test_repos_cmd()

    assert exc_info.value.exit_code == _common.EXIT_AUTH
    assert captured == [("RuntimeError", "unexpected_cli_error", "unexpected")]


def test_unexpected_potpie_login_failure_is_captured(monkeypatch) -> None:
    captured: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        "adapters.inbound.cli.telemetry.sentry_runtime.capture_unexpected_cli_error",
        lambda exc, *, error_code, error_kind: captured.append(
            (type(exc).__name__, error_code, error_kind)
        ),
    )
    monkeypatch.setattr(_login_impl, "get_store", lambda: object())
    monkeypatch.setattr(
        _login_impl,
        "run_browser_login_flow",
        lambda: (_ for _ in ()).throw(RuntimeError("browser exploded")),
    )
    _common.set_json(True)

    with pytest.raises(typer.Exit) as exc_info:
        _login_impl.potpie_login_impl()

    assert exc_info.value.exit_code == _common.EXIT_AUTH
    assert captured == [("RuntimeError", "unexpected_cli_error", "unexpected")]

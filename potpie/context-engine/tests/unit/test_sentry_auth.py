from __future__ import annotations

import pytest
import typer

from adapters.inbound.cli.auth import _login_impl, github_commands
from adapters.inbound.cli.commands import _common
from adapters.outbound.cli_auth.github import GitHubDeviceFlowError


class _FakeStore:
    def write_provider_credentials(self, provider: str, payload: dict) -> None:
        return None


def test_expected_auth_failure_is_not_captured(monkeypatch) -> None:
    captured: list[BaseException] = []
    monkeypatch.setattr(
        "adapters.inbound.cli.sentry_runtime.capture_unexpected_cli_error",
        lambda exc, *, error_code, error_kind: captured.append(exc),
    )
    monkeypatch.setattr(
        github_commands,
        "request_device_code",
        lambda: (_ for _ in ()).throw(GitHubDeviceFlowError("authorization denied")),
    )
    _common.set_store(_FakeStore())
    _common.set_json(True)

    with pytest.raises(typer.Exit) as exc_info:
        github_commands.github_login_impl()

    assert exc_info.value.exit_code == _common.EXIT_AUTH
    assert captured == []


def test_unexpected_auth_failure_is_captured(monkeypatch) -> None:
    captured: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        "adapters.inbound.cli.sentry_runtime.capture_unexpected_cli_error",
        lambda exc, *, error_code, error_kind: captured.append(
            (type(exc).__name__, error_code, error_kind)
        ),
    )
    monkeypatch.setattr(
        github_commands,
        "request_device_code",
        lambda: (_ for _ in ()).throw(RuntimeError("sdk exploded with token=secret")),
    )
    _common.set_store(_FakeStore())
    _common.set_json(True)

    with pytest.raises(typer.Exit) as exc_info:
        github_commands.github_login_impl()

    assert exc_info.value.exit_code == _common.EXIT_AUTH
    assert captured == [("RuntimeError", "unexpected_cli_error", "unexpected")]


def test_unexpected_potpie_login_failure_is_captured(monkeypatch) -> None:
    captured: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        "adapters.inbound.cli.sentry_runtime.capture_unexpected_cli_error",
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

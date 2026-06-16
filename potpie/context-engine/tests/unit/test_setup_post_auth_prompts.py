"""Post-setup GitHub and integration login prompts."""

from __future__ import annotations

import click
import pytest

from adapters.inbound.cli.ui import interactive_prompts, setup_ux


def test_maybe_prompt_github_login_runs_selected_integrations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(setup_ux, "is_interactive_tty", lambda: True)
    calls: list[str] = []

    monkeypatch.setattr(
        "typer.confirm",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(
        interactive_prompts,
        "prompt_multi_checkbox",
        lambda *_a, **_k: ["linear", "jira"],
    )
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.github_commands.github_login_impl",
        lambda: calls.append("github"),
    )
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.auth_commands.run_integration_login",
        lambda provider, *, force=False: calls.append(provider),
    )

    monkeypatch.setattr(setup_ux, "_maybe_prompt_first_pot", lambda **_k: None)
    setup_ux.maybe_prompt_github_login(repo=None, default_pot_name="default")

    assert calls == ["linear", "jira"]


def test_maybe_prompt_github_login_runs_github_when_confirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(setup_ux, "is_interactive_tty", lambda: True)
    calls: list[str] = []

    monkeypatch.setattr(
        "typer.confirm",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        interactive_prompts,
        "prompt_multi_checkbox",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.github_commands.github_login_impl",
        lambda: calls.append("github"),
    )

    monkeypatch.setattr(setup_ux, "_maybe_prompt_first_pot", lambda **_k: None)
    setup_ux.maybe_prompt_github_login(repo=None)

    assert calls == ["github"]


def test_maybe_prompt_github_login_skips_integration_on_ctrl_c_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(setup_ux, "is_interactive_tty", lambda: True)
    calls: list[str] = []

    monkeypatch.setattr(
        "typer.confirm",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(
        interactive_prompts,
        "prompt_multi_checkbox",
        lambda *_a, **_k: ["linear", "jira", "confluence"],
    )

    def _login(provider: str, *, force: bool = False) -> None:
        if provider == "linear":
            raise KeyboardInterrupt
        calls.append(provider)

    monkeypatch.setattr(
        "adapters.inbound.cli.auth.auth_commands.run_integration_login",
        _login,
    )
    monkeypatch.setattr(setup_ux, "_maybe_prompt_first_pot", lambda **_k: None)
    setup_ux.maybe_prompt_github_login(repo=None, default_pot_name="default")

    assert calls == ["jira", "confluence"]
    assert "Skipped Linear" in capsys.readouterr().out


def test_maybe_prompt_github_login_skips_integration_on_click_abort(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(setup_ux, "is_interactive_tty", lambda: True)
    calls: list[str] = []

    monkeypatch.setattr(
        "typer.confirm",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(
        interactive_prompts,
        "prompt_multi_checkbox",
        lambda *_a, **_k: ["jira", "confluence"],
    )

    def _login(provider: str, *, force: bool = False) -> None:
        if provider == "jira":
            raise click.Abort()
        calls.append(provider)

    monkeypatch.setattr(
        "adapters.inbound.cli.auth.auth_commands.run_integration_login",
        _login,
    )
    monkeypatch.setattr(setup_ux, "_maybe_prompt_first_pot", lambda **_k: None)
    setup_ux.maybe_prompt_github_login(repo=None, default_pot_name="default")

    assert calls == ["confluence"]
    assert "Skipped Jira" in capsys.readouterr().out


def test_try_integration_login_skips_when_atlassian_confirm_aborts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from adapters.inbound.cli.commands._common import contract

    def _confirm_abort(*_args: object, **_kwargs: object) -> bool:
        raise click.Abort()

    monkeypatch.setattr("typer.confirm", _confirm_abort)
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.atlassian_auth._get_product_credentials",
        lambda _product: {},
    )
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.atlassian_auth.sys.stdin.isatty",
        lambda: True,
    )

    with contract():
        setup_ux._try_integration_login("jira")

    output = capsys.readouterr().out
    assert "Skipped Jira" in output
    assert "Unexpected internal error" not in output


def test_maybe_prompt_github_login_skips_when_not_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(setup_ux, "is_interactive_tty", lambda: False)
    called = False

    def _boom() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(
        "adapters.inbound.cli.auth.github_commands.github_login_impl",
        _boom,
    )
    setup_ux.maybe_prompt_github_login()
    assert called is False

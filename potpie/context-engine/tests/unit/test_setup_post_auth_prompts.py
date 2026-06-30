"""Post-setup GitHub and integration login prompts."""

from __future__ import annotations

import click
import pytest
import typer

from adapters.inbound.cli.commands import _common
from adapters.inbound.cli.ui import interactive_prompts, setup_ux
from tests._auth_fakes import InMemoryCredentialStore


@pytest.fixture(autouse=True)
def _reset_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(_common._state, "store", None)


def test_maybe_prompt_github_login_runs_selected_integrations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(setup_ux, "is_interactive_tty", lambda: True)
    _common.set_store(InMemoryCredentialStore())
    calls: list[str] = []

    monkeypatch.setattr(
        "typer.confirm",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(
        interactive_prompts,
        "prompt_multi_checkbox",
        lambda *_a, **_k: ["linear", "atlassian"],
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

    assert calls == ["linear", "atlassian"]


def test_post_setup_integrations_use_single_atlassian_option() -> None:
    assert setup_ux.POST_SETUP_INTEGRATION_OPTIONS == (
        ("linear", "Linear"),
        ("atlassian", "Atlassian"),
    )


def test_maybe_prompt_github_login_runs_github_when_confirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(setup_ux, "is_interactive_tty", lambda: True)
    _common.set_store(InMemoryCredentialStore())
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
        lambda *_a, **_k: ["linear", "atlassian"],
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

    assert calls == ["atlassian"]
    assert "Skipped Linear" in capsys.readouterr().out


def test_maybe_prompt_github_login_cancel_exits_setup_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(setup_ux, "is_interactive_tty", lambda: True)
    _common.set_store(InMemoryCredentialStore())

    monkeypatch.setattr("typer.confirm", lambda *_a, **_k: True)
    monkeypatch.setattr(
        interactive_prompts,
        "prompt_multi_checkbox",
        lambda *_a, **_k: pytest.fail("setup should exit before integration picker"),
    )
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.github_commands.github_login_impl",
        lambda: (_ for _ in ()).throw(typer.Exit(code=130)),
    )
    monkeypatch.setattr(
        setup_ux,
        "_maybe_prompt_first_pot",
        lambda **_k: pytest.fail("setup should exit before first-pot prompt"),
    )

    with pytest.raises(typer.Exit) as exc:
        setup_ux.maybe_prompt_github_login(repo=None)

    assert exc.value.exit_code == 130


def test_maybe_prompt_github_login_skips_when_already_authenticated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(setup_ux, "is_interactive_tty", lambda: True)
    store = InMemoryCredentialStore()
    store.write_provider_credentials(
        "github",
        {
            "access_token": "stored-token",
            "account": {"login": "octocat", "email": "octo@example.com"},
        },
    )
    _common.set_store(store)
    messages: list[str] = []

    monkeypatch.setattr(
        "adapters.inbound.cli.ui.output.print_plain_line",
        lambda message, **_kwargs: messages.append(message),
    )
    monkeypatch.setattr(
        "typer.confirm",
        lambda *_a, **_k: pytest.fail("GitHub prompt should be skipped"),
    )
    monkeypatch.setattr(
        interactive_prompts,
        "prompt_multi_checkbox",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "adapters.inbound.cli.auth.github_commands.github_login_impl",
        lambda: pytest.fail("GitHub login should be skipped"),
    )

    monkeypatch.setattr(setup_ux, "_maybe_prompt_first_pot", lambda **_k: None)
    setup_ux.maybe_prompt_github_login(repo=None)

    assert messages == ["GitHub already connected as octocat; skipping login."]


def test_maybe_prompt_github_login_prompts_when_status_check_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(setup_ux, "is_interactive_tty", lambda: True)
    calls: list[str] = []

    def _fail_status() -> dict[str, object]:
        raise RuntimeError("credential store unavailable")

    monkeypatch.setattr(setup_ux, "_github_status", _fail_status)
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
        lambda *_a, **_k: ["linear", "atlassian"],
    )

    def _login(provider: str, *, force: bool = False) -> None:
        if provider == "linear":
            raise click.Abort()
        calls.append(provider)

    monkeypatch.setattr(
        "adapters.inbound.cli.auth.auth_commands.run_integration_login",
        _login,
    )
    monkeypatch.setattr(setup_ux, "_maybe_prompt_first_pot", lambda **_k: None)
    setup_ux.maybe_prompt_github_login(repo=None, default_pot_name="default")

    assert calls == ["atlassian"]
    assert "Skipped Linear" in capsys.readouterr().out


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
        setup_ux._try_integration_login("atlassian")

    output = capsys.readouterr().out
    assert "Skipped Atlassian" in output
    assert "potpie jira login" in output
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

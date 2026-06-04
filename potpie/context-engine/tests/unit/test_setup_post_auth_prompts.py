"""Post-setup GitHub and integration login prompts."""

from __future__ import annotations

import pytest

from adapters.inbound.cli.ui import interactive_prompts, setup_ux


def test_maybe_prompt_github_login_runs_selected_integrations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(setup_ux, "is_interactive_tty", lambda: True)
    calls: list[str] = []

    monkeypatch.setattr(
        interactive_prompts,
        "prompt_yes_no",
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
        interactive_prompts,
        "prompt_yes_no",
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

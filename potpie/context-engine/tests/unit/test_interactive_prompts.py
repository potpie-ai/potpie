"""Arrow-key yes/no prompt."""

import pytest

from adapters.inbound.cli.ui import interactive_prompts as prompts


def test_prompt_yes_no_selects_with_arrows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prompts, "is_interactive_tty", lambda: True)
    keys = iter(["down", "enter"])
    monkeypatch.setattr(prompts, "_read_key", lambda: next(keys))
    assert prompts.prompt_yes_no("Connect GitHub?", default=True) is False


def test_prompt_yes_no_y_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prompts, "is_interactive_tty", lambda: True)
    monkeypatch.setattr(prompts, "_read_key", lambda: "yes")
    assert prompts.prompt_yes_no("Connect GitHub?", default=False) is True


def test_prompt_multi_checkbox_toggles_and_submits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prompts, "is_interactive_tty", lambda: True)
    keys = iter(["space", "down", "space", "enter"])
    monkeypatch.setattr(prompts, "_read_key", lambda: next(keys))
    selected = prompts.prompt_multi_checkbox(
        "Connect?",
        [("linear", "Linear"), ("jira", "Jira")],
    )
    assert selected == ["linear", "jira"]


def test_prompt_multi_checkbox_escape_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prompts, "is_interactive_tty", lambda: True)
    monkeypatch.setattr(prompts, "_read_key", lambda: "escape")
    assert prompts.prompt_multi_checkbox("Connect?", [("linear", "Linear")]) == []


def test_draw_options_repaints_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    written: list[str] = []
    monkeypatch.setattr(prompts.sys.stderr, "write", written.append)
    console = prompts._stderr_console()
    prompts._draw_options(
        console, selected=0, yes_label="Yes", no_label="No", repaint=True
    )
    assert any("\033[2A" in chunk for chunk in written)

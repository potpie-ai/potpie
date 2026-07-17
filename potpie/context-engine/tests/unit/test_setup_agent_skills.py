"""Post-setup agent skill installation."""

from __future__ import annotations

from pathlib import Path

import pytest

from potpie_context_engine.adapters.inbound.cli.ui import interactive_prompts, setup_ux


def test_install_agents_to_repo_writes_claude_bundle(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    results = setup_ux.install_agents_to_repo(repo, ["claude"])

    assert len(results) == 1
    agent, result = results[0]
    assert agent == "claude"
    assert "CLAUDE.md" in result.created
    assert (repo / "CLAUDE.md").exists()


def test_maybe_prompt_agent_skills_installs_selected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    globally_installed: list[str] = []

    monkeypatch.setattr(
        interactive_prompts,
        "prompt_multi_checkbox",
        lambda *_a, **_k: ["claude"],
    )
    monkeypatch.setattr(
        setup_ux,
        "install_agents_globally",
        lambda agents: globally_installed.extend(agents)
        or [(agent, object()) for agent in agents],
    )

    setup_ux._maybe_prompt_agent_skills(setup_agent="claude")

    assert globally_installed == ["claude"]


def test_globally_installed_harnesses_reports_all_agents_with_skills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    class _Skills:
        def status(self, *, agent: str, scope: str) -> SimpleNamespace:
            installed = agent in {"cursor", "opencode"}
            skills = (SimpleNamespace(id="potpie-cli"),) if installed else ()
            return SimpleNamespace(installed=skills)

    monkeypatch.setattr(
        "potpie_context_engine.adapters.inbound.cli.commands._common.get_host",
        lambda: SimpleNamespace(skills=_Skills()),
    )

    assert setup_ux._globally_installed_harnesses() == ["cursor", "opencode"]


def test_agent_usage_hint_formats_installed_harnesses() -> None:
    assert setup_ux._agent_usage_hint(["claude"]) == (
        "Open Claude — Potpie skills are ready to use."
    )
    assert setup_ux._agent_usage_hint(["claude", "cursor"]) == (
        "Open Claude and Cursor — Potpie skills are ready to use."
    )
    assert setup_ux._agent_usage_hint(["opencode", "codex", "cursor"]) == (
        "Open OpenCode, Codex, and Cursor — Potpie skills are ready to use."
    )
    assert setup_ux._agent_usage_hint([]) is None


def test_post_setup_wizard_runs_skills_after_integrations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    calls: list[str] = []

    monkeypatch.setattr(setup_ux, "is_interactive_tty", lambda: True)
    monkeypatch.setattr(
        "typer.confirm",
        lambda *_a, **_k: False,
    )

    def _checkbox(
        message: str, options: list[tuple[str, str]], **kwargs: object
    ) -> list[str]:
        if "integrations" in message.lower():
            return []
        calls.append("agents")
        return ["claude"]

    monkeypatch.setattr(interactive_prompts, "prompt_multi_checkbox", _checkbox)
    monkeypatch.setattr(
        setup_ux,
        "install_agents_globally",
        lambda agents: calls.extend(agents) or [(agent, object()) for agent in agents],
    )

    monkeypatch.setattr(setup_ux, "_maybe_prompt_first_pot", lambda **_k: None)
    setup_ux.maybe_prompt_github_login(repo=repo, setup_agent="claude")

    assert calls == ["agents", "claude"]

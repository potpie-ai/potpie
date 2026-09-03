"""Post-setup agent skill installation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from potpie.cli.ui import interactive_prompts, setup_ux
from potpie.cli.telemetry.context import TelemetryContext
from potpie.cli.telemetry.product_analytics import (
    ProductAnalyticsEvent,
    set_product_analytics_sink,
)


@dataclass
class _FakeSink:
    events: list[ProductAnalyticsEvent] = field(default_factory=list)

    def capture(self, event: ProductAnalyticsEvent) -> None:
        self.events.append(event)


@pytest.fixture()
def fake_sink(monkeypatch: pytest.MonkeyPatch) -> _FakeSink:
    sink = _FakeSink()
    set_product_analytics_sink(sink)
    monkeypatch.setattr(
        "potpie.cli.telemetry.product_analytics.current_telemetry_context",
        lambda: TelemetryContext(
            anonymous_install_id="install_123",
            invocation_id="invoke_456",
            daemon_session_id="daemon_789",
            environment="test",
            command="setup",
            subcommand=None,
            output_mode="human",
            cli_version="0.1.0",
            python_version="3.13.0",
            os="darwin",
            arch="arm64",
        ),
    )
    return sink


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
        lambda agents: (
            globally_installed.extend(agents) or [(agent, object()) for agent in agents]
        ),
    )

    setup_ux._maybe_prompt_agent_skills(setup_agent="claude")

    assert globally_installed == ["claude"]


def test_agent_skills_picker_emits_one_outcome_per_agent(
    monkeypatch: pytest.MonkeyPatch,
    fake_sink: _FakeSink,
) -> None:
    monkeypatch.setattr(
        interactive_prompts,
        "prompt_multi_checkbox",
        lambda *_a, **_k: ["codex", "claude"],
    )
    monkeypatch.setattr(setup_ux, "rich_enabled", lambda **_k: False)

    def _install(agents: list[str]) -> list[tuple[str, object]]:
        agent = agents[0]
        changed = ("potpie-cli",) if agent == "claude" else ()
        return [(agent, SimpleNamespace(changed=changed))]

    monkeypatch.setattr(setup_ux, "install_agents_globally", _install)

    setup_ux._maybe_prompt_agent_skills(setup_agent="claude")

    assert [event.name for event in fake_sink.events] == [
        "cli_onboarding_agent_skills_selection_outcome",
        "cli_onboarding_agent_skills_install_outcome",
        "cli_onboarding_agent_skills_install_outcome",
    ]
    selection, claude, codex = fake_sink.events
    assert selection.properties["selection_outcome"] == "selected"
    assert selection.properties["selected_agents"] == ("claude", "codex")
    assert selection.properties["selected_agent_count"] == 2
    assert selection.properties["entrypoint"] == "post_setup_agent_skills"
    assert "agent" not in selection.properties
    assert claude.properties["agent"] == "claude"
    assert claude.properties["outcome"] == "installed"
    assert codex.properties["agent"] == "codex"
    assert codex.properties["outcome"] == "already_installed"
    for event in (claude, codex):
        assert event.properties["entrypoint"] == "post_setup_agent_skills"
        assert event.properties["scope"] == "global"
        assert "failure_kind" not in event.properties
        assert "skill_id" not in event.properties


@pytest.mark.parametrize(
    ("prompt", "selection_outcome"),
    [
        (lambda *_a, **_k: [], "skipped"),
        (lambda *_a, **_k: (_ for _ in ()).throw(KeyboardInterrupt()), "cancelled"),
        (lambda *_a, **_k: (_ for _ in ()).throw(EOFError()), "cancelled"),
    ],
)
def test_agent_skills_picker_records_skipped_or_cancelled_selection(
    monkeypatch: pytest.MonkeyPatch,
    fake_sink: _FakeSink,
    prompt,
    selection_outcome: str,
) -> None:
    monkeypatch.setattr(interactive_prompts, "prompt_multi_checkbox", prompt)

    setup_ux._maybe_prompt_agent_skills(setup_agent="claude")

    assert [event.name for event in fake_sink.events] == [
        "cli_onboarding_agent_skills_selection_outcome",
    ]
    outcome = fake_sink.events[0]
    assert outcome.properties["selection_outcome"] == selection_outcome
    assert outcome.properties["selected_agent_count"] == 0
    assert outcome.properties["selected_agents"] == ()
    assert outcome.properties["entrypoint"] == "post_setup_agent_skills"
    assert "agent" not in outcome.properties


def test_agent_skills_picker_emits_failed_agent_outcome(
    monkeypatch: pytest.MonkeyPatch,
    fake_sink: _FakeSink,
) -> None:
    monkeypatch.setattr(
        interactive_prompts,
        "prompt_multi_checkbox",
        lambda *_a, **_k: ["codex"],
    )
    monkeypatch.setattr(setup_ux, "rich_enabled", lambda **_k: False)
    monkeypatch.setattr(
        setup_ux,
        "install_agents_globally",
        lambda _agents: (_ for _ in ()).throw(PermissionError("private path")),
    )

    with pytest.raises(PermissionError):
        setup_ux._maybe_prompt_agent_skills(setup_agent="codex")

    assert [event.name for event in fake_sink.events] == [
        "cli_onboarding_agent_skills_selection_outcome",
        "cli_onboarding_agent_skills_install_outcome",
    ]
    outcome = fake_sink.events[1]
    assert outcome.properties["agent"] == "codex"
    assert outcome.properties["entrypoint"] == "post_setup_agent_skills"
    assert outcome.properties["scope"] == "global"
    assert outcome.properties["outcome"] == "failed"
    assert outcome.properties["failure_kind"] == "permission_denied"
    assert "private path" not in outcome.properties.values()


def test_agent_skills_picker_keeps_prior_agent_outcomes_when_later_install_fails(
    monkeypatch: pytest.MonkeyPatch,
    fake_sink: _FakeSink,
) -> None:
    monkeypatch.setattr(
        interactive_prompts,
        "prompt_multi_checkbox",
        lambda *_a, **_k: ["claude", "codex"],
    )
    monkeypatch.setattr(setup_ux, "rich_enabled", lambda **_k: False)

    def _install(agents: list[str]) -> list[tuple[str, object]]:
        agent = agents[0]
        if agent == "codex":
            raise PermissionError("private path")
        return [(agent, SimpleNamespace(changed=("potpie-cli",)))]

    monkeypatch.setattr(setup_ux, "install_agents_globally", _install)

    with pytest.raises(PermissionError):
        setup_ux._maybe_prompt_agent_skills(setup_agent="claude")

    install_events = [
        event
        for event in fake_sink.events
        if event.name == "cli_onboarding_agent_skills_install_outcome"
    ]
    assert [event.properties["agent"] for event in install_events] == [
        "claude",
        "codex",
    ]
    assert install_events[0].properties["outcome"] == "installed"
    assert install_events[1].properties["outcome"] == "failed"
    assert install_events[1].properties["failure_kind"] == "permission_denied"


def test_agent_skills_picker_marks_in_flight_agent_cancelled(
    monkeypatch: pytest.MonkeyPatch,
    fake_sink: _FakeSink,
) -> None:
    monkeypatch.setattr(
        interactive_prompts,
        "prompt_multi_checkbox",
        lambda *_a, **_k: ["claude", "codex"],
    )
    monkeypatch.setattr(setup_ux, "rich_enabled", lambda **_k: False)

    def _install(agents: list[str]) -> list[tuple[str, object]]:
        agent = agents[0]
        if agent == "codex":
            raise KeyboardInterrupt
        return [(agent, SimpleNamespace(changed=("potpie-cli",)))]

    monkeypatch.setattr(setup_ux, "install_agents_globally", _install)

    with pytest.raises(KeyboardInterrupt):
        setup_ux._maybe_prompt_agent_skills(setup_agent="claude")

    install_events = [
        event
        for event in fake_sink.events
        if event.name == "cli_onboarding_agent_skills_install_outcome"
    ]
    assert [event.properties["agent"] for event in install_events] == [
        "claude",
        "codex",
    ]
    assert install_events[0].properties["outcome"] == "installed"
    assert install_events[1].properties["outcome"] == "cancelled"
    assert "failure_kind" not in install_events[1].properties


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
        "potpie.cli.commands._common.get_runtime",
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

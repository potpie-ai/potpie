"""Animated setup wizard rendered over the REAL host-routed setup.

``host.setup.run(plan)`` returns a fully-completed :class:`SetupReport` (every
step already executed). This module replays those steps through the Rich
checklist so first-run reads as live — but each row's *final* state is the real
``StepResult.state``, and a failed hard step shows the real ``detail`` and exits
degraded. JSON and non-interactive callers never reach here: ``commands/bootstrap``
routes them to the plain ``emit`` path instead.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from adapters.inbound.cli.ui.setup_wizard_ui import (
    SetupWizardUI,
    StepStatus,
    is_interactive_tty,
    rich_ui_enabled,
)

# Real orchestrator step_id -> (running label, done label). See
# application/services/setup_orchestrator.py for the canonical step list.
STEP_LABELS: dict[str, tuple[str, str]] = {
    "config": ("Creating config files…", "Config files ready"),
    "installer": ("Installing CLI / service…", "CLI ready"),
    "backend.provision": ("Provisioning graph backend…", "Backend ready"),
    "pot.init": ("Initializing control plane…", "Control plane ready"),
    "state_store.provision": ("Preparing state store…", "State store ready"),
    "migrator.migrate": ("Running migrations…", "Migrations applied"),
    "pot.default": ("Creating default pot…", "Default pot created"),
    "daemon": ("Starting daemon…", "Daemon ready"),
    "auth": ("Initializing local auth…", "Local auth ready"),
    "source": ("Registering repo…", "Repo registered"),
    "skills": ("Installing agent skills…", "Skills installed"),
    "scan": ("Scanning repository…", "Repository scanned"),
}

# Rows that get the animated "chomp" glyph while running.
CHOMP_STEPS = frozenset(
    {"backend.provision", "state_store.provision", "daemon", "scan"}
)

# Lifecycle StepResult.state -> wizard ChecklistStep status. A soft
# not_implemented step is informational (warn), not a failure.
STATE_MAP: dict[str, StepStatus] = {
    "done": "done",
    "skipped": "skipped",
    "failed": "failed",
    "planned": "pending",
    "not_implemented": "warn",
}

# Synthetic per-row dwell so the (already-complete) run reads as live.
_MIN_DWELL_S = 0.18

# Soft steps that are hidden when they only report no-op / stub outcomes.
_UI_HIDE_WHEN: dict[str, frozenset[str]] = {
    "auth": frozenset({"not_implemented", "skipped"}),
    "source": frozenset({"skipped"}),
    "state_store.provision": frozenset({"skipped"}),
}


def _visible_in_checklist(step: Any) -> bool:
    hidden = _UI_HIDE_WHEN.get(step.step)
    if hidden is None:
        return True
    return step.state not in hidden


def rich_enabled(*, as_json: bool) -> bool:
    """True when the animated wizard should drive output (TTY, not --json)."""
    return rich_ui_enabled(as_json=as_json)


def render_setup_report(
    report: Any,
    *,
    repo: Path,
    agent: str,
    scan: bool,
    use_rich: bool,
    pot_name: str | None = None,
) -> None:
    """Replay a completed ``SetupReport`` through the animated checklist."""
    wizard = SetupWizardUI(use_rich=use_rich)
    visible_steps = [step for step in report.steps if _visible_in_checklist(step)]
    for step in visible_steps:
        running, done = STEP_LABELS.get(step.step, (step.step, step.step))
        if step.step == "skills":
            running = f"Installing agent skills ({agent})…"
        wizard.add_step(
            step.step, running, chomp=step.step in CHOMP_STEPS, done_label=done
        )

    wizard.run_intro(repo=repo, agent=agent, scan=scan)

    failed_step_id: str | None = None
    with wizard.live():
        for step in visible_steps:
            with wizard.run_step(step.step):
                if use_rich:
                    time.sleep(_MIN_DWELL_S)
                row = wizard.get(step.step)
                row.status = STATE_MAP.get(step.state, "warn")
                if step.detail:
                    row.detail = step.detail
            if step.state == "failed" and step.hard and failed_step_id is None:
                failed_step_id = step.step
                break

    if failed_step_id is not None:
        wizard.print_failed(step_id=failed_step_id)
        return

    wizard.print_complete_summary(
        pot_name=None if report.plan.defer_default_pot else (pot_name or report.plan.pot),
        already_setup=False,
    )


# Post-setup integration picker (order used for sequential login).
POST_SETUP_INTEGRATION_OPTIONS: tuple[tuple[str, str], ...] = (
    ("linear", "Linear"),
    ("jira", "Jira"),
    ("confluence", "Confluence"),
)
POST_SETUP_INTEGRATION_ORDER: tuple[str, ...] = tuple(
    option_id for option_id, _ in POST_SETUP_INTEGRATION_OPTIONS
)

# Agent harnesses supported by ``install_agent_bundle`` (repo-local templates).
POST_SETUP_AGENT_OPTIONS: tuple[tuple[str, str], ...] = (
    ("claude", "Claude"),
    ("cursor", "Cursor"),
    ("opencode", "OpenCode"),
    ("codex", "Codex"),
    ("default", "AGENTS.md (OpenAI / generic)"),
)
POST_SETUP_AGENT_ORDER: tuple[str, ...] = tuple(
    option_id for option_id, _ in POST_SETUP_AGENT_OPTIONS
)


def install_agents_to_repo(repo: Path, agents: list[str]) -> list[tuple[str, Any]]:
    """Copy packaged skill bundles into *repo* for each harness id."""
    from adapters.outbound.skills.agent_installer import AGENT_TYPES, install_agent_bundle

    results: list[tuple[str, Any]] = []
    for agent in agents:
        key = agent.strip().lower()
        if key not in AGENT_TYPES:
            continue
        results.append((key, install_agent_bundle(repo, agent=key)))
    return results


def install_agents_globally(agents: list[str]) -> list[tuple[str, Any]]:
    """Install packaged skill bundles into each harness's global skill location."""
    from adapters.inbound.cli.commands._common import get_host
    from adapters.outbound.skills.agent_installer import AGENT_TYPES

    host = get_host()
    results: list[tuple[str, Any]] = []
    for agent in agents:
        key = agent.strip().lower()
        if key not in AGENT_TYPES or key == "default":
            continue
        results.append((key, host.skills.install(agent=key, scope="global")))
    return results


def _agent_label(agent: str) -> str:
    labels = dict(POST_SETUP_AGENT_OPTIONS)
    key = agent.strip().lower()
    if key in labels:
        return labels[key]
    return key.replace("_", " ").title()


def _install_agents_globally_with_progress(agents: list[str]) -> list[tuple[str, Any]]:
    from adapters.inbound.cli.ui.output import print_plain_line

    results: list[tuple[str, Any]] = []
    if rich_enabled(as_json=False):
        from rich.console import Group
        from rich.live import Live
        from rich.text import Text

        from adapters.inbound.cli.ui.brand import LOGO_STYLE, UI_MUTED_STYLE
        from adapters.inbound.cli.ui.setup_wizard_ui import stderr_console

        completed: list[tuple[str, str]] = []

        def render(active_label: str | None = None, dots: str = "") -> Group:
            lines = [Text("Potpie skills", style="bold")]
            for label, status in completed:
                lines.append(Text(f"  ✓ {label} ({status})", style=LOGO_STYLE))
            if active_label:
                lines.append(
                    Text(f"  Installing {active_label} skills{dots}", style=UI_MUTED_STYLE)
                )
            return Group(*lines)

        with Live(render(), console=stderr_console(), refresh_per_second=12) as live:
            for agent in agents:
                label = _agent_label(agent)
                for dots in ("   ", ".  ", ".. ", "...", " ..", "  ."):
                    live.update(render(label, dots))
                    time.sleep(0.06)
                result = install_agents_globally([agent])
                results.extend(result)
                if _result_changed(result):
                    completed.append((label, "installed"))
                else:
                    completed.append((label, "already installed"))
                live.update(render())
        return results

    for agent in agents:
        label = _agent_label(agent)
        print_plain_line(
            f"Installing Potpie skills for {label}...",
            as_json=False,
            markup=False,
        )
        result = install_agents_globally([agent])
        results.extend(result)
        if _result_changed(result):
            print_plain_line(
                f"✓ {label} (installed)",
                as_json=False,
                markup=False,
            )
        else:
            print_plain_line(
                f"✓ {label} (already installed)",
                as_json=False,
                markup=False,
            )
    return results


def _result_changed(result: list[tuple[str, Any]]) -> bool:
    return any(bool(getattr(item, "changed", ())) for _, item in result)


def _format_agent_list(labels: list[str]) -> str:
    if len(labels) <= 1:
        return labels[0] if labels else ""
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def _agent_usage_hint(agent_ids: list[str]) -> str | None:
    if not agent_ids:
        return None
    labels = [_agent_label(agent_id) for agent_id in agent_ids]
    return (
        f"Open {_format_agent_list(labels)} — Potpie skills are ready to use."
    )


def _globally_installed_harnesses() -> list[str]:
    """Harnesses that already have Potpie skills on disk (any prior setup run)."""
    from adapters.inbound.cli.commands._common import get_host

    host = get_host()
    installed: list[str] = []
    for agent in POST_SETUP_AGENT_ORDER:
        if agent == "default":
            continue
        try:
            status = host.skills.status(agent=agent, scope="global")
        except ValueError:
            continue
        if status.installed:
            installed.append(agent)
    return installed


def _try_login(handler) -> None:
    """Run a login handler; continue setup when the user declines or auth fails."""
    import typer

    try:
        handler()
    except typer.Exit:
        pass
    except (KeyboardInterrupt, EOFError):
        raise


def _maybe_prompt_agent_skills(*, setup_agent: str) -> None:
    from adapters.inbound.cli.ui.interactive_prompts import prompt_multi_checkbox

    valid = frozenset(agent for agent in POST_SETUP_AGENT_ORDER if agent != "default")
    default_checked = (
        frozenset({setup_agent.strip().lower()})
        if setup_agent.strip().lower() in valid
        else frozenset()
    )
    try:
        selected = prompt_multi_checkbox(
            "Which agent harnesses should receive Potpie skills globally?",
            [
                (option_id, label)
                for option_id, label in POST_SETUP_AGENT_OPTIONS
                if option_id != "default"
            ],
            default_checked=default_checked,
        )
    except (KeyboardInterrupt, EOFError):
        return

    if not selected:
        return

    selected_set = frozenset(selected)
    agents = [
        agent
        for agent in POST_SETUP_AGENT_ORDER
        if agent in selected_set and agent != "default"
    ]
    _install_agents_globally_with_progress(agents)


def maybe_prompt_github_login(
    *,
    repo: Path | None = None,
    setup_agent: str = "claude",
    default_pot_name: str = "foo-pot",
) -> None:
    """After setup: GitHub, integrations, global skills, then first pot naming."""
    if not is_interactive_tty():
        return

    from adapters.inbound.cli.auth.auth_commands import run_integration_login
    from adapters.inbound.cli.auth.github_commands import github_login_impl
    from adapters.inbound.cli.ui.interactive_prompts import prompt_multi_checkbox

    import typer

    try:
        confirmed = typer.confirm(
            "Would you like to log in to GitHub now?",
            default=True,
        )
    except (KeyboardInterrupt, EOFError):
        confirmed = False
    if confirmed:
        _try_login(github_login_impl)

    try:
        selected = prompt_multi_checkbox(
            "Which integrations would you like to connect?",
            list(POST_SETUP_INTEGRATION_OPTIONS),
        )
    except (KeyboardInterrupt, EOFError):
        selected = []

    if selected:
        selected_set = frozenset(selected)
        for provider in POST_SETUP_INTEGRATION_ORDER:
            if provider not in selected_set:
                continue
            _try_login(lambda p=provider: run_integration_login(p))

    if repo is not None:
        _maybe_prompt_agent_skills(setup_agent=setup_agent)

    _maybe_prompt_first_pot(repo=repo, default_pot_name=default_pot_name)


def _register_repo_source(*, repo: str) -> None:
    from adapters.inbound.cli.commands._common import get_host

    host = get_host()
    active = host.pots.active_pot()
    if active is None:
        return
    existing = host.pots.list_sources(pot_id=active.pot_id)
    if any(s.kind == "repo" and s.name == repo for s in existing):
        return
    host.pots.add_source(pot_id=active.pot_id, kind="repo", location=repo)


def _maybe_prompt_first_pot(
    *,
    repo: Path | None,
    default_pot_name: str,
) -> None:
    from adapters.inbound.cli.commands._common import get_host
    from adapters.inbound.cli.ui.interactive_prompts import prompt_first_pot_name
    from adapters.inbound.cli.ui.potpie_logo_anim import play_setup_finish
    from adapters.inbound.cli.ui.setup_wizard_ui import stderr_console

    try:
        name = prompt_first_pot_name(default=default_pot_name).strip()
    except (KeyboardInterrupt, EOFError):
        return
    if not name:
        name = default_pot_name

    repo_str = str(repo.resolve()) if repo is not None else None
    pot = get_host().pots.create_pot(name=name, repo=repo_str, use=True)
    if repo_str:
        _register_repo_source(repo=repo_str)

    play_setup_finish(
        stderr_console(),
        pot_name=pot.name,
        agent_hint=_agent_usage_hint(_globally_installed_harnesses()),
    )


__all__ = [
    "STEP_LABELS",
    "STATE_MAP",
    "CHOMP_STEPS",
    "rich_enabled",
    "render_setup_report",
    "maybe_prompt_github_login",
    "install_agents_globally",
    "install_agents_to_repo",
    "POST_SETUP_INTEGRATION_OPTIONS",
    "POST_SETUP_INTEGRATION_ORDER",
    "POST_SETUP_AGENT_OPTIONS",
    "POST_SETUP_AGENT_ORDER",
]

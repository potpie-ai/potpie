"""Setup wizard UI for the real host-routed setup flow."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import click

from potpie.cli.commands._common import get_store
from potpie.cli.repo_location import resolve_repo_location
from potpie.cli.telemetry.onboarding_events import (
    capture_github_prompt_outcome,
    capture_github_prompt_shown,
    capture_integration_auth_event,
    capture_onboarding_event,
    capture_project_binding_event,
    capture_wizard_event,
    elapsed_ms,
    now_ms,
    onboarding_entrypoint,
    sanitized_failure_kind,
)
from potpie.cli.ui.setup_wizard_ui import (
    SetupWizardUI,
    StepStatus,
    is_interactive_tty,
    live_ui_enabled,
)
from potpie_context_engine.domain.lifecycle import SetupPlan, SetupReport
from potpie_context_engine.domain.ports.services.setup import (
    SetupObserver,
    SetupOrchestrator,
)

# Real orchestrator step_id -> (running label, done label). See
# application/services/setup_orchestrator.py for the canonical step list.
STEP_LABELS: dict[str, tuple[str, str]] = {
    "config": ("Creating config files…", "Config files ready"),
    "installer": ("Installing CLI / service…", "CLI ready"),
    "embeddings.model": (
        "Setting up embedding model…",
        "Embedding model ready",
    ),
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
    {"embeddings.model", "backend.provision", "state_store.provision", "daemon", "scan"}
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
    "auth": frozenset({"done", "failed", "not_implemented", "skipped"}),
    "source": frozenset({"skipped"}),
    "state_store.provision": frozenset({"skipped"}),
}


def _visible_in_checklist(step: Any) -> bool:
    hidden = _UI_HIDE_WHEN.get(step.step)
    if hidden is None:
        return True
    return step.state not in hidden


def rich_enabled(*, as_json: bool) -> bool:
    """True when setup may use repainting Rich Live regions."""
    return live_ui_enabled(as_json=as_json)


def interactive_onboarding_enabled(*, as_json: bool) -> bool:
    """True when setup can ask post-run onboarding prompts."""
    if as_json:
        return False
    return is_interactive_tty()


class _CompositeSetupObserver:
    def __init__(self, *observers: SetupObserver | None) -> None:
        self._observers = tuple(
            observer for observer in observers if observer is not None
        )

    def step_started(self, *, step: str, hard: bool) -> None:
        for observer in self._observers:
            try:
                observer.step_started(step=step, hard=hard)
            except Exception:  # noqa: BLE001,S112 - one observer must not block another.
                continue

    def step_completed(self, *, result: Any, duration_ms: int) -> None:
        for observer in self._observers:
            try:
                observer.step_completed(result=result, duration_ms=duration_ms)
            except Exception:  # noqa: BLE001,S112 - one observer must not block another.
                continue


class _WizardSetupObserver:
    def __init__(self, wizard: SetupWizardUI) -> None:
        self._wizard = wizard
        self._visible_steps = {step.step_id for step in wizard.steps}

    def step_started(self, *, step: str, hard: bool) -> None:
        del hard
        if step not in self._visible_steps:
            return
        try:
            self._wizard.start_step(step)
        except KeyError:
            return

    def step_completed(self, *, result: Any, duration_ms: int) -> None:
        if result.step not in self._visible_steps:
            return
        try:
            self._wizard.complete_step(
                result.step,
                status=STATE_MAP.get(result.state, "warn"),
                detail=result.detail,
                duration_ms=duration_ms,
            )
        except KeyError:
            return


class _PlainSetupObserver:
    def __init__(self, planned_steps: list[Any], *, agent: str) -> None:
        self._visible_steps = {step.step for step in planned_steps}
        self._agent = agent

    def step_started(self, *, step: str, hard: bool) -> None:
        del hard
        if step not in self._visible_steps:
            return
        running, _done = _step_labels(step, agent=self._agent)
        click.echo(f"› {running}", err=True)

    def step_completed(self, *, result: Any, duration_ms: int) -> None:
        del duration_ms
        if result.step not in self._visible_steps:
            return
        _running, done = _step_labels(result.step, agent=self._agent)
        mark = {
            "done": "✓",
            "skipped": "–",
            "failed": "✗",
            "not_implemented": "!",
        }.get(result.state, "!")
        line = f"{mark} {done}"
        if result.detail:
            line = f"{line} {result.detail}"
        click.echo(line, err=True)


def run_setup_live(
    setup: SetupOrchestrator,
    plan: SetupPlan,
    *,
    repo: Path,
    agent: str,
    scan: bool,
    use_rich: bool,
    config_home: Path | None = None,
    pot_name: str | None = None,
    observer: SetupObserver | None = None,
) -> SetupReport:
    """Run setup while the Rich checklist reflects the real active step."""
    started_ms = now_ms()
    capture_wizard_event("cli_onboarding_wizard_shown")
    wizard = SetupWizardUI(use_rich=use_rich)
    preview = setup.preview(plan)
    visible_steps = [step for step in preview.steps if _visible_in_preview(step)]
    for step in visible_steps:
        running, done = _step_labels(step.step, agent=agent)
        wizard.add_step(
            step.step, running, chomp=step.step in CHOMP_STEPS, done_label=done
        )

    wizard.run_intro(repo=repo, agent=agent, scan=scan)
    setup.set_observer(_CompositeSetupObserver(_WizardSetupObserver(wizard), observer))
    with wizard.live():
        report = setup.run(plan)

    failed_step_id = _first_hard_failed_step(report)
    if failed_step_id is not None:
        wizard.print_failed(step_id=failed_step_id)
        capture_wizard_event(
            "cli_onboarding_wizard_failed_view_shown",
            duration_ms=elapsed_ms(started_ms),
            failed_step=failed_step_id,
        )
        return report

    _print_complete_summary(
        wizard,
        report,
        config_home=config_home,
        pot_name=pot_name,
        started_ms=started_ms,
    )
    return report


def run_setup_plain(
    setup: SetupOrchestrator,
    plan: SetupPlan,
    *,
    repo: Path,
    agent: str,
    scan: bool,
    observer: SetupObserver | None = None,
) -> SetupReport:
    """Run setup with deterministic streaming lines for non-live terminals."""
    preview = setup.preview(plan)
    visible_steps = [step for step in preview.steps if _visible_in_preview(step)]
    click.echo("Potpie Setup", err=True)
    click.echo("Setting up local onboarding for this repo.", err=True)
    extras = f"repo={repo.resolve()} agent={agent}"
    if scan:
        extras += " scan=on"
    click.echo(extras, err=True)
    click.echo("", err=True)
    setup.set_observer(
        _CompositeSetupObserver(
            _PlainSetupObserver(visible_steps, agent=agent),
            observer,
        )
    )
    return setup.run(plan)


def render_setup_report(
    report: Any,
    *,
    repo: Path,
    agent: str,
    scan: bool,
    use_rich: bool,
    config_home: Path | None = None,
    pot_name: str | None = None,
) -> None:
    """Replay a completed ``SetupReport`` through the animated checklist."""
    started_ms = now_ms()
    capture_wizard_event("cli_onboarding_wizard_shown")
    wizard = SetupWizardUI(use_rich=use_rich)
    visible_steps = [step for step in report.steps if _visible_in_checklist(step)]
    for step in visible_steps:
        running, done = _step_labels(step.step, agent=agent)
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
        capture_wizard_event(
            "cli_onboarding_wizard_failed_view_shown",
            duration_ms=elapsed_ms(started_ms),
            failed_step=failed_step_id,
        )
        return

    _print_complete_summary(
        wizard,
        report,
        config_home=config_home,
        pot_name=pot_name,
        started_ms=started_ms,
    )


def _visible_in_preview(step: Any) -> bool:
    if step.step == "auth":
        return False
    if step.step == "source" and step.skip_reason:
        return False
    if step.step == "state_store.provision" and step.skip_reason:
        return False
    return True


def _first_hard_failed_step(report: Any) -> str | None:
    for step in report.steps:
        if step.hard and step.state == "failed":
            return step.step
    return None


def _print_complete_summary(
    wizard: SetupWizardUI,
    report: Any,
    *,
    config_home: Path | None,
    pot_name: str | None,
    started_ms: int,
) -> None:
    setup_path = str(config_home / "config.json") if config_home else ""
    data_path = str(config_home) if config_home else ""
    wizard.print_complete_summary(
        setup_path=setup_path,
        data_path=data_path,
        pot_name=None
        if report.plan.defer_default_pot
        else (pot_name or report.plan.pot),
        already_setup=False,
    )
    capture_wizard_event(
        "cli_onboarding_wizard_completed",
        duration_ms=elapsed_ms(started_ms),
    )


def _step_labels(step_id: str, *, agent: str) -> tuple[str, str]:
    running, done = STEP_LABELS.get(step_id, (step_id, step_id))
    if step_id == "skills":
        running = f"Installing agent skills ({agent})…"
    return running, done


# Post-setup integration picker (order used for sequential login).
POST_SETUP_INTEGRATION_OPTIONS: tuple[tuple[str, str], ...] = (
    ("linear", "Linear"),
    ("atlassian", "Atlassian"),
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
    from potpie.skills.installer import (
        AGENT_TYPES,
        install_agent_bundle,
    )
    from potpie.runtime import cli_template_resources

    results: list[tuple[str, Any]] = []
    template_resources = cli_template_resources()
    for agent in agents:
        key = agent.strip().lower()
        if key not in AGENT_TYPES:
            continue
        results.append(
            (
                key,
                install_agent_bundle(
                    repo,
                    agent=key,
                    template_resources=template_resources,
                ),
            )
        )
    return results


def install_agents_globally(agents: list[str]) -> list[tuple[str, Any]]:
    """Install packaged skill bundles into each harness's global skill location."""
    from potpie.cli.commands._common import get_cli_runtime
    from potpie.skills.installer import AGENT_TYPES

    runtime = get_cli_runtime()
    results: list[tuple[str, Any]] = []
    for agent in agents:
        key = agent.strip().lower()
        if key not in AGENT_TYPES or key == "default":
            continue
        results.append((key, runtime.skills.install(agent=key, scope="global")))
    return results


def _agent_label(agent: str) -> str:
    labels = dict(POST_SETUP_AGENT_OPTIONS)
    key = agent.strip().lower()
    if key in labels:
        return labels[key]
    return key.replace("_", " ").title()


def _install_agents_globally_with_progress(agents: list[str]) -> list[tuple[str, Any]]:
    from potpie.cli.ui.output import print_plain_line

    results: list[tuple[str, Any]] = []
    if rich_enabled(as_json=False):
        from rich.console import Group
        from rich.live import Live
        from rich.text import Text

        from potpie.cli.ui.brand import LOGO_STYLE, UI_MUTED_STYLE
        from potpie.cli.ui.setup_wizard_ui import stderr_console

        completed: list[tuple[str, str]] = []

        def render(active_label: str | None = None, dots: str = "") -> Group:
            lines = [Text("Potpie skills", style="bold")]
            for label, status in completed:
                lines.append(Text(f"  ✓ {label} ({status})", style=LOGO_STYLE))
            if active_label:
                lines.append(
                    Text(
                        f"  Installing {active_label} skills{dots}",
                        style=UI_MUTED_STYLE,
                    )
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
    return f"Open {_format_agent_list(labels)} — Potpie skills are ready to use."


def _globally_installed_harnesses() -> list[str]:
    """Harnesses that already have Potpie skills on disk (any prior setup run)."""
    from potpie.cli.commands._common import get_cli_runtime

    runtime = get_cli_runtime()
    installed: list[str] = []
    for agent in POST_SETUP_AGENT_ORDER:
        if agent == "default":
            continue
        try:
            status = runtime.skills.status(agent=agent, scope="global")
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
    except typer.Exit as exc:
        if getattr(exc, "exit_code", None) == 130:
            raise
        pass
    except (KeyboardInterrupt, EOFError):
        raise


def _integration_label(provider: str) -> str:
    labels = dict(POST_SETUP_INTEGRATION_OPTIONS)
    key = provider.strip().lower()
    return labels.get(key, key.replace("_", " ").title())


def _integration_login_command(provider: str) -> str:
    key = provider.strip().lower()
    if key == "atlassian":
        key = "jira"
    return f"potpie {key} login"


def _print_integration_skipped(provider: str) -> None:
    from potpie.cli.ui.format import print_line

    label = _integration_label(provider)
    command = _integration_login_command(provider)
    message = f"Skipped {label} — connect later with `{command}`."
    print_line("", as_json=False, markup=False)
    print_line(message, as_json=False, tone="skipped")
    print_line("", as_json=False, markup=False)


def _integration_login_aborted(exc: BaseException) -> bool:
    if isinstance(exc, (KeyboardInterrupt, EOFError, click.Abort)):
        return True
    return type(exc).__name__ == "Abort"


def _try_integration_login(provider: str) -> None:
    """Run one post-setup integration login; Ctrl+C skips without aborting setup."""
    from potpie.auth.auth_commands import run_integration_login

    import typer

    started_ms = now_ms()
    try:
        run_integration_login(provider)
    except typer.Exit:
        pass
    except BaseException as exc:
        if not _integration_login_aborted(exc):
            raise
        capture_integration_auth_event(
            "cli_onboarding_integration_auth_skipped",
            provider=provider,
            entrypoint="post_setup_integration_picker",
            duration_ms=elapsed_ms(started_ms),
            failure_kind="user_aborted",
        )
        _print_integration_skipped(provider)


def _github_status() -> dict[str, Any]:
    return get_store().get_integration_status("github")


def _github_already_authenticated() -> bool:
    try:
        status = _github_status()
    except Exception:  # noqa: BLE001
        return False
    if not bool(status.get("authenticated")):
        return False
    login = str(status.get("login") or "").strip()
    suffix = f" as {login}" if login else ""
    from potpie.cli.ui.output import print_plain_line

    print_plain_line(
        f"GitHub already connected{suffix}; skipping login.",
        as_json=False,
        markup=False,
    )
    return True


def _maybe_prompt_agent_skills(*, setup_agent: str) -> None:
    from potpie.cli.ui.interactive_prompts import prompt_multi_checkbox

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
    started_ms = now_ms()
    capture_project_binding_event(
        "cli_onboarding_agent_skills_install_started",
        entrypoint="post_setup_agent_skills",
        properties={"agent_count": len(agents)},
    )
    try:
        result = _install_agents_globally_with_progress(agents)
    except Exception as exc:  # noqa: BLE001
        capture_project_binding_event(
            "cli_onboarding_agent_skills_install_failed",
            entrypoint="post_setup_agent_skills",
            properties={
                "duration_ms": elapsed_ms(started_ms),
                "failure_kind": sanitized_failure_kind(exc),
            },
        )
        raise
    capture_project_binding_event(
        "cli_onboarding_agent_skills_install_completed",
        entrypoint="post_setup_agent_skills",
        properties={
            "changed": _result_changed(result),
            "duration_ms": elapsed_ms(started_ms),
        },
    )


def maybe_prompt_github_login(
    *,
    repo: Path | None = None,
    setup_agent: str = "claude",
    default_pot_name: str = "foo-pot",
) -> None:
    """After setup: GitHub, integrations, global skills, then first pot naming."""
    if not is_interactive_tty():
        capture_github_prompt_outcome("skipped", duration_ms=0)
        return

    from potpie.auth.github_commands import github_login_impl
    from potpie.cli.ui.interactive_prompts import prompt_multi_checkbox

    import typer

    if not _github_already_authenticated():
        prompt_started_ms = now_ms()
        capture_github_prompt_shown(default_answer=True)
        try:
            confirmed = typer.confirm(
                "Would you like to log in to GitHub now?",
                default=True,
            )
        except (KeyboardInterrupt, EOFError):
            capture_github_prompt_outcome(
                "aborted",
                duration_ms=elapsed_ms(prompt_started_ms),
            )
            confirmed = False
        else:
            capture_github_prompt_outcome(
                "accepted" if confirmed else "declined",
                duration_ms=elapsed_ms(prompt_started_ms),
            )
        if confirmed:
            with onboarding_entrypoint("post_setup_github_prompt"):
                _try_login(github_login_impl)
    else:
        capture_github_prompt_outcome("skipped", duration_ms=0)

    capture_onboarding_event(
        "cli_onboarding_integration_picker_shown",
        phase="integration_auth",
        entrypoint="post_setup_integration_picker",
        properties={"available_provider_count": len(POST_SETUP_INTEGRATION_OPTIONS)},
    )
    try:
        selected = prompt_multi_checkbox(
            "Which integrations would you like to connect?",
            list(POST_SETUP_INTEGRATION_OPTIONS),
        )
    except (KeyboardInterrupt, EOFError):
        selected = []
    capture_onboarding_event(
        "cli_onboarding_integration_picker_completed",
        phase="integration_auth",
        entrypoint="post_setup_integration_picker",
        properties={
            "selected_provider_count": len(selected),
            "selected_providers": tuple(selected),
        },
    )

    if selected:
        selected_set = frozenset(selected)
        for provider in POST_SETUP_INTEGRATION_ORDER:
            if provider not in selected_set:
                continue
            with onboarding_entrypoint("post_setup_integration_picker"):
                _try_integration_login(provider)

    if repo is not None:
        capture_project_binding_event(
            "cli_onboarding_project_binding_started",
            entrypoint="post_setup_first_pot",
            properties={"repo_provided": True, "agent": setup_agent},
        )
        _maybe_prompt_agent_skills(setup_agent=setup_agent)

    _maybe_prompt_first_pot(repo=repo, default_pot_name=default_pot_name)


def _register_repo_source(*, repo: str) -> str:
    from potpie.cli.commands._common import get_host
    from potpie.cli.commands.pots import register_repo_source

    started_ms = now_ms()
    capture_project_binding_event(
        "cli_onboarding_repo_source_add_started",
        entrypoint="post_setup_first_pot",
        properties={"source_kind": "repo"},
    )
    host = get_host()
    active = host.pots.active_pot()
    if active is None:
        capture_project_binding_event(
            "cli_onboarding_repo_source_add_completed",
            entrypoint="post_setup_first_pot",
            properties={"step_state": "skipped", "duration_ms": elapsed_ms(started_ms)},
        )
        return "skipped"
    existing = host.pots.list_sources(pot_id=active.pot_id)
    resolved = resolve_repo_location(repo)
    if any(
        s.kind == "repo"
        and (
            getattr(s, "location", None) == resolved
            or getattr(s, "name", None) == resolved
        )
        for s in existing
    ):
        capture_project_binding_event(
            "cli_onboarding_repo_source_add_completed",
            entrypoint="post_setup_first_pot",
            properties={"step_state": "skipped", "duration_ms": elapsed_ms(started_ms)},
        )
        return "skipped"
    try:
        register_repo_source(host, pot_id=active.pot_id, location=repo)
    except Exception as exc:  # noqa: BLE001
        capture_project_binding_event(
            "cli_onboarding_repo_source_add_failed",
            entrypoint="post_setup_first_pot",
            properties={
                "failure_kind": sanitized_failure_kind(exc),
                "duration_ms": elapsed_ms(started_ms),
            },
        )
        raise
    capture_project_binding_event(
        "cli_onboarding_repo_source_add_completed",
        entrypoint="post_setup_first_pot",
        properties={"step_state": "done", "duration_ms": elapsed_ms(started_ms)},
    )
    return "done"


def _maybe_prompt_first_pot(
    *,
    repo: Path | None,
    default_pot_name: str,
) -> None:
    from potpie.cli.commands._common import get_host
    from potpie.cli.ui.interactive_prompts import prompt_first_pot_name
    from potpie.cli.ui.potpie_logo_anim import play_setup_finish
    from potpie.cli.ui.setup_wizard_ui import stderr_console

    capture_project_binding_event(
        "cli_onboarding_first_pot_prompt_shown",
        entrypoint="post_setup_first_pot",
        properties={"repo_provided": repo is not None},
    )
    try:
        name = prompt_first_pot_name(default=default_pot_name).strip()
    except (KeyboardInterrupt, EOFError):
        capture_project_binding_event(
            "cli_onboarding_project_binding_incomplete",
            entrypoint="post_setup_first_pot",
            properties={"missing_piece": "first_pot"},
        )
        return
    if not name:
        name = default_pot_name

    repo_str = str(repo.resolve()) if repo is not None else None
    name_source = "default" if name == default_pot_name else "custom"
    pot = get_host().pots.create_pot(name=name, use=True)
    capture_project_binding_event(
        "cli_onboarding_first_pot_created",
        entrypoint="post_setup_first_pot",
        properties={"pot_name_source": name_source, "repo_provided": repo is not None},
    )
    source_state = "missing"
    if repo_str:
        source_state = _register_repo_source(repo=repo_str)
    capture_project_binding_event(
        "cli_onboarding_project_binding_completed",
        entrypoint="post_setup_first_pot",
        properties={"source_state": source_state},
    )

    agent_hint = _agent_usage_hint(_globally_installed_harnesses())
    if rich_enabled(as_json=False):
        from potpie.cli.ui.potpie_logo_anim import play_setup_finish
        from potpie.cli.ui.setup_wizard_ui import stderr_console

        play_setup_finish(
            stderr_console(),
            pot_name=pot.name,
            agent_hint=agent_hint,
        )
        return

    from potpie.cli.ui.output import print_plain_line

    print_plain_line(f"✓ Pot {pot.name} is ready.", as_json=False, markup=False)
    if agent_hint:
        print_plain_line(agent_hint, as_json=False, markup=False)


__all__ = [
    "STEP_LABELS",
    "STATE_MAP",
    "CHOMP_STEPS",
    "rich_enabled",
    "interactive_onboarding_enabled",
    "run_setup_live",
    "run_setup_plain",
    "render_setup_report",
    "maybe_prompt_github_login",
    "install_agents_globally",
    "install_agents_to_repo",
    "POST_SETUP_INTEGRATION_OPTIONS",
    "POST_SETUP_INTEGRATION_ORDER",
    "POST_SETUP_AGENT_OPTIONS",
    "POST_SETUP_AGENT_ORDER",
]

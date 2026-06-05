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
STATE_MAP: dict[str, str] = {
    "done": "done",
    "skipped": "skipped",
    "failed": "failed",
    "planned": "pending",
    "not_implemented": "warn",
}

# Synthetic per-row dwell so the (already-complete) run reads as live.
_MIN_DWELL_S = 0.18


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
    config_home: Path | None = None,
    pot_name: str | None = None,
) -> None:
    """Replay a completed ``SetupReport`` through the animated checklist."""
    wizard = SetupWizardUI(use_rich=use_rich)
    for step in report.steps:
        running, done = STEP_LABELS.get(step.step, (step.step, step.step))
        if step.step == "skills":
            running = f"Installing agent skills ({agent})…"
        wizard.add_step(
            step.step, running, chomp=step.step in CHOMP_STEPS, done_label=done
        )

    wizard.run_intro(repo=repo, agent=agent, scan=scan)

    failed_step_id: str | None = None
    with wizard.live():
        for step in report.steps:
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

    setup_path = str(config_home / "config.json") if config_home else ""
    data_path = str(config_home) if config_home else ""
    wizard.print_complete_summary(
        setup_path=setup_path,
        data_path=data_path,
        pot_name=pot_name or report.plan.pot,
        already_setup=False,
    )


def maybe_prompt_github_login() -> None:
    """After a successful setup, offer to connect GitHub via device flow."""
    if not is_interactive_tty():
        return
    import typer

    from adapters.inbound.cli.auth.github_commands import github_login_impl

    try:
        confirmed = typer.confirm(
            "\nWould you like to log in to GitHub now?", default=True
        )
    except typer.Abort:
        confirmed = False
    if confirmed:
        github_login_impl()


__all__ = [
    "STEP_LABELS",
    "STATE_MAP",
    "CHOMP_STEPS",
    "rich_enabled",
    "render_setup_report",
    "maybe_prompt_github_login",
]

"""Bootstrap + profile commands: ``setup`` / ``status`` / ``doctor`` / ``config``.

``setup`` runs the documented idempotent first-run sequence against the host
services (proving the journey shape). ``status`` is the cheap aggregate composed
from all three services via ``context_status``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from adapters.inbound.cli.commands._common import (
    EXIT_DEGRADED,
    contract,
    emit,
    get_host,
    is_json,
    resolve_pot_id,
)
from adapters.inbound.cli.telemetry.onboarding_events import (
    CliSetupAnalyticsObserver,
    begin_setup_run,
    capture_activation_succeeded,
    capture_project_binding_event,
    capture_setup_completed,
    capture_setup_dry_run_completed,
    capture_setup_started,
    elapsed_ms,
    now_ms,
)
from adapters.inbound.cli.ui import setup_ux
from bootstrap import sentry_metrics_runtime
from domain.errors import CapabilityNotImplemented
from domain.lifecycle import SetupPlan, SetupReport
from domain.ports.agent_context import StatusRequest


def register(root: typer.Typer) -> None:
    @root.command()
    def setup(
        repo: str = typer.Option(".", "--repo"),
        pot: str = typer.Option("foo-pot", "--pot"),
        agent: str = typer.Option("claude", "--agent"),
        backend: str = typer.Option(
            None,
            "--backend",
            help="Graph backend profile (defaults to the active backend).",
        ),
        scan: bool = typer.Option(False, "--scan"),
        dry_run: bool = typer.Option(
            False, "--dry-run", help="Show the steps without executing."
        ),
        yes: bool = typer.Option(False, "--yes", "-y", help="Assume yes for prompts."),
    ) -> None:
        """Idempotent first-run: provision config, storage, daemon, default pot, skills."""
        with contract():
            host = get_host()
            # --backend selects the storage profile for this run. Backend
            # selection happens at wiring time, so rebuild the host on the chosen
            # profile when it differs from the active one (keeps the report honest).
            if backend and backend != host.backend.profile:
                from adapters.inbound.cli.commands._common import set_host
                from adapters.outbound.graph.backends import build_backend
                from bootstrap.host_wiring import build_host_shell

                host = build_host_shell(
                    backend=build_backend(backend), profile=host.profile
                )
                set_host(host)
            json_output = is_json()
            use_rich = setup_ux.rich_enabled(as_json=json_output) and not yes
            plan = SetupPlan(
                mode=host.profile if host.profile in ("local", "managed") else "local",
                host_mode="in_process"
                if getattr(host.daemon, "in_process", False)
                else "daemon",
                backend=host.backend.profile,
                repo=repo,
                pot=pot,
                agent=agent,
                scan=scan,
                assume_yes=yes,
                defer_default_pot=use_rich,
                defer_skills=use_rich,
            )
            setup_started_ms = now_ms()
            begin_setup_run()
            host.setup.set_observer(CliSetupAnalyticsObserver())
            capture_setup_started(
                plan,
                interactive=use_rich,
                json_output=json_output,
            )

            if dry_run:
                preview = host.setup.preview(plan)
                capture_setup_dry_run_completed(
                    plan=plan,
                    planned_step_count=len(preview.steps),
                    hard_step_count=sum(1 for step in preview.steps if step.hard),
                )
                emit(preview.to_dict(), human=_preview_human(preview))
                _emit_setup_run_metric(plan, result="dry_run", dry_run=True)
                return

            report = host.setup.run(plan)
            capture_setup_completed(
                plan=plan,
                ok=report.ok,
                duration_ms=elapsed_ms(setup_started_ms),
                hard_failed_step=_first_hard_failed_step(report),
                soft_warning_count=_soft_warning_count(report),
            )
            if report.ok and not use_rich:
                _capture_plain_project_binding(report)
            _emit_setup_run_metric(
                report.plan,
                result="ok" if report.ok else "degraded",
                dry_run=False,
            )
            _emit_setup_step_metrics(report)

            # Animated wizard for interactive TTYs; --json and --yes/non-interactive
            # fall through to the plain, machine-readable emit path.
            if use_rich:
                setup_ux.render_setup_report(
                    report,
                    repo=Path(repo),
                    agent=agent,
                    scan=scan,
                    use_rich=True,
                )
                if report.ok:
                    setup_ux.maybe_prompt_github_login(
                        repo=Path(repo),
                        setup_agent=agent,
                        default_pot_name=pot,
                    )
            else:
                emit(report.to_dict(), human=_setup_human(report))

            if not report.ok:
                raise typer.Exit(code=EXIT_DEGRADED)

    @root.command()
    def status(
        verify: bool = typer.Option(
            False,
            "--verify",
            help="Verify integration credentials with a lightweight API check.",
        ),
        host: bool = typer.Option(
            False,
            "--host",
            help="Show host/pot readiness instead of integration auth status.",
        ),
        intent: str = typer.Option(
            "feature",
            "--intent",
            help="Intent for host status (use with --host or non-default harness/pot).",
        ),
        harness: str = typer.Option(
            "claude",
            "--harness",
            help="Harness for host status (use with --host or non-default intent/pot).",
        ),
        pot: str = typer.Option(
            None,
            "--pot",
            help="Pot for host status (use with --host or non-default intent/harness).",
        ),
    ) -> None:
        """Integration auth status by default; use --host for daemon/pot readiness."""
        host_status = (
            host or pot is not None or intent != "feature" or harness != "claude"
        )
        if not host_status:
            from adapters.inbound.cli.auth.auth_commands import integration_status

            integration_status(verify=verify)
            return

        with contract():
            shell = get_host()
            pot_id = resolve_pot_id(shell, pot)
            report = shell.agent_context.status(
                StatusRequest(pot_id=pot_id, intent=intent, harness=harness)
            )
            _capture_host_status_activation()
            emit(
                {
                    "profile": report.profile,
                    "daemon_up": report.daemon_up,
                    "active_pot": report.active_pot,
                    "backend_ready": report.backend_ready,
                    "data_plane": dict(report.data_plane),
                    "pot_summary": dict(report.pot_summary),
                    "skills": _nudge_dict(report.skills),
                    "recommended_next_action": report.recommended_next_action,
                },
                human=_status_human(report),
            )

    @root.command()
    def doctor() -> None:
        """Local diagnostics: daemon, backend capabilities, skill drift."""
        with contract():
            host = get_host()
            caps = host.backend.capabilities()
            emit(
                {
                    "daemon": host.daemon.status(),
                    "backend_profile": host.backend.profile,
                    "backend_capabilities": list(caps.implemented()),
                    "ledger": {
                        "available": host.ledger.status().available,
                        "binding": host.ledger.status().binding,
                    },
                },
                human=(
                    f"daemon: {host.daemon.status()['mode']} (up)\n"
                    f"backend: {host.backend.profile} caps={', '.join(caps.implemented())}\n"
                    f"ledger: {host.ledger.status().binding} "
                    f"available={host.ledger.status().available}"
                ),
            )

    @root.command()
    def whoami() -> None:
        """Show the current host identity (local OSS reports a 'none' identity)."""
        with contract():
            ident = get_host().auth.whoami()
            emit(
                {"subject": ident.subject, "mode": ident.mode, "detail": ident.detail},
                human=f"{ident.subject} (mode={ident.mode})"
                + (f" — {ident.detail}" if ident.detail else ""),
            )

    # NOTE: top-level `login` / `logout` are the real Potpie-account flows,
    # registered in commands/auth.py. Managed-backend auth remains `cloud login`.

    @root.command()
    def use(
        ref: str,
        local: bool = typer.Option(False, "--local", help="Force local-origin pot."),
        managed: bool = typer.Option(
            False, "--managed", help="Select a managed-origin pot."
        ),
    ) -> None:
        """Select the active pot by name/id (top-level alias for `pot use`)."""
        with contract():
            if managed:
                raise CapabilityNotImplemented(
                    "host.pots.use_managed",
                    detail="managed pot routing is not implemented",
                    recommended_next_action="select a local pot; managed routing lands in HU3",
                )
            pot = get_host().pots.use_pot(ref=ref)
            emit(
                {"id": pot.pot_id, "name": pot.name, "origin": "local"},
                human=f"active pot → {pot.name} (local)",
            )

    config_app = typer.Typer(
        help="Local config get/set (persisted to <home>/config.json)."
    )

    @config_app.command("get")
    def config_get(key: str) -> None:
        with contract():
            value = get_host().config.get(key)
            emit({key: value}, human=f"{key}={value}")

    @config_app.command("set")
    def config_set(key: str, value: str) -> None:
        with contract():
            get_host().config.set(key, value)
            emit(
                {"key": key, "value": value, "persisted": True},
                human=f"set {key}={value}",
            )

    root.add_typer(config_app, name="config")


def _nudge_dict(nudge) -> dict[str, object] | None:
    if nudge is None:
        return None
    return {
        "agent": nudge.agent,
        "missing": list(nudge.missing),
        "outdated": list(nudge.outdated),
        "install_command": nudge.install_command,
    }


def _step_line(step) -> str:
    line = f"  - {step.step}: {step.state}"
    return f"{line} — {step.detail}" if step.detail else line


def _preview_human(preview) -> str:
    lines = [
        f"dry-run: {len(preview.steps)} steps "
        f"(mode={preview.plan.mode}, host_mode={preview.plan.host_mode}, "
        f"backend={preview.plan.backend}):",
    ]
    for s in preview.steps:
        tag = "hard" if s.hard else "soft"
        line = f"  - {s.step} [{tag}] ({s.owner}): {s.action}"
        if s.skip_reason:
            line += f" — skip: {s.skip_reason}"
        lines.append(line)
    lines.append("  (no changes made; run without --dry-run to execute)")
    return "\n".join(lines)


def _setup_human(report) -> str:
    header = "setup complete" if report.ok else "setup incomplete (hard step missing)"
    lines = [f"{header} (mode={report.plan.mode}, backend={report.plan.backend}):"]
    lines.extend(_step_line(s) for s in report.steps)
    lines.append("  next: potpie status")
    return "\n".join(lines)


def _status_human(report) -> str:
    lines = [
        f"profile={report.profile} daemon={'up' if report.daemon_up else 'down'} "
        f"pot={report.active_pot} backend_ready={report.backend_ready}",
    ]
    counts = dict(report.data_plane).get("counts") or {}
    if counts:
        lines.append(f"  graph: {counts}")
    if report.skills and (report.skills.missing or report.skills.outdated):
        lines.append(
            f"  skills: missing={list(report.skills.missing)} → {report.skills.install_command}"
        )
    if report.recommended_next_action:
        lines.append(f"  next: {report.recommended_next_action}")
    return "\n".join(lines)


def _emit_setup_run_metric(plan: SetupPlan, *, result: str, dry_run: bool) -> None:
    sentry_metrics_runtime.count(
        "ce.setup.runs_total",
        attributes={
            "result": result,
            "backend": plan.backend,
            "host_mode": plan.host_mode,
            "scan": plan.scan,
            "dry_run": dry_run,
        },
    )


def _emit_setup_step_metrics(report: SetupReport) -> None:
    for step in report.steps:
        sentry_metrics_runtime.count(
            "ce.setup.step_total",
            attributes={
                "step": step.step,
                "state": step.state,
                "hard": step.hard,
            },
        )


__all__ = ["register"]


def _first_hard_failed_step(report) -> str | None:
    for step in report.steps:
        if step.hard and not step.ok:
            return step.step
    return None


def _soft_warning_count(report) -> int:
    return sum(1 for step in report.steps if not step.hard and not step.ok)


def _capture_plain_project_binding(report) -> None:
    source = _step_state(report, "source")
    skills = _step_state(report, "skills")
    if source is None and skills is None:
        return
    capture_project_binding_event(
        "cli_onboarding_project_binding_started",
        entrypoint="setup",
        properties={
            "repo_provided": report.plan.repo is not None,
            "agent": report.plan.agent,
        },
    )
    completed = source in {"done", "skipped"} and skills in {"done", "skipped"}
    capture_project_binding_event(
        "cli_onboarding_project_binding_completed"
        if completed
        else "cli_onboarding_project_binding_incomplete",
        entrypoint="setup",
        properties={
            "source_state": source or "missing",
            "skills_state": skills or "missing",
        },
    )


def _step_state(report, step_id: str) -> str | None:
    for step in report.steps:
        if step.step == step_id:
            return step.state
    return None


def _capture_host_status_activation() -> None:
    capture_activation_succeeded(
        command="status --host",
        result_kind="status_result",
    )

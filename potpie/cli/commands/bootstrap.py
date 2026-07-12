"""Bootstrap + profile commands: ``setup`` / ``status`` / ``doctor`` / ``config``.

``setup`` runs the documented idempotent first-run sequence against the host
services (proving the journey shape). ``status`` is the cheap aggregate composed
from all three services via ``context_status``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import typer
from potpie.config import (
    KNOWN_CONFIG_KEYS,
    public_config_value,
)
from potpie.install.status import (
    cli_install_human,
)
from potpie.cli.commands._common import (
    EXIT_DEGRADED,
    EXIT_VALIDATION,
    contract,
    emit,
    fail,
    get_cli_runtime,
    is_json,
)
from potpie.cli.telemetry.onboarding_events import (
    CliSetupAnalyticsObserver,
    begin_setup_run,
    capture_activation_succeeded,
    capture_setup_completed,
    capture_setup_dry_run_completed,
    capture_setup_started,
    elapsed_ms,
    now_ms,
)
from potpie.cli.ui import setup_ux
from potpie.runtime.telemetry import sentry_metrics
from potpie.setup import SetupPlan, SetupReport


def register(root: typer.Typer) -> None:
    @root.command()
    def setup(
        repo: str = typer.Option(".", "--repo"),
        pot: str = typer.Option("default", "--pot"),
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
        daemon: bool = typer.Option(
            None,
            "--daemon/--in-process",
            help=(
                "Provision a real detached daemon. Defaults to "
                "$CONTEXT_ENGINE_HOST_MODE or daemon."
            ),
        ),
        embeddings: str = typer.Option(
            None,
            "--embeddings",
            help=(
                "Embedding mode for local semantic search "
                "(sentence-transformers, auto, local, none)."
            ),
        ),
        embedding_model: str = typer.Option(
            None,
            "--embedding-model",
            help="SentenceTransformer model to prepare during setup.",
        ),
    ) -> None:
        """Idempotent first-run: provision config, storage, daemon, default pot, skills."""
        with contract():
            _run_product_setup_command(
                repo=repo,
                pot=pot,
                agent=agent,
                backend=backend,
                scan=scan,
                dry_run=dry_run,
                yes=yes,
                daemon=daemon,
                embeddings=embeddings,
                embedding_model=embedding_model,
            )
            return

    @root.command()
    def status(
        verify: bool = typer.Option(
            False,
            "--verify",
            help="Moved to `potpie integration status --verify`.",
        ),
        host: bool = typer.Option(
            False,
            "--host",
            help="Deprecated no-op; status reports host/pot readiness by default.",
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
        """context_status — host, pot, backend, and skill readiness."""
        _ = host  # Backward-compatible flag; readiness is now the default.
        if verify:
            fail(
                code="validation_error",
                message="`--verify` moved to `potpie integration status --verify`.",
                next_action=("potpie integration status --verify"),
                exit_code=EXIT_VALIDATION,
            )

        with contract():
            runtime = get_cli_runtime()
            report = runtime.status.get(pot_id=pot, harness=harness)
            _capture_host_status_activation()
            emit(report.to_dict(), human=_product_status_human(report))
            return

    @root.command()
    def doctor() -> None:
        """Local diagnostics: daemon, backend capabilities, skill drift."""
        with contract():
            runtime = get_cli_runtime()
            report = runtime.status.doctor()
            emit(report, human=_product_doctor_human(report))
            return

    @root.command()
    def whoami() -> None:
        """Show the current Potpie account identity."""
        with contract():
            ident = get_cli_runtime().auth.whoami()
            emit(
                {
                    "subject": ident.subject,
                    "authenticated": ident.authenticated,
                    "auth_type": ident.auth_type,
                },
                human=f"{ident.subject} (auth={ident.auth_type})",
            )

    config_app = typer.Typer(
        help=(
            "Local config get/set/list (persisted to <home>/config.json). "
            f"Known keys: {', '.join(KNOWN_CONFIG_KEYS)}."
        )
    )

    def _emit_config_list() -> None:
        config = get_cli_runtime().config.list_public()
        payload = {
            "config": config,
            "known_keys": list(KNOWN_CONFIG_KEYS),
        }
        if not config:
            human = "config: (empty)"
        else:
            lines = [f"{key}={value}" for key, value in config.items()]
            human = "\n".join(lines)
        emit(payload, human=human)

    @config_app.command("list")
    def config_list() -> None:
        """List all non-secret config entries."""
        with contract():
            _emit_config_list()

    @config_app.command("get")
    def config_get(
        key: str | None = typer.Argument(
            None,
            help=(
                "Config key to read. Omit to list all non-secret entries "
                "(same as `potpie config list`)."
            ),
        ),
    ) -> None:
        with contract():
            if key is None:
                _emit_config_list()
                return
            value = get_cli_runtime().config.get(key)
            value = public_config_value(key, value)
            emit({key: value}, human=f"{key}={value}")

    @config_app.command("set")
    def config_set(key: str, value: str) -> None:
        with contract():
            get_cli_runtime().config.set(key, value)
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


def _run_product_setup_command(
    *,
    repo: str,
    pot: str,
    agent: str,
    backend: str | None,
    scan: bool,
    dry_run: bool,
    yes: bool,
    daemon: bool | None,
    embeddings: str | None,
    embedding_model: str | None,
) -> None:
    runtime = get_cli_runtime()
    json_output = is_json()
    human_output = not json_output
    interactive_onboarding = (
        human_output
        and setup_ux.interactive_onboarding_enabled(as_json=json_output)
        and not yes
    )
    use_live = human_output and setup_ux.rich_enabled(as_json=json_output) and not yes
    stream_plain_progress = human_output and not use_live
    selected_embeddings = _setup_embeddings_choice(embeddings)
    selected_embedding_model = _setup_embedding_model(embedding_model)
    selected_mode = (
        "daemon"
        if daemon is True
        else "in-process"
        if daemon is False
        else runtime.settings.runtime_mode
    )
    selected_backend = backend or runtime.settings.backend
    if (
        selected_mode != runtime.settings.runtime_mode
        or selected_backend != runtime.settings.backend
    ):
        from potpie.runtime.composition import create_runtime

        runtime = create_runtime(
            settings=replace(
                runtime.settings,
                runtime_mode=selected_mode,
                backend=selected_backend,
            )
        )
    plan = SetupPlan(
        host_mode=selected_mode,
        backend=selected_backend,
        repo=repo,
        pot=pot,
        agent=agent,
        scan=scan,
        assume_yes=yes,
        defer_default_pot=interactive_onboarding,
        defer_skills=interactive_onboarding,
        embeddings=selected_embeddings,
        embedding_model=selected_embedding_model,
    )
    setup_started_ms = now_ms()
    begin_setup_run()
    capture_setup_started(
        plan,
        interactive=interactive_onboarding,
        json_output=json_output,
    )
    runtime.setup.set_observer(CliSetupAnalyticsObserver())

    if dry_run:
        preview = runtime.setup.preview(plan)
        capture_setup_dry_run_completed(
            plan=plan,
            planned_step_count=len(preview.steps),
            hard_step_count=sum(1 for step in preview.steps if step.hard),
        )
        emit(preview.to_dict(), human=_preview_human(preview))
        _emit_setup_run_metric(plan, result="dry_run", dry_run=True)
        return

    if use_live:
        report = setup_ux.run_setup_live(
            runtime.setup,
            plan,
            repo=Path(repo),
            agent=agent,
            scan=scan,
            use_rich=True,
            config_home=runtime.settings.data_dir,
            observer=CliSetupAnalyticsObserver(),
        )
    elif stream_plain_progress:
        report = setup_ux.run_setup_plain(
            runtime.setup,
            plan,
            repo=Path(repo),
            agent=agent,
            scan=scan,
            observer=CliSetupAnalyticsObserver(),
        )
    else:
        report = runtime.setup.run(plan)

    capture_setup_completed(
        plan=plan,
        ok=report.ok,
        duration_ms=elapsed_ms(setup_started_ms),
        hard_failed_step=_first_hard_failed_step(report),
        soft_warning_count=_soft_warning_count(report),
    )
    _emit_setup_run_metric(
        report.plan,
        result="ok" if report.ok else "degraded",
        dry_run=False,
    )
    _emit_setup_step_metrics(report)
    if not use_live:
        emit(
            report.to_dict(),
            human=_setup_human(report, include_steps=not stream_plain_progress),
        )
    if interactive_onboarding and report.ok:
        setup_ux.maybe_prompt_github_login(
            repo=Path(repo), setup_agent=agent, default_pot_name=pot
        )
    if not report.ok:
        raise typer.Exit(code=EXIT_DEGRADED)


def _product_status_human(report) -> str:
    lines = [
        f"ready={report.ready} runtime={report.runtime_mode} daemon={report.daemon_state}",
        f"pot={report.pot_name or report.pot_id or '(none)'} backend={report.backend} ready={report.backend_ready}",
        f"storage={report.storage_ready} ingestion={report.ingestion_ready} sources={report.source_count}",
        f"skills={report.skills_state} setup={report.setup_state}",
    ]
    lines.extend(f"issue: {issue}" for issue in report.issues)
    if report.recommended_next_action:
        lines.append(f"next: {report.recommended_next_action.get('command')}")
    return "\n".join(lines)


def _product_doctor_human(report: dict[str, object]) -> str:
    lines = [
        f"ready={report.get('ready')} runtime={report.get('runtime_mode')} daemon={report.get('daemon_state')}",
        f"backend={report.get('backend')} ready={report.get('backend_ready')}",
        cli_install_human(report.get("cli_install", {})),
    ]
    issues = report.get("issues") or ()
    lines.extend(f"issue: {issue}" for issue in issues)
    return "\n".join(lines)


def _setup_human(report, *, include_steps: bool = True) -> str:
    header = "setup complete" if report.ok else "setup incomplete (hard step missing)"
    lines = [f"{header} (mode={report.plan.mode}, backend={report.plan.backend}):"]
    if include_steps:
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
    sentry_metrics.count(
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
        sentry_metrics.count(
            "ce.setup.step_total",
            attributes={
                "step": step.step,
                "state": step.state,
                "hard": step.hard,
            },
        )


def _setup_embeddings_choice(raw: str | None) -> str:
    choice = (raw or "sentence-transformers").strip().lower().replace("_", "-")
    aliases = {
        "legacy": "sentence-transformers",
        "sbert": "sentence-transformers",
        "minilm": "sentence-transformers",
        "all-minilm-l6-v2": "sentence-transformers",
        "hashing": "local",
        "default": "local",
        "off": "none",
        "disabled": "none",
        "lexical": "none",
    }
    return aliases.get(choice, choice)


def _setup_embedding_model(raw: str | None) -> str:
    if raw is not None and raw.strip():
        return raw.strip()
    return "all-MiniLM-L6-v2"


__all__ = ["register"]


def _first_hard_failed_step(report) -> str | None:
    for step in report.steps:
        if step.hard and not step.ok:
            return step.step
    return None


def _soft_warning_count(report) -> int:
    return sum(1 for step in report.steps if not step.hard and not step.ok)


def _capture_host_status_activation() -> None:
    capture_activation_succeeded(
        command="status",
        result_kind="status_result",
    )

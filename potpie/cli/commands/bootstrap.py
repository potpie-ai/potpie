"""Bootstrap + profile commands: ``setup`` / ``status`` / ``doctor`` / ``config``.

``setup`` runs the documented idempotent first-run sequence against the host
services (proving the journey shape). ``status`` is the cheap aggregate composed
from all three services via ``context_status``.
"""

from __future__ import annotations

import os
from pathlib import Path

import typer

from potpie.cli.cli_install_status import (
    cli_install_human,
    collect_cli_install_status,
)
from potpie.cli.commands._common import (
    EXIT_DEGRADED,
    EXIT_VALIDATION,
    contract,
    current_repo_identity_for_cli,
    emit,
    fail,
    get_host,
    is_json,
    repo_default_pot_id,
    repo_effective_pot_info,
    resolve_pot_id,
    use_pot_selection,
)
from potpie.cli.telemetry.onboarding_events import (
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
from potpie.cli.ui import setup_ux
from potpie_context_engine.adapters.outbound.intelligence.local_embedder import (
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    configured_embedder_choice,
    configured_embedding_model,
)
from potpie.services.config_service import KNOWN_CONFIG_KEYS, public_config_value
from potpie_context_engine.bootstrap import sentry_metrics_runtime
from potpie_context_engine.domain.embedding_modes import normalize_embedding_mode
from potpie_context_engine.domain.errors import CapabilityNotImplemented
from potpie_context_engine.domain.lifecycle import SetupPlan, SetupReport
from potpie_context_engine.domain.ports.agent_context import StatusRequest


def _effective_current_repo_pot_id(
    host, *, repo_identity: str | None, active_pot_id: str | None
) -> str | None:
    """Mirror CLI repo-pot resolution without raising structured command errors."""
    if not repo_identity:
        return None

    routing = repo_effective_pot_info(host)
    effective = routing.get("effective_pot") or {}
    effective_id = effective.get("id")
    if effective_id:
        return str(effective_id)
    if routing.get("status") == "ambiguous":
        candidate_ids = {
            str(row.get("id")) for row in routing.get("candidates", ()) if row.get("id")
        }
        return active_pot_id if active_pot_id in candidate_ids else None
    return active_pot_id


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
            json_output = is_json()
            human_output = not json_output
            interactive_onboarding = (
                human_output
                and setup_ux.interactive_onboarding_enabled(as_json=json_output)
                and not yes
            )
            use_live = (
                human_output and setup_ux.rich_enabled(as_json=json_output) and not yes
            )
            stream_plain_progress = human_output and not use_live
            selected_embeddings = _setup_embeddings_choice(embeddings)
            selected_embedding_model = _setup_embedding_model(embedding_model)
            _apply_setup_embedding_env(
                embeddings=selected_embeddings,
                embedding_model=selected_embedding_model,
                explicit_embeddings=embeddings is not None,
                explicit_model=embedding_model is not None,
            )
            from potpie.services.host_wiring import default_backend_profile

            if human_output:
                host, selected_backend, in_process = _build_local_setup_host(
                    backend=backend,
                    daemon=daemon,
                    default_backend=default_backend_profile(),
                )
            else:
                host = get_host()
                in_process = getattr(host.daemon, "in_process", False)
                selected_backend = backend or (
                    getattr(host.backend, "profile", default_backend_profile())
                    if in_process
                    else default_backend_profile()
                )
            # --backend selects the storage profile for this run. Backend
            # selection happens at wiring time, so rebuild the host on the chosen
            # profile when it differs from the active one (keeps the report honest).
            if (
                not use_live
                and in_process
                and backend
                and backend != host.backend.profile
            ):
                from potpie.cli.commands._common import set_host
                from potpie_context_engine.adapters.outbound.graph.backends import build_backend
                from potpie.services.host_wiring import build_host_shell

                host = build_host_shell(
                    backend=build_backend(backend), profile=host.profile
                )
                set_host(host)
                in_process = getattr(host.daemon, "in_process", False)
                selected_backend = host.backend.profile
            if (
                not use_live
                and daemon is not None
                and host.daemon.in_process != (not daemon)
            ):
                import os

                from potpie.cli.commands._common import set_host
                from potpie_context_engine.adapters.outbound.graph.backends import build_backend
                from potpie.services.host_wiring import build_host_shell

                os.environ["CONTEXT_ENGINE_HOST_MODE"] = (
                    "daemon" if daemon else "in_process"
                )
                host = build_host_shell(
                    backend=build_backend(selected_backend), profile=host.profile
                )
                set_host(host)
                in_process = getattr(host.daemon, "in_process", False)
                selected_backend = host.backend.profile
            plan = SetupPlan(
                mode=host.profile if host.profile in ("local", "managed") else "local",
                host_mode="in_process" if in_process else "daemon",
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
            if in_process:
                host.setup.set_observer(CliSetupAnalyticsObserver())
            capture_setup_started(
                plan,
                interactive=interactive_onboarding,
                json_output=json_output,
            )

            if dry_run:
                if in_process or host.daemon.status().get("up"):
                    preview = host.setup.preview(plan)
                else:
                    from potpie.services.host_wiring import build_host_shell

                    preview_host = build_host_shell()
                    preview = preview_host.setup.preview(plan)
                capture_setup_dry_run_completed(
                    plan=plan,
                    planned_step_count=len(preview.steps),
                    hard_step_count=sum(1 for step in preview.steps if step.hard),
                )
                emit(preview.to_dict(), human=_preview_human(preview))
                _emit_setup_run_metric(plan, result="dry_run", dry_run=True)
                return

            if not in_process and not human_output:
                host.daemon.ensure(plan)
                daemon_status = host.daemon.status()
                running_backend = daemon_status.get("backend")
                if backend:
                    _raise_if_backend_mismatch(running_backend, backend)

            if not in_process and human_output:
                _validate_existing_daemon_backend(host, requested_backend=backend)

            if use_live:
                report = setup_ux.run_setup_live(
                    host.setup,
                    plan,
                    repo=Path(repo),
                    agent=agent,
                    scan=scan,
                    use_rich=True,
                    config_home=getattr(host.daemon, "home", None),
                    observer=CliSetupAnalyticsObserver(),
                )
            elif stream_plain_progress:
                report = setup_ux.run_setup_plain(
                    host.setup,
                    plan,
                    repo=Path(repo),
                    agent=agent,
                    scan=scan,
                    observer=CliSetupAnalyticsObserver(),
                )
            else:
                report = host.setup.run(plan)
            capture_setup_completed(
                plan=plan,
                ok=report.ok,
                duration_ms=elapsed_ms(setup_started_ms),
                hard_failed_step=_first_hard_failed_step(report),
                soft_warning_count=_soft_warning_count(report),
            )
            if report.ok and not interactive_onboarding:
                _capture_plain_project_binding(report)
            _emit_setup_run_metric(
                report.plan,
                result="ok" if report.ok else "degraded",
                dry_run=False,
            )
            _emit_setup_step_metrics(report)

            # Setup progress streams live or line-by-line for humans; --json remains
            # machine-readable. Onboarding prompts are independent of live rendering.
            if not use_live:
                emit(
                    report.to_dict(),
                    human=_setup_human(
                        report,
                        include_steps=not stream_plain_progress,
                    ),
                )
            if interactive_onboarding and report.ok:
                setup_ux.maybe_prompt_github_login(
                    repo=Path(repo),
                    setup_agent=agent,
                    default_pot_name=pot,
                )

            if not report.ok:
                raise typer.Exit(code=EXIT_DEGRADED)

    @root.command()
    def status(
        verify: bool = typer.Option(
            False,
            "--verify",
            help="Moved to `potpie auth status --verify`.",
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
                message="`--verify` moved to `potpie auth status --verify`.",
                next_action=(
                    "Run `potpie auth status --verify` for integration auth status, "
                    "or `potpie status` for context readiness."
                ),
                exit_code=EXIT_VALIDATION,
            )

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
            pot = host.pots.active_pot()
            pot_id = getattr(pot, "pot_id", "") if pot is not None else ""
            readiness = host.backend.mutation.readiness(pot_id)
            daemon_status = host.daemon.status()

            repo_identity = current_repo_identity_for_cli()
            effective_current_repo_pot = _effective_current_repo_pot_id(
                host,
                repo_identity=repo_identity,
                active_pot_id=pot_id or None,
            )
            default_pot_id = repo_default_pot_id(host, repo_identity)

            cli_install = collect_cli_install_status()
            emit(
                {
                    "daemon": daemon_status,
                    "cli_install": cli_install,
                    "backend_profile": host.backend.profile,
                    "backend_ready": readiness.ready,
                    "backend_readiness": {
                        "profile": readiness.profile,
                        "ready": readiness.ready,
                        "capability_ready": dict(readiness.capability_ready),
                        "detail": readiness.detail,
                    },
                    "backend_capabilities": list(caps.implemented()),
                    "active_pot": pot_id or None,
                    "effective_current_repo_pot": effective_current_repo_pot,
                    "repo_default_pot": default_pot_id,
                    "recommended_next_action": None
                    if readiness.ready
                    else "Run `potpie backend doctor` or inspect `potpie graph status --json`.",
                    "ledger": {
                        "available": host.ledger.status().available,
                        "binding": host.ledger.status().binding,
                    },
                },
                human=(
                    f"daemon: {daemon_status['mode']} (up={daemon_status.get('up')})\n"
                    f"{cli_install_human(cli_install)}\n"
                    f"backend: {host.backend.profile} ready={readiness.ready} "
                    f"caps={', '.join(caps.implemented())}\n"
                    f"ledger: {host.ledger.status().binding} "
                    f"available={host.ledger.status().available}"
                    + (
                        f"\nrepo: {repo_identity} → {effective_current_repo_pot}"
                        + (
                            f" (default={default_pot_id})"
                            if default_pot_id
                            else " (no repo default set)"
                        )
                        if repo_identity
                        else ""
                    )
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
        also_default_for_current_repo: bool = typer.Option(
            False,
            "--also-default-for-current-repo",
            help="Also set the current repo's local default pot to this pot.",
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
            host = get_host()
            payload, human = use_pot_selection(
                host,
                ref,
                also_default_for_current_repo=also_default_for_current_repo,
                origin="local",
            )
            emit(payload, human=human)

    config_app = typer.Typer(
        help=(
            "Local config get/set/list (persisted to <home>/config.json). "
            f"Known keys: {', '.join(KNOWN_CONFIG_KEYS)}."
        )
    )

    def _emit_config_list() -> None:
        config = get_host().config.list_public()
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
            value = get_host().config.get(key)
            value = public_config_value(key, value)
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


def _setup_embeddings_choice(raw: str | None) -> str:
    if raw is not None:
        choice = normalize_embedding_mode(raw)
    else:
        configured = configured_embedder_choice()
        choice = normalize_embedding_mode(configured or "sentence-transformers")
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
    configured = configured_embedding_model()
    return configured or DEFAULT_SENTENCE_TRANSFORMER_MODEL


def _apply_setup_embedding_env(
    *,
    embeddings: str,
    embedding_model: str,
    explicit_embeddings: bool,
    explicit_model: bool,
) -> None:
    if explicit_embeddings:
        os.environ["CONTEXT_ENGINE_EMBEDDER"] = embeddings
    else:
        os.environ.setdefault("CONTEXT_ENGINE_EMBEDDER", embeddings)
    if explicit_model:
        os.environ["CONTEXT_ENGINE_EMBEDDING_MODEL"] = embedding_model
    else:
        os.environ.setdefault("CONTEXT_ENGINE_EMBEDDING_MODEL", embedding_model)


def _build_local_setup_host(
    *,
    backend: str | None,
    daemon: bool | None,
    default_backend: str,
):
    """Build a local setup host so the Rich wizard can observe real steps."""
    import os

    from potpie.cli.commands._common import set_host
    from potpie_context_engine.adapters.outbound.graph.backends import build_backend
    from potpie.services.host_wiring import build_host_shell

    selected_backend = backend or default_backend
    if daemon is not None:
        os.environ["CONTEXT_ENGINE_HOST_MODE"] = "daemon" if daemon else "in_process"
    host = build_host_shell(backend=build_backend(selected_backend))
    set_host(host)
    return host, host.backend.profile, getattr(host.daemon, "in_process", False)


def _validate_existing_daemon_backend(host, *, requested_backend: str | None) -> None:
    if not requested_backend:
        return
    daemon_status = host.daemon.status()
    if not daemon_status.get("up"):
        return
    running_backend = daemon_status.get("backend")
    _raise_if_backend_mismatch(running_backend, requested_backend)


def _raise_if_backend_mismatch(running_backend: object, requested_backend: str) -> None:
    if not isinstance(running_backend, str):
        raise ValueError(
            "daemon is running but its backend could not be verified; "
            "stop it with 'potpie daemon stop' before changing backend"
        )
    if running_backend != requested_backend:
        raise ValueError(
            "daemon is already running with backend "
            f"{running_backend!r}; stop it with 'potpie daemon stop' "
            f"before running setup with backend {requested_backend!r}"
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
        command="status",
        result_kind="status_result",
    )

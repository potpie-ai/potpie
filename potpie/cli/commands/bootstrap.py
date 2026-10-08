"""Bootstrap + profile commands: ``setup`` / ``status`` / ``doctor`` / ``config``.

``setup`` runs the documented idempotent first-run sequence against the host
services (proving the journey shape). ``status`` is the cheap aggregate composed
from all three services via ``context_status``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import click
import typer

from potpie.cli.cli_install_status import (
    cli_install_human,
    collect_cli_install_status,
)
from potpie.cli.commands._common import (
    EXIT_DEGRADED,
    EXIT_VALIDATION,
    activation_command_outcome,
    contract,
    current_repo_identity_for_cli,
    emit,
    fail,
    get_auth_service,
    get_config_service,
    get_daemon_service,
    get_engine_client,
    get_ledger_service,
    get_pot_service,
    get_root_runtime,
    get_setup_service,
    get_skill_service,
    is_json,
    repo_default_pot_id,
    repo_effective_pot_info,
    require_text,
    resolve_pot_id,
    run_engine_operation,
    use_pot_selection,
)
from potpie.cli.telemetry.onboarding_events import (
    CliSetupAnalyticsObserver,
    begin_setup_run,
    capture_project_binding_event,
    capture_setup_completed,
    capture_setup_dry_run_completed,
    capture_setup_incomplete,
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
from potpie.agent_context import (
    LOW_CONFIDENCE_THRESHOLD,
    QUALITY_SUMMARY_LIMIT,
    QualitySummaryUnavailable,
    quality_block,
    status_next_action,
)
from potpie.config.local import (
    GRAPH_PROTOCOLS_KEY,
    KNOWN_CONFIG_KEYS,
    SWITCH_VALUES,
    is_known_config_key,
    is_secret_config_key,
    normalize_switch,
    public_config_value,
)
from potpie_context_engine.bootstrap import sentry_metrics_runtime
from potpie_context_engine.domain.embedding_modes import normalize_embedding_mode
from potpie_context_engine.core.errors import CapabilityNotImplemented
from potpie_context_engine.core.lifecycle import SetupPlan, SetupReport
from potpie_context_engine.core.ports.agent_context import StatusReport
from potpie_context_engine.requests import DataPlaneStatusRequest, QualityRequest


def _embedded_graph_servers(profile: str) -> dict | None:
    """Report leaked embedded graph servers, for the profile that has them.

    Only ``falkordb_lite`` runs one. It is daemonized, so it outlives whatever
    started it; anything that died without stopping it leaves a ``redis-server``
    holding a db file and its memory until the machine reboots, and nothing
    else on the system will ever mention it.
    """
    if profile != "falkordb_lite":
        return None
    try:
        from potpie_context_engine.adapters.outbound.graph.falkordb_writer import (
            embedded_server_report,
        )
        from potpie_context_engine.adapters.outbound.settings_env import (
            EnvContextEngineSettings,
        )

        return embedded_server_report(EnvContextEngineSettings().falkordb_lite_path())
    except Exception:  # noqa: BLE001 - a diagnostic must never break doctor
        return None


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
                "(auto, sentence-transformers, local, none). Defaults to auto: "
                "sentence-transformers when the potpie[embeddings] extra is "
                "installed, otherwise the bundled hashing embedder."
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
            from potpie.runtime.composition import default_backend_profile

            if human_output:
                host, selected_backend, in_process = _build_local_setup_host(
                    backend=backend,
                    daemon=daemon,
                    default_backend=default_backend_profile(),
                )
            else:
                host = get_root_runtime()
                in_process = getattr(get_daemon_service(host), "in_process", False)
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
                from potpie.cli.commands._common import set_runtime
                from potpie_context_engine.adapters.outbound.graph.backends import (
                    build_backend,
                )
                from potpie.runtime.composition import build_local_runtime

                runtime = build_local_runtime(
                    backend=build_backend(backend), profile=host.profile
                )
                set_runtime(runtime)
                host = runtime.root
                in_process = getattr(get_daemon_service(host), "in_process", False)
                selected_backend = host.backend.profile
            if (
                not use_live
                and daemon is not None
                and get_daemon_service(host).in_process != (not daemon)
            ):
                import os

                from potpie.cli.commands._common import set_runtime
                from potpie_context_engine.adapters.outbound.graph.backends import (
                    build_backend,
                )
                from potpie.runtime.composition import build_local_runtime

                os.environ["CONTEXT_ENGINE_HOST_MODE"] = (
                    "daemon" if daemon else "in_process"
                )
                runtime = build_local_runtime(
                    backend=build_backend(selected_backend), profile=host.profile
                )
                set_runtime(runtime)
                host = runtime.root
                in_process = getattr(get_daemon_service(host), "in_process", False)
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
            setup_observer = CliSetupAnalyticsObserver()
            if in_process:
                get_setup_service(host).set_observer(setup_observer)
            capture_setup_started(
                plan,
                interactive=interactive_onboarding,
                json_output=json_output,
                dry_run=dry_run,
            )

            try:
                if dry_run:
                    if in_process or get_daemon_service(host).status().get("up"):
                        preview = get_setup_service(host).preview(plan)
                    else:
                        from potpie.runtime.composition import build_local_runtime

                        preview_runtime = build_local_runtime()
                        preview = get_setup_service(preview_runtime.root).preview(plan)
                    capture_setup_dry_run_completed(
                        plan=plan,
                        planned_step_count=len(preview.steps),
                        hard_step_count=sum(1 for step in preview.steps if step.hard),
                    )
                    emit(preview.to_dict(), human=_preview_human(preview))
                    _emit_setup_run_metric(plan, result="dry_run", dry_run=True)
                    return

                if not in_process and not human_output:
                    get_daemon_service(host).ensure(plan)
                    daemon_status = get_daemon_service(host).status()
                    running_backend = daemon_status.get("backend")
                    if backend:
                        _raise_if_backend_mismatch(running_backend, backend)

                if not in_process and human_output:
                    _validate_existing_daemon_backend(host, requested_backend=backend)

                if use_live:
                    report = setup_ux.run_setup_live(
                        get_setup_service(host),
                        plan,
                        repo=Path(repo),
                        agent=agent,
                        scan=scan,
                        use_rich=True,
                        config_home=getattr(get_daemon_service(host), "home", None),
                        observer=setup_observer,
                    )
                elif stream_plain_progress:
                    report = setup_ux.run_setup_plain(
                        get_setup_service(host),
                        plan,
                        repo=Path(repo),
                        agent=agent,
                        scan=scan,
                        observer=setup_observer,
                    )
                else:
                    report = get_setup_service(host).run(plan)
            except (KeyboardInterrupt, EOFError, typer.Abort, click.Abort):
                capture_setup_incomplete(
                    plan=plan,
                    incomplete_kind="cancelled",
                    duration_ms=elapsed_ms(setup_started_ms),
                    failure_stage=setup_observer.current_or_last_step,
                    dry_run=dry_run,
                )
                raise
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
        with contract():
            with activation_command_outcome(
                command="status", result_kind="status_result"
            ):
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

                shell = get_root_runtime()
                pot_id = resolve_pot_id(shell, pot)
                client = get_engine_client(pot)
                data_plane = run_engine_operation(
                    client.data_plane_status(DataPlaneStatusRequest())
                )
                report = _build_context_status_report(
                    shell,
                    pot_id=pot_id,
                    intent=intent,
                    harness=harness,
                    data_plane=data_plane,
                    quality_summary=_status_quality_summary(client),
                )
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
            host = get_root_runtime()
            caps = host.backend.capabilities()
            pot = get_pot_service(host).active_pot()
            pot_id = getattr(pot, "pot_id", "") if pot is not None else ""
            readiness = host.backend.mutation.readiness(pot_id)
            daemon_status = get_daemon_service(host).status()

            repo_identity = current_repo_identity_for_cli()
            effective_current_repo_pot = _effective_current_repo_pot_id(
                host,
                repo_identity=repo_identity,
                active_pot_id=pot_id or None,
            )
            default_pot_id = repo_default_pot_id(host, repo_identity)

            cli_install = collect_cli_install_status()
            embedded_servers = _embedded_graph_servers(host.backend.profile)
            resources, resource_index = _resource_doctor_blocks(pot_id)
            emit(
                {
                    "resources": resources,
                    "resource_index": resource_index,
                    "daemon": daemon_status,
                    "embedded_graph_servers": embedded_servers,
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
                        "available": get_ledger_service(host).status().available,
                        "binding": get_ledger_service(host).status().binding,
                    },
                },
                human=(
                    f"daemon: {daemon_status['mode']} (up={daemon_status.get('up')})\n"
                    f"{cli_install_human(cli_install)}\n"
                    f"backend: {host.backend.profile} ready={readiness.ready} "
                    f"caps={', '.join(caps.implemented())}\n"
                    f"ledger: {get_ledger_service(host).status().binding} "
                    f"available={get_ledger_service(host).status().available}"
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
                    + (
                        f"\nembedded servers: {embedded_servers['detail']}"
                        if embedded_servers and embedded_servers.get("detail")
                        else ""
                    )
                    + "\n"
                    + _resource_doctor_human(resources, resource_index)
                ),
            )

    @root.command()
    def whoami() -> None:
        """Show the current host identity (local OSS reports a 'none' identity)."""
        with contract():
            ident = get_auth_service().whoami()
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
            host = get_root_runtime()
            payload, human = use_pot_selection(
                host,
                ref,
                also_default_for_current_repo=also_default_for_current_repo,
                origin="local",
            )
            emit(payload, human=human)

    config_app = typer.Typer(
        help=(
            "Local config get/set/unset/list (persisted to <home>/config.json). "
            f"Known keys: {', '.join(KNOWN_CONFIG_KEYS)}. "
            "`set` accepts only those; `unset` accepts any key, so a value "
            "stored before the catalog was enforced can still be removed."
        )
    )

    def _emit_config_list() -> None:
        config = get_config_service().list_public()
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
            # Distinct from the omitted argument above: `config get ''` reads a
            # key that cannot exist and must not answer like an unset one.
            key = require_text(key, argument="key", example="potpie config get backend")
            value = get_config_service().get(key)
            value = public_config_value(key, value)
            emit({key: value}, human=f"{key}={value}")

    @config_app.command("set")
    def config_set(key: str, value: str) -> None:
        """Persist one known config key. The write keeps the value; the echo does not.

        The catalog check turns a typo into a refusal instead of a persisted key
        nothing reads, and keeps ``config.json`` from becoming a secret store.
        The echo shares ``get``/``list`` redaction, including a credential typed
        inside a URL value.
        """
        with contract():
            key = require_text(
                key, argument="key", example="potpie config set backend embedded"
            )
            if not is_known_config_key(key):
                fail(
                    code="validation_error",
                    message=f"unknown config key {key!r}",
                    detail={"key": key, "known_keys": list(KNOWN_CONFIG_KEYS)},
                    # Names the exit too: a key stored under this name before
                    # the gate existed is read by nothing, and `unset` is the
                    # only command that can still clear it.
                    next_action=(
                        f"use one of: {', '.join(KNOWN_CONFIG_KEYS)} — "
                        "a key already stored under this name is read by nothing; "
                        f"remove it with 'potpie config unset {key}'"
                    ),
                    exit_code=EXIT_VALIDATION,
                )
            if key == "resource_index":
                value = _require_resource_index_profile(value)
            if key == GRAPH_PROTOCOLS_KEY:
                value = _require_switch(key, value)
            get_config_service().set(key, value)
            shown = public_config_value(key, value)
            payload: dict[str, object] = {
                "key": key,
                "value": shown,
                "redacted": is_secret_config_key(key) or shown != value,
                "persisted": True,
            }
            human = f"set {key}={shown}"
            if key in _STARTUP_CONFIG_KEYS:
                payload.update(_STARTUP_CONFIG_NOTE)
                human += f"\n{_STARTUP_CONFIG_NOTE['next_action']}"
            emit(payload, human=human)

    @config_app.command("unset")
    def config_unset(key: str) -> None:
        """Remove one config key. Accepts keys the catalog no longer knows.

        Ungated on purpose, where ``set`` is gated: the write gate strands every
        key this file used to accept, credentials among them, and removal is the
        only repair left. ``removed`` distinguishes "it is gone" from "there was
        nothing here"; both exit 0.
        """
        with contract():
            key = require_text(
                key, argument="key", example="potpie config unset github_token"
            )
            removed = get_config_service().unset(key)
            payload: dict[str, object] = {"key": key, "removed": removed}
            human = (
                f"unset {key}" if removed else f"{key} was not set (nothing removed)"
            )
            if removed and key in _STARTUP_CONFIG_KEYS:
                payload.update(_STARTUP_CONFIG_NOTE)
                human += f"\n{_STARTUP_CONFIG_NOTE['next_action']}"
            emit(payload, human=human)

    root.add_typer(config_app, name="config")


def _resource_doctor_blocks(pot_id: str) -> tuple[dict, dict]:
    """The document store and its index, as ``doctor`` rows that never raise.

    Beside each other, not merged: the bytes can be healthy while the index
    that makes them findable is off, stale, or mid-drain, and that gap is
    invisible until a search quietly returns less than it should. Both go
    through the typed engine boundary, so a daemon that is down or refuses the
    handshake becomes ``available: false`` with the reason instead of taking
    the rest of the report with it.
    """
    from potpie_context_engine.requests import (
        ResourceIndexStatusRequest,
        ResourceStatusRequest,
    )

    if not pot_id:
        gap = {"available": False, "detail": "no active pot; resources are per-pot"}
        return dict(gap), dict(gap)

    def probe(call, render) -> dict:
        try:
            return {"available": True, **render(run_engine_operation(call()))}
        except typer.Exit:
            raise
        except Exception as exc:  # noqa: BLE001 - a diagnostic row must not crash doctor
            return {"available": False, "detail": str(exc) or type(exc).__name__}

    client = None

    def engine():
        nonlocal client
        if client is None:
            client = get_engine_client(pot_id)
        return client

    resources = probe(
        lambda: engine().resource_status(ResourceStatusRequest()),
        lambda status: {
            "kind": status.kind,
            "ready": status.ready,
            "location": status.location,
            "documents": status.documents,
            "detail": status.detail,
        },
    )
    index = probe(
        lambda: engine().resource_index_status(ResourceIndexStatusRequest()),
        lambda status: {
            "profile": status.profile,
            "ready": status.ready,
            "capabilities": list(status.capabilities),
            "match_mode": status.match_mode,
            "documents": status.documents,
            "chunks": status.chunks,
            "pending_embeddings": status.pending_embeddings,
            "embedder": status.embedder,
            "detail": status.detail,
        },
    )
    return resources, index


def _resource_doctor_human(resources: dict, index: dict) -> str:
    if resources.get("available"):
        line = f"resources: {resources['kind']} ready={resources['ready']}"
        if resources.get("documents") is not None:
            line += f" documents={resources['documents']}"
        if resources.get("location"):
            line += f" ({resources['location']})"
    else:
        line = f"resources: unavailable — {resources.get('detail')}"
    if index.get("available"):
        index_line = (
            f"resource index: {index['profile']} ready={index['ready']} "
            f"mode={index['match_mode']} chunks={index['chunks']}"
        )
        if index.get("pending_embeddings"):
            index_line += f" pending={index['pending_embeddings']}"
        if index.get("detail"):
            index_line += f"\n  ! {index['detail']}"
    else:
        index_line = f"resource index: unavailable — {index.get('detail')}"
    return f"{line}\n{index_line}"


#: Keys the local runtime reads once, when it is composed. A daemon that is
#: already running keeps serving the value it started with.
_STARTUP_CONFIG_KEYS: frozenset[str] = frozenset({GRAPH_PROTOCOLS_KEY})
_STARTUP_CONFIG_NOTE: dict[str, object] = {
    "restart_required": True,
    "next_action": (
        "takes effect when the runtime next starts: "
        "run 'potpie daemon restart' if a daemon is running"
    ),
}


def _require_switch(key: str, value: str) -> str:
    """``on`` or ``off`` for an on/off key, refusing anything else.

    Refused here rather than read as off later: a typo would otherwise leave
    the feature silently disabled with ``config get`` showing the typo.
    """
    normalized = normalize_switch(value)
    if normalized is None:
        fail(
            code="validation_error",
            message=f"{key} must be one of: {', '.join(SWITCH_VALUES)} (got {value!r})",
            detail={"key": key, "values": list(SWITCH_VALUES)},
            next_action=f"potpie config set {key} on",
            exit_code=EXIT_VALIDATION,
        )
    return normalized


def _require_resource_index_profile(value: str) -> str:
    """A ``resource_index`` value the index registry will accept, normalized.

    Checked here rather than at the next command: an unknown profile only
    surfaces later as an index that reports ``ready=False``, which is a long
    way from the typo that caused it.
    """
    from potpie_context_engine.adapters.outbound.resources.index import (
        KNOWN_PROFILES,
    )

    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"off", "disabled"}:
        normalized = "none"
    if normalized not in KNOWN_PROFILES:
        fail(
            code="validation_error",
            message=f"unknown resource index profile {value!r}",
            detail={"key": "resource_index", "profiles": list(KNOWN_PROFILES)},
            next_action=f"use one of: {', '.join(KNOWN_PROFILES)}",
            exit_code=EXIT_VALIDATION,
        )
    return normalized


def _build_context_status_report(
    shell,
    *,
    pot_id: str,
    intent: str,
    harness: str,
    data_plane,
    quality_summary=None,
) -> StatusReport:
    """Join root-owned status surfaces with the engine-owned data plane."""
    aggregate = get_pot_service(shell).aggregate_status(pot_id=pot_id)
    active = aggregate.active_pot
    nudge = get_skill_service(shell).nudge(agent=harness) if harness else None
    backend_ready = bool(data_plane.backend_ready)
    quality = quality_block(dict(data_plane.quality), summary=quality_summary)
    next_action = status_next_action(
        has_pot=active is not None,
        backend_ready=backend_ready,
        quality=quality,
    )
    return StatusReport(
        pot_id=pot_id,
        profile=shell.profile,
        daemon_up=True,
        active_pot=active.name if active else None,
        backend_ready=backend_ready,
        data_plane={
            "backend_profile": data_plane.backend_profile,
            "backend_ready": data_plane.backend_ready,
            "reader_backed_includes": list(data_plane.reader_backed_includes),
            "counts": dict(data_plane.counts),
            "freshness": dict(data_plane.freshness),
            "quality": quality,
        },
        pot_summary={
            "pot_count": aggregate.pot_count,
            "sources": [source.name for source in aggregate.sources],
        },
        skills=nudge,
        recommended_next_action=next_action,
        metadata={"intent": intent},
    )


def _status_quality_summary(client):
    """The graph-quality summary ``graph quality summary`` reports, for status.

    The backend's quality projection only counts claims, so without it status
    reports a healthy graph however many findings are open. A failure here
    never fails status; it is reported as an unavailable quality block.
    """
    from potpie.cli.commands._common import EngineClientError

    try:
        return run_engine_operation(
            client.quality(
                QualityRequest(
                    report="summary",
                    limit=QUALITY_SUMMARY_LIMIT,
                    confidence_threshold=LOW_CONFIDENCE_THRESHOLD,
                )
            )
        )
    except EngineClientError as exc:
        message = str(getattr(exc.error, "message", exc))
        return QualitySummaryUnavailable(
            detail=f"quality summary unavailable: {message}"
        )
    except Exception as exc:  # noqa: BLE001 - status must survive a bad probe
        return QualitySummaryUnavailable(detail=f"quality summary unavailable: {exc}")


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
    data_plane = dict(report.data_plane)
    counts = data_plane.get("counts") or {}
    if counts:
        lines.append(f"  graph: {counts}")
    quality_line = _quality_line(data_plane.get("quality"))
    if quality_line:
        lines.append(quality_line)
    if report.skills and (report.skills.missing or report.skills.outdated):
        lines.append(
            f"  skills: missing={list(report.skills.missing)} → {report.skills.install_command}"
        )
    if report.recommended_next_action:
        lines.append(f"  next: {report.recommended_next_action}")
    return "\n".join(lines)


def _quality_line(quality) -> str | None:
    """The graph-quality summary as one human line, or ``None`` if there is none.

    Whatever the JSON quality block knows, the prose says: the open findings,
    or why they could not be counted.
    """
    if not isinstance(quality, Mapping):
        return None
    status = quality.get("findings_status") or quality.get("status")
    if quality.get("findings_status") == "unavailable":
        detail = quality.get("detail")
        return f"  quality: unavailable{f' — {detail}' if detail else ''}"
    if "open_findings" not in quality:
        return f"  quality: {status}" if status else None
    open_findings = int(quality.get("open_findings") or 0)
    return f"  quality: {status or 'unknown'} ({open_findings} open findings)"


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
        # `auto`, not `sentence-transformers`: the base install leaves the
        # embeddings extra out, and an explicit choice it cannot honour warns on
        # every later command. `auto` uses sentence-transformers once the extra
        # is installed; setup reports the fallback, and the extra, once.
        configured = configured_embedder_choice()
        choice = normalize_embedding_mode(configured or "auto")
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

    from potpie.cli.commands._common import set_runtime
    from potpie_context_engine.adapters.outbound.graph.backends import build_backend
    from potpie.runtime.composition import build_local_runtime

    selected_backend = backend or default_backend
    if daemon is not None:
        os.environ["CONTEXT_ENGINE_HOST_MODE"] = "daemon" if daemon else "in_process"
    runtime = build_local_runtime(backend=build_backend(selected_backend))
    set_runtime(runtime)
    host = runtime.root
    return (
        host,
        host.backend.profile,
        getattr(get_daemon_service(host), "in_process", False),
    )


def _validate_existing_daemon_backend(host, *, requested_backend: str | None) -> None:
    if not requested_backend:
        return
    daemon_status = get_daemon_service(host).status()
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

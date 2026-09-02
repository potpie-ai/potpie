"""Bootstrap + profile commands: ``setup`` / ``status`` / ``doctor`` / ``config``.

``setup`` runs the documented idempotent first-run sequence against the host
services (proving the journey shape). ``status`` is the cheap aggregate composed
from all three services via ``context_status``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    get_host_for,
    invalidate_host_snapshot,
    is_json,
    origin_from_use_flags,
    repo_default_pot_id,
    repo_effective_pot_info,
    require_text,
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
from potpie_context_engine.application.services.config_service import (
    KNOWN_CONFIG_KEYS,
    is_known_config_key,
    is_secret_config_key,
    public_config_value,
)
from potpie_context_engine.bootstrap import sentry_metrics_runtime
from potpie_context_engine.domain.embedding_modes import (
    DISABLED_EMBEDDER_ALIASES,
    HASHING_EMBEDDER_ALIASES,
    SEMANTIC_EMBEDDER_ALIASES,
    normalize_embedding_mode,
)
from potpie_context_core.errors import (
    CapabilityNotImplemented,
    ContextEngineDisabled,
    PotNotFound,
)
from potpie_context_core.lifecycle import SetupPlan, SetupReport
from potpie_context_core.ports.agent_context import StatusRequest


@dataclass(frozen=True, slots=True)
class _Gap:
    """Why one ``doctor`` probe could not answer, and what to do about it.

    The repair is carried separately rather than folded into the sentence
    because it is the only thing the report can act on: when the *first* probe
    is the one that failed, this is where ``recommended_next_action`` comes
    from, and the error boundary's generic "check readiness with 'potpie
    doctor'" is never an answer inside doctor.
    """

    detail: str
    next_action: str | None = None

    def __str__(self) -> str:  # pragma: no cover - convenience for f-strings
        return self.detail


def _probe(call: Callable[[], Any]) -> tuple[Any, _Gap | None]:
    """Run one ``doctor`` probe, turning a refusal into a reportable gap.

    A diagnostic that dies on the first unavailable surface is the opposite of
    a diagnostic: the operator learns one thing and loses the other twelve.
    That is not hypothetical against a managed host, where ``ledger`` is not
    deployed and answers by raising — enough, before this, to make
    ``potpie doctor`` unusable there and to hide the resource-index block
    entirely.

    The first four are the whole vocabulary a host uses to say "not here": a
    surface it does not implement, a backend that is down, a pot that is gone,
    an argument it will not accept.

    The last three are the *other* way a host fails to answer, and they are here
    because callers pass the unwrapping in with the call. A managed service
    built from a different revision of this repo can answer with a payload whose
    shape has moved — a dict where a DTO was, a field that has been renamed —
    and reading it raises ``AttributeError``/``TypeError``/``KeyError`` well
    after the request succeeded. That is cross-repo drift, the same family this
    command exists to diagnose, so it must become a row rather than take the
    other twelve rows down with it. Anything outside these seven is a real bug
    and still crashes, loudly, where it can be seen.
    """
    try:
        return call(), None
    except (
        CapabilityNotImplemented,
        ContextEngineDisabled,
        PotNotFound,
        ValueError,
        AttributeError,
        TypeError,
        KeyError,
    ) as exc:
        return None, _Gap(
            str(exc), getattr(exc, "recommended_next_action", None) or None
        )


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


def _doctor_next_action(
    readiness: Any, sections: dict[str, dict[str, Any]]
) -> str | None:
    """The one repair worth printing — and never a pointer back at doctor.

    Ordering matters more than it looks. A probe that failed usually carries the
    repair its raiser chose, and that beats any guess made here. Only when
    nothing failed with advice does the backend line apply, and only when the
    backend actually answered: telling someone to run ``potpie backend doctor``
    against a host that never replied sends them to diagnose the wrong thing.

    What must never appear is the error boundary's fallback, "check
    backend/daemon readiness with 'potpie doctor'". It is the right answer for
    every other command and a dead end here — it is what the operator was
    already running when the daemon turned out to be down.
    """
    for row in sections.values():
        if row["status"] != "ok" and row.get("recommended_next_action"):
            return str(row["recommended_next_action"])
    if (sections.get("backend_capabilities") or {}).get("status") != "ok":
        return (
            "the active host did not answer: start the local daemon with "
            "'potpie setup', or check the configured host with 'potpie host list'"
        )
    if readiness is not None and readiness.get("ready", False):
        return None
    # A backend that answered "not ready" already said why, and on a base
    # install the why is a package name. Printing "run backend doctor" over the
    # top of "the driver is not installed" sends the operator to re-read the
    # sentence they are being shown.
    detail = (readiness or {}).get("detail")
    if detail:
        return str(detail)
    return "Run `potpie backend doctor` or inspect `potpie graph status --json`."


#: Doctor's remote blocks, unwrapped from the host's answer into plain dicts.
#:
#: Each runs inside its own ``_probe``, which is the whole point: reading a
#: renamed or re-typed field off a drifted managed host raises there and becomes
#: one ``unavailable`` row, instead of escaping the command and costing the
#: operator every other block in the report.
def _readiness_block(readiness: Any) -> dict[str, Any]:
    return {
        "profile": readiness.profile,
        "ready": readiness.ready,
        "capability_ready": dict(readiness.capability_ready),
        "detail": readiness.detail,
    }


def _ledger_block(ledger: Any) -> dict[str, Any]:
    return {"available": ledger.available, "binding": ledger.binding}


def _resources_block(resources: Any) -> dict[str, Any]:
    return {
        "kind": resources.kind,
        "ready": resources.ready,
        "location": resources.location,
        "documents": resources.documents,
        "detail": resources.detail,
    }


def _resource_index_block(index: Any) -> dict[str, Any]:
    return {
        "profile": index.profile,
        "ready": index.ready,
        "capabilities": list(index.capabilities),
        "match_mode": index.match_mode,
        "documents": index.documents,
        "chunks": index.chunks,
        "pending_embeddings": index.pending_embeddings,
        "embedder": index.embedder,
        "shared_store": index.shared_store,
        "detail": index.detail,
    }


def _doctor_human(
    *,
    host_label: str,
    daemon_status: dict[str, Any],
    cli_install: dict[str, Any],
    profile: str | None,
    caps: Any,
    readiness: Any,
    readiness_gap: _Gap | None,
    ledger: Any,
    ledger_gap: _Gap | None,
    resources: Any,
    resources_gap: _Gap | None,
    resource_index: Any,
    resource_index_gap: _Gap | None,
    repo_identity: str | None,
    effective_current_repo_pot: str | None,
    default_pot_id: str | None,
    embedded_servers: dict[str, Any] | None,
    advertised: frozenset[str] | None,
    degraded: list[str],
) -> str:
    """The same report as ``--json``, in lines.

    Built here rather than inline so the command body stays a list of probes,
    but the rule it enforces is the point: human mode is never a *superset* of
    the JSON and never a subset either. The host, daemon and install lines in
    particular print whether or not the remote probes answered — they are the
    blocks that need no host, and losing them to an unreachable service is the
    failure this whole command was rewritten for.
    """
    from potpie.daemon.surfaces import RPC_SURFACES

    lines = [
        f"host: {host_label}",
        f"daemon: {daemon_status.get('mode')} (up={daemon_status.get('up')})",
        cli_install_human(cli_install),
        f"backend: {profile or 'unknown'} "
        + (f"ready={readiness['ready']} " if readiness else f"({readiness_gap}) ")
        + f"caps={', '.join(caps) if caps is not None else '(unavailable)'}",
        f"ledger: {ledger['binding']} available={ledger['available']}"
        if ledger
        else f"ledger: unavailable — {ledger_gap}",
        (
            f"resources: {resources['kind']} ready={resources['ready']}"
            + (
                f" documents={resources['documents']}"
                if resources["documents"] is not None
                else ""
            )
            + (f" ({resources['location']})" if resources["location"] else "")
        )
        if resources
        else f"resources: unavailable — {resources_gap}",
        (
            f"resource index: {resource_index['profile']} "
            f"ready={resource_index['ready']} "
            f"mode={resource_index['match_mode']} "
            f"chunks={resource_index['chunks']}"
            + (
                f" pending={resource_index['pending_embeddings']}"
                if resource_index["pending_embeddings"]
                else ""
            )
            + (f"\n  ! {resource_index['detail']}" if resource_index["detail"] else "")
        )
        if resource_index
        else f"resource index: unavailable — {resource_index_gap}",
    ]
    if repo_identity:
        lines.append(
            f"repo: {repo_identity} → {effective_current_repo_pot}"
            + (
                f" (default={default_pot_id})"
                if default_pot_id
                else " (no repo default set)"
            )
        )
    if embedded_servers and embedded_servers.get("detail"):
        lines.append(f"embedded servers: {embedded_servers['detail']}")
    missing = sorted(RPC_SURFACES - advertised) if advertised is not None else []
    if missing:
        lines.append(f"host does not serve: {', '.join(missing)}")
    if degraded:
        lines.append(
            f"degraded: {', '.join(degraded)} (run with --json for the detail)"
        )
    return "\n".join(lines)


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
        # Kept declared, and hidden, so the flag can be *explained* rather than
        # met with Click's "No such option": there are command lines and scripts
        # out there passing it, and every one of them believes a scan happened.
        scan: bool = typer.Option(
            False,
            "--scan",
            help="Rejected: setup has no working-tree scan.",
            hidden=True,
        ),
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
        remote: str = typer.Option(
            None,
            "--remote",
            metavar="URL",
            help=(
                "Set this machine up as a remote-only client of the managed "
                "service at URL. Provisions nothing locally."
            ),
        ),
        token: str = typer.Option(
            None,
            "--token",
            help=(
                "API key for --remote. Omit when the service runs with auth disabled."
            ),
        ),
    ) -> None:
        """Idempotent first-run: provision config, storage, daemon, default pot, skills."""
        with contract():
            if remote is not None:
                _run_remote_setup(
                    url=remote,
                    token=token,
                    agent=agent,
                    dry_run=dry_run,
                    local_only_flags={
                        "--backend": backend is not None,
                        "--daemon/--in-process": daemon is not None,
                        "--embeddings": embeddings is not None,
                        "--embedding-model": embedding_model is not None,
                        "--scan": bool(scan),
                    },
                )
                return
            if token is not None:
                fail(
                    code="validation_error",
                    message="'--token' only applies with '--remote'.",
                    next_action=(
                        "pass '--remote <url> --token <key>' to configure a "
                        "managed host, or drop --token to provision this machine"
                    ),
                )
            _refuse_remote_setup_target()
            # Every option this run was given, checked against the registry the
            # runtime itself reads, before a single byte is written. Setup is
            # twelve steps deep and only the middle of it validated anything, so
            # a typo was discovered — when it was discovered at all — after
            # config.json, the backend store and the daemon already existed:
            # `--agent cursur` exited 0 having installed no skills, `--pot ''`
            # burned a hard step on a name the pot service was always going to
            # refuse, and `--embeddings semantic-transformers` persisted itself
            # into config.json, where every later process silently read it as
            # "unknown, fall back to hashing" — a durable downgrade of semantic
            # search that nothing ever reports.
            _validate_setup_options(
                pot=pot, agent=agent, embeddings=embeddings, repo=repo, scan=scan
            )
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
            from potpie_context_engine.bootstrap.host_wiring import (
                default_backend_profile,
            )

            # Setup runs in *this* process regardless of output mode. It used to
            # branch: humans got a locally-wired host, and `--json` got whichever
            # host the origin resolved to — for the default local origin, a
            # daemon RPC carrying the entire first run (model download, backend
            # provision, migrations, skill install) under one 30s client
            # deadline. On a cold machine that deadline expires while the daemon
            # completes the work, and the CLI reports `unavailable` at exit 2 for
            # a setup that succeeded. So the one mode agents and CI use was the
            # only mode that could report a false failure, and the two modes were
            # running two different programs. Running locally also puts `--repo .`
            # back in the caller's working directory: resolved daemon-side it
            # named whatever directory first started the daemon, so `setup` in
            # repo B registered repo A. The detached daemon is still provisioned
            # and hardened — the orchestrator's own `daemon` step does that.
            host, selected_backend, in_process = _build_local_setup_host(
                backend=backend,
                daemon=daemon,
                default_backend=default_backend_profile(),
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
                from potpie_context_engine.adapters.outbound.graph.backends import (
                    build_backend,
                )
                from potpie_context_engine.bootstrap.host_wiring import build_host_shell

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
                from potpie_context_engine.adapters.outbound.graph.backends import (
                    build_backend,
                )
                from potpie_context_engine.bootstrap.host_wiring import build_host_shell

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
                    from potpie_context_engine.bootstrap.host_wiring import (
                        build_host_shell,
                    )

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

            # One check for both output modes, before anything is written: a
            # daemon already serving a different backend must stop the run rather
            # than have setup provision underneath it. `--json` used to *start*
            # the daemon here first, because the run itself was about to travel
            # over its RPC; now that setup executes in-process the orchestrator's
            # own `daemon` step starts it, in dependency order, after config and
            # the backend exist.
            if not in_process:
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
            # Setup creates the pot and registers the repo; anything resolved
            # from the host before this point is stale now.
            invalidate_host_snapshot()
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

    # There is no `--host` here, and adding one back would be a regression. The
    # root callback owns `--host local|managed`; the boolean no-op this command
    # used to declare shadowed it, so `potpie status --host managed` consumed
    # the flag, left `managed` unclaimed, and failed with "Got unexpected extra
    # argument(s) (managed)" — a message naming neither the flag nor the fix.
    # Readiness is reported by default, so the no-op had nothing to do but break
    # the one spelling a caller would reach for.
    @root.command()
    def status(
        verify: bool = typer.Option(
            False,
            "--verify",
            help="Moved to `potpie auth status --verify`.",
        ),
        intent: str = typer.Option(
            "feature",
            "--intent",
            help="Intent to report readiness for.",
        ),
        harness: str = typer.Option(
            "claude",
            "--harness",
            help="Harness to report skill readiness for.",
        ),
        pot: str = typer.Option(
            None,
            "--pot",
            help="Pot to report on (defaults to the resolved scope for this repo).",
        ),
    ) -> None:
        """context_status — host, pot, backend, quality, and skill readiness."""
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
        """Diagnostics for the active host: backend, capabilities, skill drift.

        Doctor is the command you run *because* something is wrong, so no single
        unavailable surface may abort it. That is an invariant over every remote
        call in this function, not a list of the calls that were once known to
        fail: the previous round wrapped the four probes a managed host refuses
        and left bare the four it happened to answer, so a host that does not
        serve ``backend`` — or a local daemon that is simply not running — still
        took the entire report down, including the daemon, install and
        embedded-server blocks that needed no host at all. Everything therefore
        goes through ``section()``, which records a refusal as a row instead of
        raising it.

        **Doctor exits 0 whenever it produced a report.** Degradation is data:
        against a managed host two sections are permanently unavailable, and a
        non-zero exit there would make the command useless as a health gate.
        Branch on ``ok`` and ``degraded_sections`` in the payload, not on the
        exit code — and do not "fix" this later.
        """
        with contract():
            from potpie.cli import hosts
            from potpie.daemon.surfaces import RPC_SURFACES

            host = get_host()
            # One row per probe, in probe order: {"status", "detail",
            # "recommended_next_action"}. The shape is fixed whether the probe
            # answered or not, so an agent can branch on
            # sections[name]["status"] without first checking the key is there.
            sections: dict[str, dict[str, Any]] = {}

            def section(
                name: str, call: Callable[[], Any], *, skip: str | None = None
            ) -> tuple[Any, _Gap | None]:
                value, gap = (None, _Gap(skip)) if skip else _probe(call)
                sections[name] = {
                    "status": "ok" if gap is None else "unavailable",
                    "detail": gap.detail if gap else None,
                    "recommended_next_action": gap.next_action if gap else None,
                }
                return value, gap

            # Every host attribute is read *inside* the lambda: `get_host()`
            # binds late, so `host.backend` is itself the call that builds (and
            # can refuse) the host.
            # Each remote answer is unwrapped *inside* its probe, so a payload
            # whose shape has drifted is caught by the same net as a surface that
            # refuses. Unwrapped in the payload dict below instead, one renamed
            # field on a managed host killed the whole report — including the
            # daemon and install blocks that never needed a host.
            caps, _caps_gap = section(
                "backend_capabilities",
                lambda: list(host.backend.capabilities().implemented()),
            )
            profile, _profile_gap = section(
                "backend_profile", lambda: host.backend.profile
            )
            pot, pot_gap = section("active_pot", lambda: host.pots.active_pot())
            pot_id = getattr(pot, "pot_id", "") if pot is not None else ""
            # Readiness is per-pot and a managed backend rejects the empty id
            # outright, so a host with no pot — or one that could not be asked
            # which pot is active — is not asked at all.
            readiness, readiness_gap = section(
                "backend_readiness",
                lambda: _readiness_block(host.backend.mutation.readiness(pot_id)),
                skip=None
                if pot_id
                else (
                    pot_gap.detail
                    if pot_gap
                    else "no active pot; backend readiness is per-pot"
                ),
            )
            ledger, ledger_gap = section(
                "ledger", lambda: _ledger_block(host.ledger.status())
            )
            resources, resources_gap = section(
                "resources",
                lambda: _resources_block(host.resources.status(pot_id=pot_id or None)),
            )
            # Beside the store, not inside it: the bytes can be perfectly
            # healthy while the index that makes them findable is off, stale,
            # or mid-drain — and that gap is invisible until a search quietly
            # returns less than it should.
            resource_index, resource_index_gap = section(
                "resource_index",
                lambda: _resource_index_block(
                    host.resources.index_status(pot_id=pot_id or None)
                ),
            )
            # Always the local daemon, even when the active host is managed: a
            # remote service has no process here, and the local daemon's health
            # is what `potpie daemon restart` would act on either way.
            origin = hosts.current_origin()
            daemon_status, daemon_gap = section(
                "daemon", lambda: get_host_for(hosts.LOCAL).daemon.status()
            )
            if not isinstance(daemon_status, dict):
                # Same whole-key-set rule as the blocks below: the healthy shape
                # carries up/mode/home/pid/detail, so a fallback that dropped
                # `home`/`pid` would KeyError the very consumer this row exists
                # to keep answering.
                daemon_status = {
                    "up": None,
                    "mode": "unknown",
                    "home": None,
                    "pid": None,
                    "detail": daemon_gap.detail if daemon_gap else None,
                }

            repo_identity = current_repo_identity_for_cli()
            routing, _routing_gap = section(
                "repo_routing",
                lambda: (
                    _effective_current_repo_pot_id(
                        host,
                        repo_identity=repo_identity,
                        active_pot_id=pot_id or None,
                    ),
                    repo_default_pot_id(host, repo_identity),
                ),
            )
            effective_current_repo_pot, default_pot_id = routing or (None, None)

            # Not a section: `None` here means "this host does not publish a
            # surface list", which is the normal answer from anything older than
            # the `/surfaces` endpoint and not a failure of anything.
            advertised = hosts.advertised_surfaces(host)

            cli_install = collect_cli_install_status()
            embedded_servers = _embedded_graph_servers(profile or "")
            degraded = sorted(
                name for name, row in sections.items() if row["status"] != "ok"
            )
            emit(
                {
                    "ok": not degraded,
                    "sections": sections,
                    "degraded_sections": degraded,
                    "host": {
                        "origin": origin,
                        "label": hosts.origin_label(origin),
                        "advertised_surfaces": sorted(advertised)
                        if advertised is not None
                        else None,
                        # Never computed from silence: an unanswered host has an
                        # unknown surface list, not an empty one.
                        "missing_surfaces": sorted(RPC_SURFACES - advertised)
                        if advertised is not None
                        else None,
                    },
                    "daemon": daemon_status,
                    "embedded_graph_servers": embedded_servers,
                    "cli_install": cli_install,
                    "backend_profile": profile,
                    "backend_ready": readiness["ready"] if readiness else None,
                    # Degraded blocks keep the *whole* key set with `None`
                    # values, so a caller reading `payload["resources"]["documents"]`
                    # gets None rather than a KeyError on the very hosts this
                    # report exists for.
                    "backend_readiness": readiness
                    if readiness
                    else {
                        "profile": None,
                        "ready": None,
                        "capability_ready": {},
                        "detail": None,
                        "unavailable": readiness_gap.detail if readiness_gap else None,
                    },
                    "backend_capabilities": caps if caps is not None else [],
                    "active_pot": pot_id or None,
                    "effective_current_repo_pot": effective_current_repo_pot,
                    "repo_default_pot": default_pot_id,
                    "recommended_next_action": _doctor_next_action(readiness, sections),
                    "ledger": ledger
                    if ledger
                    else {
                        "available": False,
                        "binding": None,
                        "unavailable": ledger_gap.detail if ledger_gap else None,
                    },
                    "resources": resources
                    if resources
                    else {
                        "kind": None,
                        "ready": False,
                        "location": None,
                        "documents": None,
                        "detail": None,
                        "unavailable": resources_gap.detail if resources_gap else None,
                    },
                    "resource_index": resource_index
                    if resource_index
                    else {
                        "profile": None,
                        "ready": False,
                        "capabilities": [],
                        "match_mode": None,
                        "documents": None,
                        "chunks": None,
                        "pending_embeddings": None,
                        "embedder": None,
                        "shared_store": None,
                        "detail": None,
                        "unavailable": resource_index_gap.detail
                        if resource_index_gap
                        else None,
                    },
                },
                human=_doctor_human(
                    host_label=hosts.origin_label(origin),
                    daemon_status=daemon_status,
                    cli_install=cli_install,
                    profile=profile,
                    caps=caps,
                    readiness=readiness,
                    readiness_gap=readiness_gap,
                    ledger=ledger,
                    ledger_gap=ledger_gap,
                    resources=resources,
                    resources_gap=resources_gap,
                    resource_index=resource_index,
                    resource_index_gap=resource_index_gap,
                    repo_identity=repo_identity,
                    effective_current_repo_pot=effective_current_repo_pot,
                    default_pot_id=default_pot_id,
                    embedded_servers=embedded_servers,
                    advertised=advertised,
                    degraded=degraded,
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
        local: bool = typer.Option(
            False, "--local", help="Select a local-origin pot (same as 'local:<ref>')."
        ),
        managed: bool = typer.Option(
            False,
            "--managed",
            help="Select a managed-origin pot (same as 'managed:<ref>').",
        ),
        also_default_for_current_repo: bool = typer.Option(
            False,
            "--also-default-for-current-repo",
            help="Also set the current repo's local default pot to this pot.",
        ),
    ) -> None:
        """Select the active pot by name/id (top-level alias for `pot use`)."""
        with contract():
            # The flags are origin *selectors*, exactly equivalent to qualifying
            # the ref — not labels on a selection made somewhere else.
            requested_origin = origin_from_use_flags(local=local, managed=managed)
            host = get_host()
            payload, human = use_pot_selection(
                host,
                ref,
                also_default_for_current_repo=also_default_for_current_repo,
                origin=requested_origin,
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
            # Distinct from the omitted argument above: `config get ''` is a
            # read of a key that cannot exist, and it answered `{"": null}` —
            # indistinguishable from a real key that happens to be unset.
            key = require_text(key, argument="key", example="potpie config get backend")
            value = get_host().config.get(key)
            value = public_config_value(key, value)
            emit({key: value}, human=f"{key}={value}")

    @config_app.command("set")
    def config_set(key: str, value: str) -> None:
        """Persist one known config key. The write keeps the value; the echo does not.

        Both guards below will read as dead code to someone skimming, and both
        are load-bearing.

        The catalog check turns a typo back into a refusal: ``config set emebdder
        hashing`` used to persist a key nothing reads and print "set". It is also
        what keeps ``config.json`` from being pressed into service as a secret
        store — see ``is_known_config_key``.

        The redaction is forward defence. No key in today's catalog is
        secret-shaped, so with the check above it is currently unreachable — but
        this was the one config command of three that echoed its raw argument,
        to stdout *and* into the ``--json`` payload that agents and CI capture,
        while ``get``/``list`` redacted the same key. Sharing their predicate is
        what stops the writer drifting back apart from the readers the day a
        credential key joins the catalog.
        """
        with contract():
            if not is_known_config_key(key):
                fail(
                    code="validation_error",
                    message=f"unknown config key {key!r}",
                    detail={"key": key, "known_keys": list(KNOWN_CONFIG_KEYS)},
                    # Names the exit as well as the catalog. This gate is newer
                    # than the homes it applies to: anyone who ran `config set
                    # github_token` while it was still accepted has a key in
                    # config.json that nothing reads and that `config get` still
                    # answers `<redacted>` for, as though it were managed.
                    # Refusing to rewrite it is right — nothing would read the
                    # new value either — but refusing without naming `config
                    # unset` would leave a credential the CLI can neither rotate
                    # nor clear, which is a worse place than before the gate.
                    next_action=(
                        f"use one of: {', '.join(KNOWN_CONFIG_KEYS)} — "
                        "a key already stored under this name is read by nothing; "
                        f"remove it with 'potpie config unset {key}'"
                    ),
                    exit_code=EXIT_VALIDATION,
                )
            get_host().config.set(key, value)
            shown = public_config_value(key, value)
            emit(
                {
                    "key": key,
                    "value": shown,
                    # Both halves are needed. The key test alone missed the
                    # credential a URL value can carry (`ledger.url` is not a
                    # secret-shaped key), and the value comparison alone would
                    # tell a user who literally typed "<redacted>" that they had
                    # been redacted when nothing was withheld.
                    "redacted": is_secret_config_key(key) or shown != value,
                    "persisted": True,
                },
                human=f"set {key}={shown}",
            )

    @config_app.command("unset")
    def config_unset(key: str) -> None:
        """Remove one config key. Accepts keys the catalog no longer knows.

        Deliberately ungated, where ``set`` is gated. The catalog check on the
        writer strands every key this file used to accept, and the ones that
        matter are credentials: ``config set github_token`` was accepted for
        long enough that real homes hold one, ``config get`` still answers
        ``<redacted>`` for it as though something managed it, and nothing reads
        it. Sending that key back through the write gate is right — a rewrite
        would be just as dead — so removal is the only repair left, and gating
        it on the same catalog would refuse the exact keys it exists to clear.

        ``removed`` distinguishes "there was one, it is gone" from "there was
        nothing here". Both are exit 0 because both leave the user where they
        asked to be, but reporting the first for the second is the
        success-for-work-that-did-not-happen shape this CLI is being audited
        for.
        """
        with contract():
            # Ungated on the *catalog*, not on emptiness: `config unset ''`
            # cannot clear anything, and answering "'' was not set (nothing
            # removed)" at exit 0 reads as a checked, negative answer.
            key = require_text(
                key, argument="key", example="potpie config unset github_token"
            )
            removed = get_host().config.unset(key)
            emit(
                {"key": key, "removed": removed},
                human=(
                    f"unset {key}"
                    if removed
                    else f"{key} was not set (nothing removed)"
                ),
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


def _quality_line(quality: Any) -> str | None:
    """The graph-quality summary as one human line, or ``None`` if there is none.

    ``status`` grew a real quality block — the open findings ``graph quality``
    reports, not just the backend's projection — and only ``--json`` could see
    it. The human view showed counts and then, one line later, a next-action
    telling the reader to go and count the findings themselves. Whatever the
    envelope knows, the prose says.
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


#: The embedding modes ``--embeddings`` documents, in the order its help lists
#: them. Acceptance is still derived from the runtime's alias sets; this is only
#: what a refusal offers back.
_CANONICAL_EMBEDDING_MODES: tuple[str, ...] = (
    "sentence-transformers",
    "auto",
    "local",
    "none",
)


def _validate_setup_options(
    *, pot: str, agent: str, embeddings: str | None, repo: str, scan: bool
) -> None:
    """Refuse every setup option this machine cannot honour, before it writes.

    Each check names the registry the *runtime* reads, never a second list kept
    here: ``AGENT_TYPES`` is what ``install_agent_bundle`` dispatches on and what
    the post-setup wizard already filters by, and the embedding aliases are the
    exact sets ``build_embedder`` branches on. A parallel copy would drift, and a
    drifted allow-list is worse than none — it refuses harnesses that work.

    ``--backend`` is absent on purpose: ``build_backend`` already raises for an
    unknown profile, with the full profile list, before the host is built.
    """
    from potpie_context_engine.adapters.outbound.skills.agent_installer import (
        AGENT_TYPES,
    )

    if scan:
        fail(
            code="validation_error",
            message=(
                "'--scan' does nothing: setup has no working-tree scan, and never "
                "ran one. Repository knowledge is written by harness-led ingestion."
            ),
            next_action=(
                "drop --scan; setup already registers this repo as a source, and "
                "the harness fills the graph from it (see the potpie-repo-baseline "
                "skill, or 'potpie graph propose'/'potpie graph commit')"
            ),
        )

    # Not `require_text`: the failure this closes is a *pot* that cannot be
    # named, and the pot service's own refusal (which arrives eleven steps
    # later) is the wording to keep.
    if not pot.strip():
        fail(
            code="validation_error",
            message=(
                "A pot name cannot be empty or only whitespace — it is the ref "
                "'potpie pot use' and '--pot' resolve against."
            ),
            next_action="pass a name, e.g. --pot my-project",
        )

    normalized_agent = agent.strip().lower()
    if normalized_agent not in AGENT_TYPES:
        fail(
            code="validation_error",
            message=f"Unknown agent harness {agent!r}.",
            detail={"agent": agent, "known_agents": list(AGENT_TYPES)},
            next_action=(
                f"use one of: {', '.join(AGENT_TYPES)} — or '--agent default' to "
                "write AGENTS.md only and install no harness skills"
            ),
        )

    if embeddings is not None and _setup_embeddings_choice(embeddings) not in (
        DISABLED_EMBEDDER_ALIASES | HASHING_EMBEDDER_ALIASES | SEMANTIC_EMBEDDER_ALIASES
    ):
        fail(
            code="validation_error",
            message=f"Unknown embedding mode {embeddings!r}.",
            # The canonical four, not the alias set the check accepts: the
            # aliases exist for back-compat and half of them ("1", "off") do not
            # read as embedding modes at all.
            detail={
                "embeddings": embeddings,
                "known_modes": list(_CANONICAL_EMBEDDING_MODES),
            },
            next_action=(
                f"use one of: {', '.join(_CANONICAL_EMBEDDING_MODES)} — an "
                "unrecognised mode is persisted to config.json and read back as "
                "the hashing embedder by every later run, which downgrades "
                "semantic search for good without ever saying so"
            ),
        )

    _validate_setup_repo(repo)


def _validate_setup_repo(repo: str) -> None:
    """Refuse a ``--repo`` this machine cannot resolve to a real repository.

    Only *local* refs are checkable: a path is either there or it is not, and a
    path that is not there was registered anyway — as a source, and as the repo
    default — so every later repo-scoped command routed through a pot bound to a
    directory nobody has. A remote ref (``github.com/acme/shop``,
    ``git@github.com:acme/shop.git``) names something this command cannot reach
    and is left to the ingestion that will, exactly as ``source add`` leaves it.
    """
    raw = (repo or "").strip()
    if not raw:
        fail(
            code="validation_error",
            message="--repo cannot be empty.",
            next_action="pass a path or a remote ref, e.g. --repo . or --repo github.com/acme/shop",
        )
    if raw.lower() in (".", "current"):
        return
    if not raw.startswith((".", "~", "/")):
        return
    path = Path(raw).expanduser()
    if path.is_dir():
        return
    fail(
        code="validation_error",
        message=f"No such directory for --repo: {path}.",
        next_action=(
            "pass a directory that exists (e.g. --repo . for this one), or a "
            "remote ref like 'github.com/acme/shop'"
        ),
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


def _refuse_remote_setup_target() -> None:
    """Refuse ``--host managed setup``: setup only ever provisions this machine.

    Setup now runs entirely in-process so a cold first run cannot trip the
    daemon's client deadline. That also means it silently ignores wherever the
    invocation was pointed: ``--host managed setup`` provisioned a *local*
    backend, created a *local* pot and started a *local* daemon, then exited 0
    without ever contacting the host it was aimed at. An error that turns into a
    wrong-host success is the failure this module's sibling registry states as a
    rule — enumeration degrades, targeting fails loud.

    Keyed to an explicit override rather than to the resolved origin: a caller
    whose *persisted* pointer is managed has always been able to run `setup` to
    fix up their local install, and taking that away would strand them with no
    way to provision the machine they are typing on.
    """
    from potpie.cli import hosts

    if hosts.origin_overridden() and hosts.selected_origin() != hosts.LOCAL:
        label = hosts.origin_label(hosts.selected_origin())
        fail(
            code="validation_error",
            message=(
                f"'setup' provisions the local machine and cannot target "
                f"{label}: backend, daemon, default pot and skills are all local."
            ),
            next_action=(
                "run 'potpie setup' without --host to provision this machine, "
                "or 'potpie host list' to see what the hosts already hold"
            ),
        )


def _run_remote_setup(
    *,
    url: str,
    token: str | None,
    agent: str,
    dry_run: bool,
    local_only_flags: dict[str, bool],
) -> None:
    """Set this machine up as a client of a managed host, provisioning nothing.

    The other half of :func:`_refuse_remote_setup_target`. That function is
    right that ``setup`` provisions the local machine and must not be *aimed*
    at a remote one — but a remote-only install still has a first run, and until
    this existed there was no command that performed it. The base ``potpie``
    distribution ships without a local backend or a daemon, so on Windows, on
    Linux older than glibc 2.39, and on anyone who simply does not want a local
    graph, ``potpie setup`` was a wizard whose every step was inapplicable.

    What a client's first run actually consists of: prove the endpoint answers,
    record it, make it active, and install the skills — which are files on
    *this* filesystem and need no host at all (see
    ``potpie.cli.commands.skills``). No config.json, no backend, no daemon, no
    default pot: the pots live on the service, and creating a local one here
    would be the wrong-host success this module refuses everywhere else.

    Ordered so nothing is written until nothing can refuse it, matching
    ``host set``: address, then credential, then reachability, and only then the
    registry. A ``setup`` that stored an unreachable endpoint and exited 0 would
    leave the machine in the state it was run to get out of.
    """
    from potpie.cli import hosts
    from potpie.cli.commands.host import (
        probe_managed_endpoint,
        resolved_token as resolve_managed_token,
        validated_base_url,
    )

    named = [flag for flag, given in local_only_flags.items() if given]
    if named:
        # Refused rather than ignored: every one of these provisions local
        # storage, and a run that silently dropped them would report success for
        # a setup that did none of what was asked.
        fail(
            code="validation_error",
            message=(
                f"'--remote' provisions nothing on this machine, so "
                f"{', '.join(sorted(named))} cannot apply."
            ),
            next_action=(
                "drop those flags to configure the managed host, or run "
                "'potpie setup' without --remote to provision this machine"
            ),
        )

    base_url = validated_base_url(url)
    resolved_token = resolve_managed_token(token)

    if dry_run:
        emit(
            {
                "mode": "remote",
                "endpoint": base_url,
                "steps": [
                    "verify endpoint",
                    "store host",
                    "activate host",
                    "install skills",
                ],
                "dry_run": True,
            },
            human=(
                f"would point this machine at {base_url}, make it the active "
                f"host, and install {agent} skills locally"
            ),
        )
        return

    pots = probe_managed_endpoint(base_url, resolved_token)
    hosts.set_managed_endpoint(base_url, resolved_token)
    hosts.set_persisted_origin(hosts.MANAGED)

    # Built in process, exactly as `potpie skills install` does. A skill install
    # writes into this machine's harness directories, so it neither needs nor
    # wants the host that was just configured.
    from potpie_context_engine.bootstrap.host_wiring import build_skill_manager

    installed = build_skill_manager().install(
        agent=agent, skill_id=None, scope="global"
    )

    payload = {
        "mode": "remote",
        "endpoint": base_url,
        "active_origin": hosts.MANAGED,
        "pots_visible": len(pots),
        "agent": installed.agent,
        "skills_changed": len(installed.changed),
    }
    human = (
        f"managed host → {base_url} ({len(pots)} pots visible)\n"
        f"active host → managed\n"
        f"{agent} skills installed ({len(installed.changed)} changed)"
    )
    # Same warning `host set` gives, for the same reason: the write succeeded
    # and is still not what the next command will use.
    shadow = hosts.managed_env_override()
    payload["shadowed_by_env"] = shadow
    if shadow:
        human = (
            f"{human}\n! POTPIE_MANAGED_URL={shadow} is set and outranks the "
            "stored host; commands will use it until it is unset"
        )
    emit(payload, human=human)


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
    from potpie_context_engine.bootstrap.host_wiring import build_host_shell

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

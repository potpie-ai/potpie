"""Pot + source commands → ``HostShell.pots`` (PotManagementService)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import httpx
import typer

from adapters.inbound.cli.commands._common import (
    contract,
    current_repo_identity_for_cli,
    emit,
    enrich_with_pot_guidance,
    fail,
    get_host,
    pot_scope_info,
    repo_pot_candidates,
    resolve_pot_id,
)
from adapters.inbound.cli.telemetry.onboarding_events import (
    capture_project_binding_event,
    elapsed_ms,
    now_ms,
    sanitized_failure_kind,
)
from adapters.outbound.cli_auth.potpie_api_config import (
    resolve_potpie_api_base_url,
    resolve_potpie_auth_config,
)
from adapters.outbound.http.potpie_context_api_client import (
    PotpieContextApiClient,
    PotpieContextApiError,
)
from adapters.inbound.cli.repo_location import repo_identity_key, resolve_repo_location
from domain.errors import CapabilityNotImplemented

pot_app = typer.Typer(help="Pots: workspace/tenant boundaries.")
default_app = typer.Typer(help="Repo-local default pot routing.")
source_app = typer.Typer(
    help="Source registry for a pot; registration does not ingest or scan."
)
linear_team_app = typer.Typer(help="Linear teams attached to a context pot.")
jira_project_app = typer.Typer(help="Jira projects reachable from a context pot.")
pot_app.add_typer(default_app, name="default")


def _potpie_api_client() -> PotpieContextApiClient:
    try:
        return PotpieContextApiClient(
            resolve_potpie_api_base_url(),
            auth_headers_provider=lambda: resolve_potpie_auth_config().headers,
            reauth_provider=lambda: (
                resolve_potpie_auth_config(force_refresh=True).headers
            ),
            client_surface="cli",
            client_name="potpie-cli",
        )
    except ValueError as exc:
        fail(
            code="auth_missing",
            message="Potpie API not configured.",
            detail=str(exc),
            next_action="run 'potpie login' or set POTPIE_API_KEY",
        )


@pot_app.command("list")
def pot_list(
    local: bool = typer.Option(
        False, "--local", help="Local-origin pots only (default)."
    ),
    managed: bool = typer.Option(False, "--managed", help="Managed-origin pots only."),
    all_: bool = typer.Option(False, "--all", help="Local + managed pots."),
) -> None:
    with contract():
        # Managed-origin listing needs login + managed routing (HU3). '--managed'
        # alone is the structured not-implemented contract; '--all' shows locals
        # and flags managed as pending so it never crashes.
        if managed and not all_:
            raise CapabilityNotImplemented(
                "host.pots.list_managed",
                detail="managed pot listing is not implemented",
                recommended_next_action="run 'potpie login'; managed routing lands in HU3",
            )
        pots = get_host().pots.list_pots()
        payload: dict[str, object] = {
            "pots": [
                {"id": p.pot_id, "name": p.name, "active": p.active, "origin": "local"}
                for p in pots
            ],
        }
        pot_rows = [f"{'*' if p.active else ' '} {p.name} ({p.pot_id})" for p in pots]
        human_lines = ["Local", *(pot_rows or ["(no pots)"])]
        if all_:
            payload["managed_pending"] = True
            human_lines.append("  (managed pots require 'potpie login' — HU3)")
        emit(payload, human="\n".join(human_lines))


@pot_app.command("info")
def pot_info() -> None:
    with contract():
        active = get_host().pots.active_pot()
        if active is None:
            emit({"active_pot": None}, human="(no active pot)")
            return
        emit(
            {"active_pot": {"id": active.pot_id, "name": active.name}},
            human=f"active: {active.name} ({active.pot_id})",
        )


def _fail_api_unreachable(*, label: str, exc: httpx.RequestError) -> None:
    fail(
        code="api_unreachable",
        message=f"{label} failed.",
        detail=str(exc),
        next_action=(
            "check POTPIE_API_BASE_URL / POTPIE_API_KEY or run 'potpie login'; "
            "remote event ingestion is not served by the embedded local graph store"
        ),
    )


def _repo_key_from_option(repo: str) -> str:
    value = (repo or "").strip()
    if value in ("", ".", "current"):
        repo_key = current_repo_identity_for_cli()
    else:
        repo_key = repo_identity_key(value)
    if not repo_key:
        raise ValueError("--repo must resolve to a git remote or path")
    return repo_key


@pot_app.command("create")
def pot_create(
    name: str,
    repo: str = typer.Option(None, "--repo"),
    use: bool = typer.Option(False, "--use"),
) -> None:
    with contract():
        host = get_host()
        pot = host.pots.create_pot(name=name, repo=repo, use=use)
        payload, human = enrich_with_pot_guidance(
            host,
            pot.pot_id,
            {"id": pot.pot_id, "name": pot.name, "active": pot.active},
            human=(
                f"created pot '{pot.name}' ({pot.pot_id})"
                f"{' [active]' if pot.active else ''}"
            ),
            repo=repo,
        )
        emit(payload, human=human)


@pot_app.command("use")
def pot_use(ref: str) -> None:
    with contract():
        host = get_host()
        pot = host.pots.use_pot(ref=ref)
        payload, human = enrich_with_pot_guidance(
            host,
            pot.pot_id,
            {"id": pot.pot_id, "name": pot.name},
            human=f"active pot → {pot.name}",
        )
        emit(payload, human=human)


@pot_app.command("linked")
def pot_linked(repo: str = typer.Option("current", "--repo")) -> None:
    """Show pots linked to a repo source and the local default, if any."""
    with contract():
        host = get_host()
        linked = repo_pot_candidates(host, repo)
        candidates = list(linked.get("candidates", ()))
        repo_key = linked.get("repo")
        lines = [f"repo {repo_key or '(unknown)'}"]
        default_id = linked.get("default_pot_id")
        if default_id:
            default = next(
                (row for row in candidates if row.get("pot_id") == default_id),
                None,
            )
            default_name = default.get("name") if default else default_id
            lines.append(f"default: {default_name} ({default_id})")
        else:
            lines.append("default: (unset)")
        if candidates:
            for row in candidates:
                counts = row.get("counts") or {}
                markers = [
                    label
                    for label, enabled in (
                        ("default", row.get("default")),
                        ("active", row.get("active")),
                    )
                    if enabled
                ]
                suffix = f"  {', '.join(markers)}" if markers else ""
                lines.append(
                    f"  {row.get('name')} ({row.get('pot_id')}) "
                    f"sources={row.get('source_count', 0)} "
                    f"claims={counts.get('claims', 0)} entities={counts.get('entities', 0)}"
                    f"{suffix}"
                )
        else:
            lines.append("  (no linked pots)")
        emit(linked, human="\n".join(lines))


@default_app.command("show")
def pot_default_show(repo: str = typer.Option("current", "--repo")) -> None:
    with contract():
        host = get_host()
        linked = repo_pot_candidates(host, repo)
        default_id = linked.get("default_pot_id")
        payload = {
            "repo": linked.get("repo"),
            "default_pot_id": default_id,
            "candidates": linked.get("candidates", ()),
        }
        if not default_id:
            emit(
                payload,
                human=f"repo {linked.get('repo') or '(unknown)'} default: (unset)",
            )
            return
        info = pot_scope_info(host, default_id)
        emit(
            {**payload, "default_pot": info},
            human=f"repo {linked.get('repo')} default: {info['name']} ({default_id})",
        )


@default_app.command("set")
def pot_default_set(ref: str, repo: str = typer.Option("current", "--repo")) -> None:
    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, ref, infer_from_repo=False)
        repo_key = _repo_key_from_option(repo)
        host.pots.set_repo_default(repo=repo_key, pot_id=pot_id)
        info = pot_scope_info(host, pot_id)
        emit(
            {"repo": repo_key, "default_pot": info},
            human=f"repo {repo_key} default → {info['name']} ({pot_id})",
        )


@default_app.command("clear")
def pot_default_clear(repo: str = typer.Option("current", "--repo")) -> None:
    with contract():
        host = get_host()
        repo_key = _repo_key_from_option(repo)
        cleared = host.pots.clear_repo_default(repo=repo_key)
        emit(
            {"repo": repo_key, "cleared": cleared},
            human=(
                f"repo {repo_key} default cleared"
                if cleared
                else f"repo {repo_key} default was not set"
            ),
        )


@pot_app.command("rename")
def pot_rename(ref: str, new_name: str) -> None:
    with contract():
        pot = get_host().pots.rename_pot(ref=ref, new_name=new_name)
        emit({"id": pot.pot_id, "name": pot.name}, human=f"renamed → {pot.name}")


@pot_app.command("reset")
def pot_reset(
    ref: str = typer.Argument(None),
    confirm: bool = typer.Option(False, "--confirm"),
) -> None:
    with contract():
        host = get_host()
        target = ref or resolve_pot_id(host)
        pot = host.pots.reset_pot(ref=target, confirm=confirm)
        emit(
            {"id": pot.pot_id, "reset": True},
            human=f"reset graph state for '{pot.name}'",
        )


@linear_team_app.command("ingest")
def linear_team_ingest(
    team: str = typer.Argument(..., help="Linear team id or key, e.g. ENG."),
    pot: str = typer.Option(None, "--pot", help="Pot id/name (default: active pot)."),
    count: int = typer.Option(
        120,
        "--count",
        min=1,
        max=1000,
        help="Soft per-kind item limit for the one-shot ingestion playbook.",
    ),
) -> None:
    """Queue one-shot ingestion for a Linear team's recent graph history."""
    with contract():
        team_key = team.strip()
        if not team_key:
            raise ValueError("Linear team id/key is required.")

        pid = resolve_pot_id(get_host(), pot)
        source_id = f"one_shot_ingest:linear:{team_key.lower()}:{uuid.uuid4()}"
        try:
            status_code, data = _potpie_api_client().submit_event(
                pot_id=pid,
                source_system="linear",
                event_type="linear_team",
                action="one_shot_ingest",
                source_id=source_id,
                payload={"team": team_key, "count": count},
                provider=None,
                provider_host=None,
                repo_name=None,
                occurred_at=datetime.now(timezone.utc),
            )
        except PotpieContextApiError as exc:
            fail(
                code="api_error",
                message="Linear ingest failed.",
                detail=str(exc.detail),
            )
        except httpx.RequestError as exc:
            _fail_api_unreachable(label="Linear ingest", exc=exc)

        out = {
            "status": "queued" if status_code == 202 else data.get("status", "applied"),
            "pot_id": pid,
            "team": team_key,
            "count": count,
            "source_id": source_id,
            "event_id": data.get("event_id"),
            "batch_id": data.get("batch_id"),
        }
        if status_code == 409:
            out["status"] = "duplicate"
        emit(
            out,
            human=(
                f"Queued Linear team ingest for {team_key} in pot {pid} "
                f"(event {out.get('event_id') or 'unknown'})."
            ),
        )


@linear_team_app.command("diff-sync")
def linear_team_diff_sync(
    team: str = typer.Argument(..., help="Linear team id or key, e.g. ENG."),
    pot: str = typer.Option(None, "--pot", help="Pot id/name (default: active pot)."),
    since: str = typer.Option(
        None,
        "--since",
        help="ISO-8601 lower bound for source enumeration. Omit to use the "
        "last graph-audit cursor from history.",
    ),
    count: int = typer.Option(
        120,
        "--count",
        min=1,
        max=1000,
        help="Soft per-kind item limit for the diff-sync playbook.",
    ),
) -> None:
    """Queue an incremental graph-audit diff-sync for a Linear team."""
    with contract():
        team_key = team.strip()
        if not team_key:
            raise ValueError("Linear team id/key is required.")

        pid = resolve_pot_id(get_host(), pot)
        source_id = f"diff_sync:linear:{team_key.lower()}:{uuid.uuid4()}"
        payload: dict[str, object] = {"team": team_key, "count": count}
        if since:
            payload["since"] = since
        try:
            status_code, data = _potpie_api_client().submit_event(
                pot_id=pid,
                source_system="linear",
                event_type="linear_team",
                action="diff_sync",
                source_id=source_id,
                payload=payload,
                provider=None,
                provider_host=None,
                repo_name=None,
                occurred_at=datetime.now(timezone.utc),
            )
        except PotpieContextApiError as exc:
            fail(
                code="api_error",
                message="Linear diff-sync failed.",
                detail=str(exc.detail),
            )
        except httpx.RequestError as exc:
            _fail_api_unreachable(label="Linear diff-sync", exc=exc)

        out = {
            "status": "queued" if status_code == 202 else data.get("status", "applied"),
            "pot_id": pid,
            "team": team_key,
            "since": since,
            "count": count,
            "source_id": source_id,
            "event_id": data.get("event_id"),
            "batch_id": data.get("batch_id"),
        }
        if status_code == 409:
            out["status"] = "duplicate"
        emit(
            out,
            human=(
                f"Queued Linear team diff-sync for {team_key} in pot {pid} "
                f"(event {out.get('event_id') or 'unknown'})."
            ),
        )


@pot_app.command("archive")
def pot_archive(ref: str) -> None:
    with contract():
        pot = get_host().pots.archive_pot(ref=ref)
        emit({"id": pot.pot_id, "archived": True}, human=f"archived '{pot.name}'")


@source_app.command("add")
def source_add(
    kind: str = typer.Argument(..., help="repo | github | document | ..."),
    location: str = typer.Argument(
        ...,
        help="Path, owner/repo, URL, or integration location to register. For "
        "repos, '.' or 'current' registers the current repo (resolved to its "
        "git remote or absolute path before storing).",
    ),
    name: str = typer.Option(None, "--name", help="Optional display/source name."),
    pot: str = typer.Option(None, "--pot", help="Pot id/name (default: resolved pot)."),
    make_default: bool = typer.Option(
        True,
        "--default/--no-default",
        help="For repo sources, set this pot as the local default for this repo.",
    ),
) -> None:
    """Register source metadata only; no ingestion or repository scan is started."""
    with contract():
        host = get_host()
        source_kind = kind.strip()
        is_repo = source_kind.lower() == "repo"
        # Registration establishes the repo→pot mapping, so the target is the
        # explicit/active pot — never inferred from existing registrations.
        pot_id = resolve_pot_id(host, pot, infer_from_repo=False)
        resolved_location = (
            resolve_repo_location(location) if is_repo else location
        )
        started_ms = now_ms()
        capture_project_binding_event(
            "cli_onboarding_repo_source_add_started",
            entrypoint="direct_command",
            properties={"source_kind": source_kind},
        )
        repo_default_set = False
        try:
            repo_default_setter = None
            if is_repo and make_default:
                repo_default_setter = getattr(host.pots, "set_repo_default", None)
                if not callable(repo_default_setter):
                    fail(
                        code="repo_default_unavailable",
                        message="This host does not support repo default bindings.",
                        next_action="upgrade the local context-engine host",
                    )
            src = host.pots.add_source(
                pot_id=pot_id,
                kind=source_kind,
                location=resolved_location,
                name=name,
            )
            if is_repo and make_default:
                repo_key = repo_identity_key(resolved_location)
                if not repo_key:
                    fail(
                        code="repo_unresolved",
                        message="Could not resolve the repository identity.",
                        next_action="pass a repo location such as '<owner>/<repo>'",
                    )
                repo_default_setter(repo=repo_key, pot_id=pot_id)
                repo_default_set = True
        except Exception as exc:  # noqa: BLE001
            capture_project_binding_event(
                "cli_onboarding_repo_source_add_failed",
                entrypoint="direct_command",
                properties={
                    "source_kind": kind,
                    "failure_kind": sanitized_failure_kind(exc),
                    "duration_ms": elapsed_ms(started_ms),
                },
            )
            raise
        capture_project_binding_event(
            "cli_onboarding_repo_source_add_completed",
            entrypoint="direct_command",
            properties={
                "source_kind": src.kind,
                "step_state": "done",
                "duration_ms": elapsed_ms(started_ms),
            },
        )
        payload, human = enrich_with_pot_guidance(
            host,
            pot_id,
            {
                "source_id": src.source_id,
                "kind": src.kind,
                "name": src.name,
                "location": resolved_location,
                "pot_id": pot_id,
                "repo_default_set": repo_default_set,
                "registration_only": True,
            },
            human=(
                f"registered source {src.kind}:{src.name} ({src.source_id}) "
                f"at {resolved_location} in pot {pot_id}\n"
                + (f"set repo default -> {pot_id}\n" if repo_default_set else "")
                + "no ingestion or scan started"
            ),
            repo=resolved_location if is_repo else None,
        )
        emit(payload, human=human)


@source_app.command("list")
def source_list(pot: str = typer.Option(None, "--pot")) -> None:
    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        sources = host.pots.list_sources(pot_id=pot_id)
        pot_info = pot_scope_info(host, pot_id)
        human = (
            "\n".join(
                [
                    (
                        f"pot={pot_info['name']} ({pot_id}) "
                        f"sources={len(sources)} claims={(pot_info.get('counts') or {}).get('claims', 0)}"
                    ),
                    *(
                        f"  {s.kind}: {getattr(s, 'location', s.name)} ({s.source_id})"
                        for s in sources
                    ),
                ]
            )
            if sources
            else (
                f"pot={pot_info['name']} ({pot_id}) "
                f"sources=0 claims={(pot_info.get('counts') or {}).get('claims', 0)}\n"
                "(no sources)"
            )
        )
        payload, human = enrich_with_pot_guidance(
            host,
            pot_id,
            {
                "pot_id": pot_id,
                "pot": pot_info,
                "source_count": len(sources),
                "sources": [
                    {
                        "id": s.source_id,
                        "kind": s.kind,
                        "name": s.name,
                        "location": getattr(s, "location", None),
                    }
                    for s in sources
                ],
            },
            human=human,
        )
        emit(payload, human=human)


@source_app.command("status")
def source_status(source_id: str, pot: str = typer.Option(None, "--pot")) -> None:
    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        src = host.pots.source_status(pot_id=pot_id, source_id=source_id)
        emit(
            {"id": src.source_id, "status": src.status, "name": src.name},
            human=f"{src.name}: {src.status}",
        )


@source_app.command("remove")
def source_remove(source_id: str, pot: str = typer.Option(None, "--pot")) -> None:
    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        host.pots.remove_source(pot_id=pot_id, source_id=source_id)
        emit({"removed": source_id}, human=f"removed source {source_id}")


pot_app.add_typer(linear_team_app, name="linear-team")


@jira_project_app.command("ingest")
def jira_project_ingest(
    project_key: str = typer.Argument(..., help="Jira project key, e.g. PROJ."),
    pot: str = typer.Option(None, "--pot", help="Pot id/name (default: active pot)."),
    count: int = typer.Option(
        120,
        "--count",
        min=1,
        max=1000,
        help="Soft per-kind item limit for the one-shot ingestion playbook.",
    ),
) -> None:
    """Queue one-shot ingestion for a Jira project's recent epics and issues."""
    with contract():
        key = project_key.strip()
        if not key:
            raise ValueError("Jira project key is required.")

        pid = resolve_pot_id(get_host(), pot)
        source_id = f"one_shot_ingest:jira:{key.lower()}:{uuid.uuid4()}"
        try:
            status_code, data = _potpie_api_client().submit_event(
                pot_id=pid,
                source_system="jira",
                event_type="jira_project",
                action="one_shot_ingest",
                source_id=source_id,
                payload={"project_key": key, "count": count},
                provider=None,
                provider_host=None,
                repo_name=None,
                occurred_at=datetime.now(timezone.utc),
            )
        except PotpieContextApiError as exc:
            fail(
                code="api_error",
                message="Jira ingest failed.",
                detail=str(exc.detail),
            )
        except httpx.RequestError as exc:
            _fail_api_unreachable(label="Jira ingest", exc=exc)

        out = {
            "status": "queued" if status_code == 202 else data.get("status", "applied"),
            "pot_id": pid,
            "project_key": key,
            "count": count,
            "source_id": source_id,
            "event_id": data.get("event_id"),
            "job_id": data.get("job_id") or data.get("batch_id"),
        }
        if status_code == 409:
            out["status"] = "duplicate"
        emit(
            out,
            human=(
                f"Queued Jira project ingest for {key} in pot {pid} "
                f"(event {out.get('event_id') or 'unknown'})."
            ),
        )


@jira_project_app.command("diff-sync")
def jira_project_diff_sync(
    project_key: str = typer.Argument(..., help="Jira project key, e.g. PROJ."),
    pot: str = typer.Option(None, "--pot", help="Pot id/name (default: active pot)."),
    since: str = typer.Option(
        None,
        "--since",
        help="ISO-8601 lower bound for source enumeration. Omit to use the "
        "last graph-audit cursor from history.",
    ),
    count: int = typer.Option(
        120,
        "--count",
        min=1,
        max=1000,
        help="Soft per-kind item limit for the diff-sync playbook.",
    ),
) -> None:
    """Queue an incremental graph-audit diff-sync for a Jira project."""
    with contract():
        key = project_key.strip()
        if not key:
            raise ValueError("Jira project key is required.")

        pid = resolve_pot_id(get_host(), pot)
        source_id = f"diff_sync:jira:{key.lower()}:{uuid.uuid4()}"
        payload: dict[str, object] = {"project_key": key, "count": count}
        if since:
            payload["since"] = since
        try:
            status_code, data = _potpie_api_client().submit_event(
                pot_id=pid,
                source_system="jira",
                event_type="jira_project",
                action="diff_sync",
                source_id=source_id,
                payload=payload,
                provider=None,
                provider_host=None,
                repo_name=None,
                occurred_at=datetime.now(timezone.utc),
            )
        except PotpieContextApiError as exc:
            fail(
                code="api_error",
                message="Jira diff-sync failed.",
                detail=str(exc.detail),
            )
        except httpx.RequestError as exc:
            _fail_api_unreachable(label="Jira diff-sync", exc=exc)

        out = {
            "status": "queued" if status_code == 202 else data.get("status", "applied"),
            "pot_id": pid,
            "project_key": key,
            "since": since,
            "count": count,
            "source_id": source_id,
            "event_id": data.get("event_id"),
            "job_id": data.get("job_id") or data.get("batch_id"),
        }
        if status_code == 409:
            out["status"] = "duplicate"
        emit(
            out,
            human=(
                f"Queued Jira project diff-sync for {key} in pot {pid} "
                f"(event {out.get('event_id') or 'unknown'})."
            ),
        )


pot_app.add_typer(jira_project_app, name="jira-project")

__all__ = ["pot_app", "source_app"]

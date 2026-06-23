"""Pot + source commands → ``HostShell.pots`` (PotManagementService)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import typer

from adapters.inbound.cli import repo_location
from adapters.inbound.cli.commands._common import (
    contract,
    current_repo_identity_for_cli,
    emit,
    fail,
    get_host,
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
from domain.errors import CapabilityNotImplemented

pot_app = typer.Typer(help="Pots: workspace/tenant boundaries.")
pot_default_app = typer.Typer(help="Current-repo default pot.")
source_app = typer.Typer(help="Sources registered to a pot.")
linear_team_app = typer.Typer(help="Linear teams attached to a context pot.")
jira_project_app = typer.Typer(help="Jira projects reachable from a context pot.")


def _repo_identity_for_cli(repo: str | None) -> str | None:
    raw = (repo or "").strip()
    if raw in ("", ".", "current"):
        return current_repo_identity_for_cli()
    return repo_location.repo_identity_key(raw)


def _pot_name(host, pot_id: str) -> str:
    for pot in host.pots.list_pots():
        if getattr(pot, "pot_id", None) == pot_id:
            return getattr(pot, "name", pot_id)
    return pot_id


def _potpie_api_client() -> PotpieContextApiClient:
    try:
        return PotpieContextApiClient(
            resolve_potpie_api_base_url(),
            auth_headers_provider=lambda: resolve_potpie_auth_config().headers,
            reauth_provider=lambda: resolve_potpie_auth_config(
                force_refresh=True
            ).headers,
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


@pot_app.command("create")
def pot_create(
    name: str,
    repo: str = typer.Option(None, "--repo"),
    use: bool = typer.Option(False, "--use"),
) -> None:
    with contract():
        pot = get_host().pots.create_pot(name=name, repo=repo, use=use)
        emit(
            {"id": pot.pot_id, "name": pot.name, "active": pot.active},
            human=f"created pot '{pot.name}' ({pot.pot_id}){' [active]' if pot.active else ''}",
        )


@pot_app.command("use")
def pot_use(ref: str) -> None:
    with contract():
        pot = get_host().pots.use_pot(ref=ref)
        emit({"id": pot.pot_id, "name": pot.name}, human=f"active pot → {pot.name}")


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


@pot_app.command("linked")
def pot_linked(
    repo: str = typer.Option(
        "current", "--repo", help="Repo ref to match, or 'current'."
    ),
) -> None:
    with contract():
        host = get_host()
        payload = repo_pot_candidates(host, repo)
        rows = payload.get("candidates", ())
        human = "\n".join(
            (
                f"{'*' if row.get('default') else ' '} "
                f"{row.get('name')} ({row.get('pot_id')})"
            )
            for row in rows
        )
        emit(payload, human=human or "(no linked pots)")


@pot_default_app.command("set")
def pot_default_set(
    ref: str,
    repo: str = typer.Option(
        "current", "--repo", help="Repo ref to bind, or 'current'."
    ),
) -> None:
    with contract():
        host = get_host()
        repo_identity = _repo_identity_for_cli(repo)
        if not repo_identity:
            fail(
                code="repo_unresolved",
                message="Could not resolve the repository identity.",
                next_action="run from a git checkout or pass '--repo <owner/repo>'",
            )
        pot_id = resolve_pot_id(host, ref, infer_from_repo=False)
        setter = getattr(host.pots, "set_repo_default", None)
        if not callable(setter):
            fail(
                code="repo_default_unavailable",
                message="This host does not support repo default bindings.",
                next_action="upgrade the local context-engine host",
            )
        setter(repo=repo_identity, pot_id=pot_id)
        name = _pot_name(host, pot_id)
        emit(
            {"repo": repo_identity, "default_pot": {"id": pot_id, "name": name}},
            human=f"default pot for {repo_identity} → {name} ({pot_id})",
        )


@pot_default_app.command("clear")
def pot_default_clear(
    repo: str = typer.Option(
        "current", "--repo", help="Repo ref to unbind, or 'current'."
    ),
) -> None:
    with contract():
        host = get_host()
        repo_identity = _repo_identity_for_cli(repo)
        if not repo_identity:
            fail(
                code="repo_unresolved",
                message="Could not resolve the repository identity.",
                next_action="run from a git checkout or pass '--repo <owner/repo>'",
            )
        clearer = getattr(host.pots, "clear_repo_default", None)
        if not callable(clearer):
            fail(
                code="repo_default_unavailable",
                message="This host does not support repo default bindings.",
                next_action="upgrade the local context-engine host",
            )
        cleared = bool(clearer(repo=repo_identity))
        emit(
            {"repo": repo_identity, "cleared": cleared},
            human=f"cleared default pot for {repo_identity}"
            if cleared
            else f"no default pot set for {repo_identity}",
        )


@source_app.command("add")
def source_add(
    kind: str = typer.Argument(..., help="repo | github | linear | …"),
    location: str = typer.Argument(...),
    name: str = typer.Option(None, "--name"),
    pot: str = typer.Option(None, "--pot"),
    set_default: bool = typer.Option(
        True,
        "--default/--no-default",
        help="For repo sources, bind this repo to the target pot.",
    ),
) -> None:
    with contract():
        host = get_host()
        source_kind = kind.strip()
        is_repo = source_kind.lower() == "repo"
        source_location = (
            repo_location.resolve_repo_location(location) if is_repo else location
        )
        pot_id = resolve_pot_id(host, pot, infer_from_repo=not is_repo)
        started_ms = now_ms()
        capture_project_binding_event(
            "cli_onboarding_repo_source_add_started",
            entrypoint="direct_command",
            properties={"source_kind": source_kind},
        )
        repo_default_set: bool | None = None
        try:
            src = host.pots.add_source(
                pot_id=pot_id, kind=source_kind, location=source_location, name=name
            )
            if is_repo and set_default:
                setter = getattr(host.pots, "set_repo_default", None)
                if callable(setter):
                    repo_identity = repo_location.repo_identity_key(source_location)
                    if repo_identity:
                        setter(repo=repo_identity, pot_id=pot_id)
                        repo_default_set = True
            elif is_repo:
                repo_default_set = False
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
        payload: dict[str, object] = {
            "source_id": src.source_id,
            "kind": src.kind,
            "name": src.name,
            "location": source_location,
            "pot_id": pot_id,
            "registration_only": True,
        }
        if repo_default_set is not None:
            payload["repo_default_set"] = repo_default_set
        emit(
            payload,
            human=(
                f"registered source {src.kind}:{src.name} ({src.source_id})\n"
                "  no ingestion or scan started"
            ),
        )


@source_app.command("list")
def source_list(pot: str = typer.Option(None, "--pot")) -> None:
    with contract():
        host = get_host()
        pot_id = resolve_pot_id(host, pot)
        sources = host.pots.list_sources(pot_id=pot_id)
        emit(
            {
                "sources": [
                    {
                        "id": s.source_id,
                        "kind": s.kind,
                        "name": s.name,
                        "location": getattr(s, "location", s.name),
                    }
                    for s in sources
                ]
            },
            human="\n".join(f"  {s.kind}: {s.name} ({s.source_id})" for s in sources)
            or "(no sources)",
        )


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


pot_app.add_typer(pot_default_app, name="default")
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

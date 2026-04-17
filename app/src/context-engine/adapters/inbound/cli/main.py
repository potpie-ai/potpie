import importlib.metadata
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer

from adapters.inbound.cli.credentials_store import (
    clear_active_pot_id,
    clear_credentials,
    clear_pot_scope_state,
    credentials_path,
    get_active_pot_id,
    get_pot_aliases,
    get_stored_api_base_url,
    get_stored_api_key,
    register_pot_alias,
    resolve_cli_pot_ref,
    set_active_pot_id,
    write_credentials,
)
from adapters.inbound.cli.agent_installer import install_agent_bundle
from adapters.inbound.cli.env_bootstrap import load_cli_env
from adapters.inbound.cli.git_project import (
    get_git_origin_remote_url,
    parse_git_remote,
    resolve_pot_id_from_git_cwd,
)
from adapters.inbound.cli.ingest_args import (
    default_episode_name,
    default_source_label,
    merge_file_body_into_ingest,
    resolve_ingest_body_and_pot,
)
from adapters.inbound.cli.output import (
    DoctorSnapshot,
    configure_cli_logging,
    emit_error,
    print_doctor_report,
    print_json_blob,
    print_ingest_result,
    print_plain_line,
    print_search_results,
)
from adapters.inbound.cli.potpie_api_config import (
    resolve_potpie_api_base_url,
    resolve_potpie_api_key,
)
from adapters.outbound.http.potpie_context_api_client import (
    PotpieContextApiClient,
    PotpieContextApiError,
)
from adapters.outbound.postgres.session import database_url
from adapters.outbound.settings_env import EnvContextEngineSettings
from bootstrap.http_projects import pot_map_from_env

try:
    __version__ = importlib.metadata.version("context-engine")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

app = typer.Typer(
    name="context-engine",
    help="Context graph CLI (configure via env; use HTTP API for full sync).",
    invoke_without_command=True,
    no_args_is_help=False,
)

pot_app = typer.Typer(help="Active pot and local pot helpers.")
pot_repo_app = typer.Typer(help="Repositories attached to a context pot (Potpie API).")

# Set by root callback; read by all subcommands (including nested `pot`).
_cli_state: dict[str, Any] = {"json": False, "verbose": False, "source": None}


def _flags() -> tuple[bool, bool]:
    return bool(_cli_state.get("json")), bool(_cli_state.get("verbose"))


def _resolved_source_label(explicit: Optional[str]) -> Optional[str]:
    """Subcommand --source wins over global `context-engine --source …`."""
    if explicit and explicit.strip():
        return explicit.strip()
    g = _cli_state.get("source")
    if isinstance(g, str) and g.strip():
        return g.strip()
    return None


def _cli_client_or_exit(verbose: bool) -> PotpieContextApiClient:
    try:
        return PotpieContextApiClient(
            resolve_potpie_api_base_url(),
            resolve_potpie_api_key(),
        )
    except ValueError as exc:
        emit_error("Potpie API not configured", str(exc), verbose=verbose)
        raise typer.Exit(code=1) from exc


def _format_api_error(exc: PotpieContextApiError) -> str:
    d = exc.detail
    if isinstance(d, dict):
        inner = d.get("detail")
        if isinstance(inner, dict):
            return json.dumps(inner)
        if isinstance(inner, str):
            return inner
        return json.dumps(d)
    return str(d)


def _ingest_result_from_http(status_code: int, data: dict[str, Any]) -> dict[str, Any]:
    if status_code == 202:
        return {
            "status": "queued",
            "episode_uuid": data.get("episode_uuid"),
            "event_id": data.get("event_id"),
            "job_id": data.get("job_id"),
        }
    if "event_id" in data or "job_id" in data:
        return {
            "status": "applied",
            "episode_uuid": data.get("episode_uuid"),
            "event_id": data.get("event_id"),
            "job_id": data.get("job_id"),
        }
    return {
        "status": "legacy_direct",
        "episode_uuid": data.get("episode_uuid"),
        "event_id": data.get("event_id"),
        "job_id": data.get("job_id"),
    }


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def _cli(
    _version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Machine-readable JSON on stdout (automation / piping).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose tracebacks on errors (API failures, etc.).",
    ),
    source: Optional[str] = typer.Option(
        None,
        "--source",
        "-s",
        help="Default source label for ingest and optional search filter (e.g. cli, mcp). "
        "May appear before the subcommand, e.g. `context-engine --source cli search …`.",
    ),
) -> None:
    if not _version:
        _cli_state["json"] = json_output
        _cli_state["verbose"] = verbose
        _cli_state["source"] = source.strip() if source and source.strip() else None
        load_cli_env()
        ve = os.getenv("CONTEXT_ENGINE_VERBOSE_NEO4J", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        configure_cli_logging(verbose or ve)


@app.command("login")
def login_cmd(
    token: str = typer.Argument(..., help="Potpie API key (from the app)"),
    url: Optional[str] = typer.Option(
        None,
        "--url",
        "-u",
        help="Potpie API base URL, e.g. http://127.0.0.1:8001 (optional; env still overrides)",
    ),
) -> None:
    """Save Potpie API key and optional base URL (required for search / ingest / reset via /api/v2/context)."""
    write_credentials(api_key=token, api_base_url=url)
    j, _ = _flags()
    print_plain_line(
        f"Saved credentials to {credentials_path()} (mode 600).",
        as_json=j,
        json_payload={"ok": True, "path": str(credentials_path())},
    )


@app.command("logout")
def logout_cmd() -> None:
    """Remove stored API credentials from disk."""
    clear_credentials()
    j, _ = _flags()
    print_plain_line(
        "Removed stored Potpie API credentials.",
        as_json=j,
        json_payload={"ok": True},
    )


@app.command("doctor")
def doctor_cmd() -> None:
    """Print Potpie API + local hints (CLI uses HTTP, not local Neo4j)."""
    load_cli_env()
    j, _ = _flags()
    s = EnvContextEngineSettings()
    cg = s.is_enabled()
    neo = bool(s.neo4j_uri())
    ce_neo = bool(
        os.getenv("CONTEXT_ENGINE_NEO4J_URI") or os.getenv("CONTEXT_ENGINE_NEO4J_URL")
    )
    legacy_neo = bool(os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL"))
    if ce_neo:
        neo_src = "context_engine"
    elif legacy_neo:
        neo_src = "legacy"
    else:
        neo_src = "missing"

    has_potpie_key_env = bool(os.getenv("POTPIE_API_KEY"))
    has_stored_key = bool(get_stored_api_key())
    base_env = os.getenv("POTPIE_API_URL") or os.getenv("POTPIE_BASE_URL")
    base_stored = get_stored_api_base_url()
    port = os.getenv("POTPIE_PORT") or os.getenv("POTPIE_API_PORT")
    db = bool(database_url())
    gh = bool(os.getenv("CONTEXT_ENGINE_GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN"))

    maps_set = bool(
        os.getenv("CONTEXT_ENGINE_REPO_TO_POT") or os.getenv("CONTEXT_ENGINE_POTS")
    )

    health_ok: Optional[bool] = None
    health_msg: Optional[str] = None
    auth_ok: Optional[bool] = None
    auth_msg: Optional[str] = None
    try:
        c = PotpieContextApiClient(
            resolve_potpie_api_base_url(),
            resolve_potpie_api_key(),
            timeout=15.0,
        )
        code, payload = c.get_health()
        health_ok = code == 200
        health_msg = None if health_ok else f"HTTP {code}"
        try:
            c.list_context_pots()
            auth_ok = True
            auth_msg = None
        except PotpieContextApiError as exc:
            auth_ok = False
            auth_msg = f"HTTP {exc.status_code}: {_format_api_error(exc)}"
    except ValueError:
        health_ok = None
        health_msg = "skipped (no base URL / API key)"
        auth_ok = None
        auth_msg = "skipped (no base URL / API key)"
    except OSError as exc:
        health_ok = False
        health_msg = str(exc)
        auth_ok = False
        auth_msg = str(exc)

    summary: list[str] = [
        "[dim]search / ingest / pot hard-reset call Potpie POST /api/v2/context/* with X-API-Key.[/dim]",
        "[dim]Local Neo4j/Graphiti is not required on this machine.[/dim]",
    ]
    if health_ok is True:
        summary.append("[green]GET /health on Potpie base URL succeeded.[/green]")
    if auth_ok is True:
        summary.append("[green]Authenticated /api/v2/context probe succeeded.[/green]")
    elif auth_ok is False:
        summary.append(
            "[red]Authenticated /api/v2/context probe failed; search / ingest / MCP calls will fail.[/red]"
        )

    snap = DoctorSnapshot(
        context_graph_enabled=cg,
        neo4j_effective_set=neo,
        neo4j_source=neo_src,
        pot_maps_set=maps_set,
        active_pot_id=get_active_pot_id() or None,
        potpie_api_key_env=has_potpie_key_env,
        potpie_stored_token=has_stored_key,
        potpie_base_url=base_env or base_stored or None,
        potpie_port_hint=(
            f"http://127.0.0.1:{port}"
            if port and not (base_env or base_stored)
            else None
        ),
        database_url_set=db,
        github_token_set=gh,
        potpie_health_ok=health_ok,
        potpie_health_message=health_msg,
        potpie_auth_ok=auth_ok,
        potpie_auth_message=auth_msg,
        summary_lines=summary,
    )
    print_doctor_report(snap, as_json=j)


@app.command("init-agent")
def init_agent_cmd(
    path: str = typer.Argument(
        ".",
        help="Repository path or a subdirectory inside it (default: current directory)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing AGENTS.md and .agents skill files when contents differ.",
    ),
) -> None:
    """Install AGENTS.md and repo-local context-engine skills into a repository."""
    j, v = _flags()
    try:
        result = install_agent_bundle(path, force=force)
    except ValueError as exc:
        emit_error("Invalid install path", str(exc), verbose=v)
        raise typer.Exit(code=1) from exc

    payload = result.to_dict()
    if j:
        print_json_blob(payload, as_json=True)
        return

    print_plain_line(f"Agent files root: {result.root}", as_json=False)
    for rel_path in result.created:
        print_plain_line(f"created {rel_path}", as_json=False)
    for rel_path in result.updated:
        print_plain_line(f"updated {rel_path}", as_json=False)
    for rel_path in result.unchanged:
        print_plain_line(f"unchanged {rel_path}", as_json=False)
    for rel_path in result.skipped:
        print_plain_line(
            f"skipped {rel_path} (use --force to overwrite)", as_json=False
        )


def _pot_id_or_git(explicit: str | None, *, cwd: str | None = None) -> str:
    if explicit:
        resolved, rerr = resolve_cli_pot_ref(explicit)
        if rerr or not resolved:
            _, v = _flags()
            emit_error(
                "Pot scope required", rerr or "Could not resolve pot id.", verbose=v
            )
            raise typer.Exit(code=1)
        return resolved
    pid, err = resolve_pot_id_from_git_cwd(cwd)
    if pid:
        return pid
    _, v = _flags()
    emit_error("Pot scope required", err or "Could not resolve pot id.", verbose=v)
    raise typer.Exit(code=1)


@pot_app.command("use")
def pot_use(
    pot_id: str = typer.Argument(
        ...,
        help="Pot UUID or a name from `pot alias` (see `pot pots`).",
    ),
) -> None:
    """Remember the active pot for CLI commands (stored in credentials.json).

    This is the global default for `search` and `ingest` when no pot UUID is passed.
    It takes precedence over CONTEXT_ENGINE_* repo maps. Use `pot unset` to fall back to maps + git.
    """
    j, v = _flags()
    resolved, rerr = resolve_cli_pot_ref(pot_id)
    if rerr or not resolved:
        emit_error("Unknown pot", rerr or "Could not resolve pot.", verbose=v)
        raise typer.Exit(code=1)
    set_active_pot_id(resolved)
    print_plain_line(
        f"Active pot set to {resolved}",
        as_json=j,
        json_payload={"ok": True, "active_pot_id": resolved, "input": pot_id},
    )


@pot_app.command("unset")
def pot_unset() -> None:
    """Clear the global active pot (use env maps + git origin again)."""
    clear_active_pot_id()
    j, _ = _flags()
    print_plain_line(
        "Active pot cleared.",
        as_json=j,
        json_payload={"ok": True, "active_pot_id": None},
    )


@pot_app.command("list")
def pot_list_cmd() -> None:
    """Show CONTEXT_ENGINE_POTS map, local name aliases, and active pot id."""
    load_cli_env()
    j, _ = _flags()
    data = {
        "active_pot_id": get_active_pot_id() or None,
        "pots_from_env": pot_map_from_env(),
        "pot_aliases": get_pot_aliases(),
    }
    print_json_blob(data, as_json=j)


@pot_app.command("pots")
def pot_pots_cmd() -> None:
    """List your context pots (GET /api/v2/context/pots)."""
    j, v = _flags()
    client = _cli_client_or_exit(v)
    try:
        rows = client.list_context_pots()
    except PotpieContextApiError as exc:
        emit_error("Could not list context pots", _format_api_error(exc), verbose=v)
        raise typer.Exit(code=1) from exc
    if j:
        print_json_blob({"context_pots": rows}, as_json=True)
        return
    if not rows:
        print_plain_line(
            'No context pots yet. Run `context-engine pot create "my-scope"` to create one on the server.',
            as_json=False,
        )
        return
    for p in rows:
        pid = p.get("id", "")
        dn = p.get("display_name", "")
        repo = p.get("primary_repo_name", "")
        print_plain_line(f"{pid}\t{dn}\t{repo}", as_json=False)
    print_plain_line(
        "[dim]Use: context-engine pot use <id>  or  pot alias <name> <id>[/dim]",
        as_json=False,
    )


@pot_app.command("alias")
def pot_alias_cmd(
    name: str = typer.Argument(..., help="Local nickname stored in credentials.json"),
    pot_uuid: str = typer.Argument(
        ...,
        metavar="POT_ID",
        help="Context pot UUID (see `pot pots`).",
    ),
) -> None:
    """Bind a short name to a pot UUID."""
    j, v = _flags()
    try:
        uuid.UUID(pot_uuid.strip())
    except ValueError:
        emit_error(
            "Invalid pot id",
            f"{pot_uuid!r} is not a UUID. Run `context-engine pot pots` to list pots.",
            verbose=v,
        )
        raise typer.Exit(code=1)
    resolved, rerr = resolve_cli_pot_ref(pot_uuid)
    if rerr or not resolved:
        emit_error("Invalid pot id", rerr or pot_uuid, verbose=v)
        raise typer.Exit(code=1)
    register_pot_alias(name, resolved)
    print_plain_line(
        f"Alias {name!r} -> {resolved}",
        as_json=j,
        json_payload={"ok": True, "name": name, "pot_id": resolved},
    )


@pot_app.command("clear-local")
def pot_clear_local_cmd() -> None:
    """Clear active pot and local aliases; keep saved API key / base URL."""
    j, _ = _flags()
    clear_pot_scope_state()
    print_plain_line(
        "Cleared active pot and pot_aliases (API login unchanged).",
        as_json=j,
        json_payload={"ok": True},
    )


@pot_app.command("create")
def pot_create_cmd(
    name: str = typer.Argument(
        ...,
        help="Display name; registers a local alias to the new server pot id",
    ),
) -> None:
    """Create a user-owned context pot on Potpie (POST /api/v2/context/pots) and save a local alias."""
    j, v = _flags()
    client = _cli_client_or_exit(v)
    try:
        row = client.create_context_pot(display_name=name)
    except PotpieContextApiError as exc:
        emit_error("Could not create context pot", _format_api_error(exc), verbose=v)
        raise typer.Exit(code=1) from exc
    pid = row.get("id")
    if not pid:
        emit_error("Unexpected response", str(row), verbose=v)
        raise typer.Exit(code=1)
    register_pot_alias(name, str(pid))
    data = {
        "id": pid,
        "display_name": row.get("display_name"),
        "primary_repo_name": row.get("primary_repo_name"),
        "alias": name,
        "message": f"Run `context-engine pot use {name!r}` or `pot use {pid}`.",
    }
    if j:
        print_json_blob(data, as_json=True)
    else:
        print_plain_line(
            f"Created context pot {pid} (alias {name!r}). "
            f"Try: context-engine pot use {name!r}",
            as_json=False,
        )


@pot_repo_app.command("list")
def pot_repo_list_cmd(
    pot_opt: Optional[str] = typer.Option(
        None,
        "--pot",
        help="Pot UUID or alias (default: active pot / git cwd, same as ingest).",
    ),
    cwd: Optional[str] = typer.Option(
        None,
        "--cwd",
        help="Git repo directory for pot inference when --pot is omitted.",
    ),
) -> None:
    """List repositories for a pot (GET /api/v2/context/pots/{id}/repositories)."""
    load_cli_env()
    j, v = _flags()
    pid = _pot_id_or_git(pot_opt, cwd=cwd)
    client = _cli_client_or_exit(v)
    try:
        rows = client.list_pot_repositories(pid)
    except PotpieContextApiError as exc:
        emit_error("Could not list repositories", _format_api_error(exc), verbose=v)
        raise typer.Exit(code=1) from exc
    if j:
        print_json_blob({"pot_id": pid, "repositories": rows}, as_json=True)
        return
    if not rows:
        print_plain_line(
            f"No repositories for pot {pid}. Try `pot repo add owner/repo`.",
            as_json=False,
        )
        return
    for r in rows:
        print_plain_line(
            f"{r.get('id', '')}\t{r.get('repo_name', '')}\t{r.get('provider', '')}",
            as_json=False,
        )


@pot_repo_app.command("add")
def pot_repo_add_cmd(
    owner_repo: str = typer.Argument(
        ...,
        help="GitHub repository as owner/repo (e.g. acme/api).",
    ),
    pot_opt: Optional[str] = typer.Option(
        None,
        "--pot",
        help="Pot UUID or alias (default: active pot / git cwd).",
    ),
    cwd: Optional[str] = typer.Option(
        None,
        "--cwd",
        help="Git repo directory for pot inference when --pot is omitted.",
    ),
) -> None:
    """Attach a GitHub.com repository to a pot (POST /api/v2/context/pots/{id}/repositories)."""
    load_cli_env()
    j, v = _flags()
    raw = owner_repo.strip()
    if "/" not in raw:
        emit_error("Invalid repo", "Use owner/repo (e.g. org/service).", verbose=v)
        raise typer.Exit(code=1)
    o, rn = raw.split("/", 1)
    o, rn = o.strip(), rn.strip()
    if not o or not rn:
        emit_error("Invalid repo", "owner and repo name required.", verbose=v)
        raise typer.Exit(code=1)
    pid = _pot_id_or_git(pot_opt, cwd=cwd)
    client = _cli_client_or_exit(v)
    try:
        out = client.add_pot_repository(pid, owner=o, repo=rn)
    except PotpieContextApiError as exc:
        emit_error("Could not add repository", _format_api_error(exc), verbose=v)
        raise typer.Exit(code=1) from exc
    if j:
        print_json_blob({"pot_id": pid, "result": out}, as_json=True)
    else:
        print_plain_line(
            f"Attached {o}/{rn} to pot {pid} (id {out.get('id', '')}).",
            as_json=False,
        )


pot_app.add_typer(pot_repo_app, name="repo")


@pot_app.command("hard-reset")
def pot_hard_reset(
    pot_id: Optional[str] = typer.Argument(
        None,
        help="Pot id (omit to infer from pot use / git --cwd, same as ingest).",
    ),
    skip_ledger: bool = typer.Option(
        False,
        "--skip-ledger",
        help="Do not delete Postgres ledger rows for this pot.",
    ),
    cwd: str = typer.Option(
        ".",
        "--cwd",
        help="Git working tree when inferring pot (ignored when pot id is passed).",
    ),
) -> None:
    """Delete all context-graph data for this pot (Potpie POST /api/v2/context/reset)."""
    j, v = _flags()
    cwd_resolved = str(Path(cwd).resolve())
    resolved = _pot_id_or_git(pot_id, cwd=cwd_resolved)
    client = _cli_client_or_exit(v)
    try:
        out = client.reset({"pot_id": resolved, "skip_ledger": skip_ledger})
    except PotpieContextApiError as exc:
        emit_error(
            "Hard reset failed",
            _format_api_error(exc),
            verbose=v,
            exc=exc if v else None,
        )
        raise typer.Exit(code=1) from exc
    if skip_ledger:
        out = {**out, "ledger_skipped": True}

    if not out.get("ok"):
        emit_error(
            "Hard reset failed",
            str(out.get("error") or "unknown"),
            verbose=v,
        )
        raise typer.Exit(code=1)

    print_json_blob(out, as_json=j)


app.add_typer(pot_app, name="pot")


@app.command("add")
def add_repo_cmd(
    path: str = typer.Argument(
        ".",
        help="Path to a git working tree (default: current directory)",
    ),
) -> None:
    """Inspect git remote for the path and print provider-scoped repo identity."""
    load_cli_env()
    j, _ = _flags()
    cwd = str(Path(path).resolve())
    url = get_git_origin_remote_url(cwd)
    if not url:
        emit_error(
            "Not a git repository",
            "Could not read origin remote.",
            hint="Run from a clone with git remote set.",
        )
        raise typer.Exit(code=1)
    parsed = parse_git_remote(url)
    if not parsed:
        emit_error("Invalid remote", f"Could not parse remote: {url!r}")
        raise typer.Exit(code=1)
    active = get_active_pot_id()
    data = {
        "cwd": cwd,
        "remote_url": url,
        "provider": parsed.provider,
        "provider_host": parsed.provider_host,
        "repo_name": parsed.owner_repo,
        "active_pot_id": active or None,
        "hint": "Map this repo to a pot with CONTEXT_ENGINE_REPO_TO_POT or `context-engine pot use`.",
    }
    print_json_blob(data, as_json=j)


@app.command("search")
def search(
    first: str = typer.Argument(
        ...,
        help="Either pot UUID + query, or only the query when pot is inferred (pot use / env / git)",
    ),
    second: Optional[str] = typer.Argument(
        None,
        help="Natural-language query when the first argument is the pot UUID",
    ),
    limit: int = typer.Option(8, "--limit", "-n", help="Max results (1–50)"),
    node_labels: Optional[str] = typer.Option(
        None,
        "--node-labels",
        help="Optional comma-separated Graphiti labels, e.g. PullRequest,Decision",
    ),
    repo_name: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Optional owner/repo to narrow results inside a multi-repo pot",
    ),
    source_description: Optional[str] = typer.Option(
        None,
        "--source",
        "-s",
        help="Optional episodic source label filter (e.g. cli). Overrides global --source when both are set.",
    ),
    include_invalidated: bool = typer.Option(
        False,
        "--include-invalidated",
        help="Return superseded facts too (Graphiti edges with invalid_at set). Ignored when --as-of is set.",
    ),
    with_temporal: bool = typer.Option(
        False,
        "--with-temporal",
        help="Plain output: show valid_at, invalid_at, created_at columns (JSON always includes them when present).",
    ),
    as_of: Optional[str] = typer.Option(
        None,
        "--as-of",
        help="ISO 8601 instant; restrict results to edges valid at that time (valid_at/invalid_at window).",
    ),
    cwd: str = typer.Option(
        ".",
        "--cwd",
        help="Git working tree used to read `origin` when inferring pot (ignored when pot UUID is explicit)",
    ),
) -> None:
    """Semantic search over Graphiti episodic entities (Potpie POST /api/v2/context/query/search)."""
    j, v = _flags()
    cwd_resolved = str(Path(cwd).resolve())
    if second is None:
        query = first
        pot_id = _pot_id_or_git(None, cwd=cwd_resolved)
    else:
        pot_id = _pot_id_or_git(first, cwd=cwd_resolved)
        query = second
    labels = None
    if node_labels:
        labels = [x.strip() for x in node_labels.split(",") if x.strip()]
    as_of_dt: Optional[datetime] = None
    if as_of:
        try:
            as_of_dt = _parse_iso_datetime(as_of)
        except ValueError as e:
            emit_error("Invalid --as-of", str(e), verbose=v)
            raise typer.Exit(code=1) from e
    body: dict[str, Any] = {
        "pot_id": pot_id,
        "query": query,
        "limit": limit,
        "node_labels": labels,
        "repo_name": repo_name,
        "source_description": _resolved_source_label(source_description),
        "include_invalidated": include_invalidated,
        "as_of": as_of_dt,
    }
    client = _cli_client_or_exit(v)
    try:
        rows = client.search(body)
    except PotpieContextApiError as exc:
        emit_error(
            "Search failed",
            _format_api_error(exc),
            verbose=v,
            exc=exc if v else None,
        )
        raise typer.Exit(code=1) from exc
    if not isinstance(rows, list):
        emit_error(
            "Search failed", f"Unexpected response: {type(rows).__name__}", verbose=v
        )
        raise typer.Exit(code=1)
    print_search_results(rows, as_json=j, with_temporal=with_temporal)


def _parse_iso_datetime(value: str) -> datetime:
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _ingest_resolve_error(code: str) -> tuple[str, str]:
    if code == "no_body":
        return (
            "Episode text required",
            "Pass text as a positional argument, or use --episode-body / -b.",
        )
    if code == "uuid_needs_body":
        return (
            "Episode text required",
            "After a pot UUID, pass the text as a second argument or use --episode-body / -b.",
        )
    if code == "two_args_first_not_uuid":
        return (
            "Invalid pot id",
            "With two positional arguments, the first must be a pot UUID.",
        )
    if code == "two_args_empty_body":
        return (
            "Episode text required",
            "Second argument or --episode-body / -b cannot be empty.",
        )
    return ("Could not parse ingest arguments", code)


def _ingest_file_conflict_error(code: str) -> tuple[str, str]:
    if code == "file_conflict_episode_body":
        return (
            "Conflicting episode sources",
            "Use either --file or --episode-body / -b, not both.",
        )
    if code == "file_conflict_second":
        return (
            "Conflicting episode sources",
            "With --file, do not pass episode text as the second positional argument.",
        )
    if code == "file_conflict_first":
        return (
            "Conflicting episode sources",
            "With --file, do not pass inline episode text as the first argument (use a pot UUID only if scoping).",
        )
    return ("Could not parse ingest arguments", code)


@app.command("ingest")
def ingest_cmd(
    first: Optional[str] = typer.Argument(
        None,
        help="Episode text, or pot UUID when a second argument is the text",
    ),
    second: Optional[str] = typer.Argument(
        None,
        help="Episode text when the first argument is a pot UUID",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Episode title (default: first line of body, truncated)",
    ),
    episode_body: Optional[str] = typer.Option(
        None,
        "--episode-body",
        "-b",
        help="Episode text (optional if you pass text as a positional argument)",
    ),
    episode_file: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="Read episode body from this file (UTF-8). Conflicts with inline body / -b.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    source_description: Optional[str] = typer.Option(
        None,
        "--source",
        "-s",
        help="Short source label (default: cli)",
    ),
    reference_time: Optional[str] = typer.Option(
        None,
        "--reference-time",
        "-t",
        help="ISO 8601 time (default: UTC now)",
    ),
    sync: bool = typer.Option(
        False,
        "--sync",
        help="Pass sync=true to Potpie (inline apply after persist when the server supports it).",
    ),
    idempotency_key: Optional[str] = typer.Option(
        None,
        "--idempotency-key",
        help="Optional dedupe key when Postgres is configured (matches HTTP idempotency_key).",
    ),
    cwd: str = typer.Option(
        ".",
        "--cwd",
        help="Git working tree for inferring repo/pot when pot UUID is omitted (default: current directory)",
    ),
) -> None:
    """Add a raw episode via Potpie POST /api/v2/context/ingest (X-API-Key).

    Quick form: context-engine ingest "Your text" (pot from pot use, env maps, or git --cwd).

    With no positional pot id, scope is: global pot use, then env repo maps, then git origin under --cwd.

    Use --file / -f to load the episode body from a UTF-8 file (optionally with an explicit pot UUID only).
    """
    j, v = _flags()
    file_body: Optional[str] = None
    if episode_file is not None:
        try:
            file_body = episode_file.expanduser().resolve().read_text(encoding="utf-8")
        except OSError as exc:
            emit_error("Could not read episode file", str(exc), verbose=v)
            raise typer.Exit(code=1) from exc
    try:
        merged_first, merged_body_opt = merge_file_body_into_ingest(
            first, second, episode_body, file_body
        )
        explicit_pot, body_text = resolve_ingest_body_and_pot(
            merged_first, second, merged_body_opt
        )
    except ValueError as exc:
        code = str(exc.args[0]) if exc.args else ""
        if code.startswith("file_conflict_"):
            title, detail = _ingest_file_conflict_error(code)
        else:
            title, detail = _ingest_resolve_error(code)
        emit_error(title, detail, verbose=v)
        raise typer.Exit(code=1) from exc
    cwd_resolved = str(Path(cwd).resolve())
    resolved_pot = _pot_id_or_git(explicit_pot, cwd=cwd_resolved)
    episode_name = (name.strip() if name else "") or default_episode_name(body_text)
    source = _resolved_source_label(source_description) or default_source_label()
    ref: datetime
    if reference_time:
        try:
            ref = _parse_iso_datetime(reference_time)
        except ValueError as e:
            emit_error("Invalid --reference-time", str(e), verbose=v)
            raise typer.Exit(code=1) from e
    else:
        ref = datetime.now(timezone.utc)

    body: dict[str, Any] = {
        "pot_id": resolved_pot,
        "name": episode_name,
        "episode_body": body_text,
        "source_description": source,
        "reference_time": ref,
        "idempotency_key": idempotency_key.strip() if idempotency_key else None,
    }
    client = _cli_client_or_exit(v)
    try:
        status_code, data = client.ingest(body, sync=sync)
    except PotpieContextApiError as exc:
        if exc.status_code == 409:
            detail = exc.detail
            ev = None
            if isinstance(detail, dict):
                inner = detail.get("detail")
                if isinstance(inner, dict):
                    ev = inner.get("event_id")
            emit_error(
                "Duplicate ingest",
                f"event_id={ev!r}" if ev else _format_api_error(exc),
                verbose=v,
            )
            raise typer.Exit(code=1) from exc
        emit_error(
            "Ingest failed",
            _format_api_error(exc),
            verbose=v,
            exc=exc if v else None,
        )
        raise typer.Exit(code=1) from exc

    out = _ingest_result_from_http(status_code, data if isinstance(data, dict) else {})
    if out["status"] in ("applied", "legacy_direct") and not out.get("episode_uuid"):
        emit_error(
            "Ingest failed",
            "Server returned no episode UUID (check Potpie logs or use --sync).",
            verbose=v,
        )
        raise typer.Exit(code=1)

    print_ingest_result(out, as_json=j)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

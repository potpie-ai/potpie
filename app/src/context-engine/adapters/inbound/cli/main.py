import importlib.metadata
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer

from adapters.inbound.cli.credentials_store import (
    clear_active_pot_id,
    clear_credentials,
    credentials_path,
    get_active_pot_id,
    get_stored_api_base_url,
    get_stored_api_key,
    set_active_pot_id,
    write_credentials,
)
from adapters.inbound.cli.env_bootstrap import load_cli_env
from adapters.inbound.cli.git_project import (
    get_git_origin_remote_url,
    parse_git_remote,
    resolve_pot_id_from_git_cwd,
)
from adapters.inbound.cli.ingest_args import (
    default_episode_name,
    default_source_label,
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
from adapters.outbound.graphiti.episodic import GraphitiEpisodicAdapter
from adapters.outbound.settings_env import EnvContextEngineSettings
from application.use_cases.ingest_episode import ingest_episode
from application.use_cases.query_context import search_pot_context
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

# Set by root callback; read by all subcommands (including nested `pot`).
_cli_state: dict[str, Any] = {"json": False, "verbose": False}


def _flags() -> tuple[bool, bool]:
    return bool(_cli_state.get("json")), bool(_cli_state.get("verbose"))


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
        help="Verbose logging (Neo4j driver); also CONTEXT_ENGINE_VERBOSE_NEO4J=1.",
    ),
) -> None:
    if not _version:
        _cli_state["json"] = json_output
        _cli_state["verbose"] = verbose
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
    """Save API token (optional; for future Potpie HTTP integrations — CLI pot scope uses env / pot use)."""
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
    """Print whether context graph env looks configured."""
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
    db = bool(os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL"))
    gh = bool(os.getenv("CONTEXT_ENGINE_GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN"))

    maps_set = bool(os.getenv("CONTEXT_ENGINE_REPO_TO_POT") or os.getenv("CONTEXT_ENGINE_POTS"))

    summary: list[str] = []
    if cg and neo:
        summary.append(
            "[green]Neo4j configured for Graphiti.[/green] `pot use` sets the global default pot; "
            "else CONTEXT_ENGINE_* maps + git `origin` (see `search` / `ingest`)."
        )
    elif not cg:
        summary.append(
            "[yellow]Context graph is opted out[/yellow] (CONTEXT_GRAPH_ENABLED=false or empty)."
        )
    elif not neo:
        summary.append(
            "[yellow]Set CONTEXT_ENGINE_NEO4J_* (preferred) or NEO4J_*[/yellow] for search/ingest."
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
            f"http://127.0.0.1:{port}" if port and not (base_env or base_stored) else None
        ),
        database_url_set=db,
        github_token_set=gh,
        summary_lines=summary,
    )
    print_doctor_report(snap, as_json=j)


def _pot_id_or_git(explicit: str | None, *, cwd: str | None = None) -> str:
    if explicit:
        return explicit
    pid, err = resolve_pot_id_from_git_cwd(cwd)
    if pid:
        return pid
    _, v = _flags()
    emit_error("Pot scope required", err or "Could not resolve pot id.", verbose=v)
    raise typer.Exit(code=1)


@pot_app.command("use")
def pot_use(
    pot_id: str = typer.Argument(..., help="Pot / project UUID to use as default scope"),
) -> None:
    """Remember the active pot for CLI commands (stored in credentials.json).

    This is the global default for `search` and `ingest` when no pot UUID is passed.
    It takes precedence over CONTEXT_ENGINE_* repo maps. Use `pot unset` to fall back to maps + git.
    """
    set_active_pot_id(pot_id)
    j, _ = _flags()
    print_plain_line(
        f"Active pot set to {pot_id}",
        as_json=j,
        json_payload={"ok": True, "active_pot_id": pot_id},
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
    """Show CONTEXT_ENGINE_POTS map and active pot id."""
    load_cli_env()
    j, _ = _flags()
    data = {
        "active_pot_id": get_active_pot_id() or None,
        "pots_from_env": pot_map_from_env(),
    }
    print_json_blob(data, as_json=j)


@pot_app.command("create")
def pot_create(name: str = typer.Argument(..., help="Display name (informational)")) -> None:
    """Generate a local pot id — prefer creating scope in Potpie; this is for standalone testing."""
    load_cli_env()
    j, _ = _flags()
    new_id = str(uuid.uuid4())
    data = {
        "message": "Add this id to CONTEXT_ENGINE_POTS or CONTEXT_ENGINE_REPO_TO_POT for your repo.",
        "suggested_pot_id": new_id,
        "name": name,
    }
    print_json_blob(data, as_json=j)


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
        emit_error("Not a git repository", "Could not read origin remote.", hint="Run from a clone with git remote set.")
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
    cwd: str = typer.Option(
        ".",
        "--cwd",
        help="Git working tree used to read `origin` when inferring pot (ignored when pot UUID is explicit)",
    ),
) -> None:
    """Semantic search over Graphiti episodic entities (same as HTTP /query/search and MCP)."""
    j, v = _flags()
    cwd_resolved = str(Path(cwd).resolve())
    if second is None:
        query = first
        pot_id = _pot_id_or_git(None, cwd=cwd_resolved)
    else:
        pot_id = first
        query = second
    settings = EnvContextEngineSettings()
    if not settings.is_enabled():
        emit_error(
            "Context graph disabled",
            "CONTEXT_GRAPH_ENABLED is false or empty.",
            verbose=v,
        )
        raise typer.Exit(code=1)
    episodic = GraphitiEpisodicAdapter(settings)
    reason = episodic.failure_reason()
    if reason:
        emit_error(
            "Graphiti unavailable",
            reason,
            hint="Check Neo4j env (CONTEXT_ENGINE_NEO4J_* or NEO4J_*) and graphiti-core install.",
            verbose=v,
        )
        raise typer.Exit(code=1)
    labels = None
    if node_labels:
        labels = [x.strip() for x in node_labels.split(",") if x.strip()]
    try:
        rows = search_pot_context(
            episodic,
            pot_id,
            query,
            limit=limit,
            node_labels=labels,
            repo_name=repo_name,
        )
    except Exception as exc:
        emit_error(
            "Search failed",
            str(exc) or type(exc).__name__,
            hint="Use --verbose for traceback.",
            verbose=v,
            exc=exc if v else None,
        )
        raise typer.Exit(code=1) from exc
    print_search_results(rows, as_json=j)


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
        return ("Episode text required", "Second argument or --episode-body / -b cannot be empty.")
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
    cwd: str = typer.Option(
        ".",
        "--cwd",
        help="Git working tree for inferring repo/pot when pot UUID is omitted (default: current directory)",
    ),
) -> None:
    """Add a raw episode to the episodic graph (same as HTTP POST /ingest).

    Quick form: context-engine ingest "Your text" (pot from pot use, env maps, or git --cwd).

    With no positional pot id, scope is: global pot use, then env repo maps, then git origin under --cwd.
    """
    j, v = _flags()
    settings = EnvContextEngineSettings()
    if not settings.is_enabled():
        emit_error(
            "Context graph disabled",
            "CONTEXT_GRAPH_ENABLED is false or empty.",
            verbose=v,
        )
        raise typer.Exit(code=1)
    episodic = GraphitiEpisodicAdapter(settings)
    reason = episodic.failure_reason()
    if reason:
        emit_error(
            "Graphiti unavailable",
            reason,
            hint="Check Neo4j env and graphiti-core install.",
            verbose=v,
        )
        raise typer.Exit(code=1)
    try:
        explicit_pot, body_text = resolve_ingest_body_and_pot(first, second, episode_body)
    except ValueError as exc:
        code = str(exc.args[0]) if exc.args else ""
        title, detail = _ingest_resolve_error(code)
        emit_error(title, detail, verbose=v)
        raise typer.Exit(code=1) from exc
    cwd_resolved = str(Path(cwd).resolve())
    resolved_pot = _pot_id_or_git(explicit_pot, cwd=cwd_resolved)
    episode_name = (name.strip() if name else "") or default_episode_name(body_text)
    source = (source_description.strip() if source_description else "") or default_source_label()
    ref: datetime
    if reference_time:
        try:
            ref = _parse_iso_datetime(reference_time)
        except ValueError as e:
            emit_error("Invalid --reference-time", str(e), verbose=v)
            raise typer.Exit(code=1) from e
    else:
        ref = datetime.now(timezone.utc)
    try:
        out = ingest_episode(
            episodic,
            resolved_pot,
            episode_name,
            body_text,
            source,
            ref,
        )
    except Exception as exc:
        emit_error(
            "Ingest failed",
            str(exc) or type(exc).__name__,
            verbose=v,
            exc=exc if v else None,
        )
        raise typer.Exit(code=1) from exc
    if out.get("episode_uuid") is None:
        emit_error(
            "Ingest failed",
            "Graphiti returned no episode UUID (check Neo4j connectivity and logs).",
            verbose=v,
        )
        raise typer.Exit(code=1)
    print_ingest_result(out, as_json=j)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

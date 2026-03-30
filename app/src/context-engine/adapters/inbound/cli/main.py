import importlib.metadata
import json
import os
from datetime import datetime, timezone
from typing import Optional

import typer

from adapters.inbound.cli.credentials_store import (
    clear_credentials,
    credentials_path,
    get_stored_api_base_url,
    get_stored_api_key,
    write_credentials,
)
from adapters.inbound.cli.env_bootstrap import load_cli_env
from adapters.inbound.cli.git_project import resolve_project_id_from_git_cwd
from adapters.outbound.graphiti.episodic import GraphitiEpisodicAdapter
from adapters.outbound.settings_env import EnvContextEngineSettings
from application.use_cases.ingest_episode import ingest_episode
from application.use_cases.query_context import search_project_context

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
) -> None:
    if not _version:
        load_cli_env()


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
    """Save API token (and optional base URL) for project auto-resolve and Potpie HTTP calls."""
    write_credentials(api_key=token, api_base_url=url)
    typer.echo(f"Saved credentials to {credentials_path()} (mode 600).")


@app.command("logout")
def logout_cmd() -> None:
    """Remove stored API credentials from disk."""
    clear_credentials()
    typer.echo("Removed stored Potpie API credentials.")


@app.command("doctor")
def doctor() -> None:
    """Print whether context graph env looks configured."""
    load_cli_env()
    s = EnvContextEngineSettings()
    cg = s.is_enabled()
    neo = bool(s.neo4j_uri())
    has_potpie_key_env = bool(os.getenv("POTPIE_API_KEY"))
    has_stored_key = bool(get_stored_api_key())
    potpie_auth = has_potpie_key_env or has_stored_key
    base_env = os.getenv("POTPIE_API_URL") or os.getenv("POTPIE_BASE_URL")
    base_stored = get_stored_api_base_url()
    port = os.getenv("POTPIE_PORT") or os.getenv("POTPIE_API_PORT")
    db = bool(os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL"))
    gh = bool(os.getenv("CONTEXT_ENGINE_GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN"))

    typer.echo("--- Context graph (search / ingest) ---")
    typer.echo(f"  CONTEXT_GRAPH_ENABLED: {cg}")
    typer.echo(f"  NEO4J_URI set: {neo}")
    if not cg:
        typer.echo(
            "  → search/ingest are off: CONTEXT_GRAPH_ENABLED is false, 0, no, off, or empty "
            "(default when unset is on)."
        )

    typer.echo("--- Potpie API (resolve project id from git origin) ---")
    typer.echo(f"  POTPIE_API_KEY in env (overrides stored): {has_potpie_key_env}")
    typer.echo(f"  Stored token (context-engine login): {has_stored_key}")
    eff_base = base_env or base_stored
    typer.echo(
        f"  Effective base URL: {eff_base or '(env/stored unset; will try http://127.0.0.1:8000 then :8001)'}"
    )
    if port and not eff_base:
        typer.echo(f"  POTPIE_PORT: {port} → http://127.0.0.1:{port}")
    if potpie_auth:
        typer.echo(
            "  → With the Potpie HTTP API running, `search`/`ingest` can resolve project id from origin."
        )
    else:
        typer.echo(
            "  → Run `context-engine login <token>` or set POTPIE_API_KEY for auto-resolve."
        )

    typer.echo("--- Other (sync / ledger / GitHub ingestion) ---")
    typer.echo(f"  DATABASE_URL / POSTGRES_URL: {db}")
    typer.echo(f"  GITHUB_TOKEN / CONTEXT_ENGINE_GITHUB_TOKEN: {gh}")

    typer.echo("--- Summary ---")
    if cg and neo and potpie_auth:
        typer.echo("  Looks ready for `search` with project id from git + Potpie API.")
    elif not cg:
        typer.echo(
            "  Context graph is opted out; remove CONTEXT_GRAPH_ENABLED=false or set it true."
        )
    elif not potpie_auth:
        typer.echo("  Run `context-engine login <token>` or set POTPIE_API_KEY.")


def _project_id_or_git(explicit: str | None) -> str:
    if explicit:
        return explicit
    pid, err = resolve_project_id_from_git_cwd()
    if pid:
        return pid
    typer.echo(err, err=True)
    raise typer.Exit(code=1)


@app.command("search")
def search(
    first: str = typer.Argument(
        ...,
        help="Either project UUID + query, or only the query when project is inferred from git",
    ),
    second: Optional[str] = typer.Argument(
        None,
        help="Natural-language query when the first argument is the project UUID",
    ),
    limit: int = typer.Option(8, "--limit", "-n", help="Max results (1–50)"),
    node_labels: Optional[str] = typer.Option(
        None,
        "--node-labels",
        help="Optional comma-separated Graphiti labels, e.g. PullRequest,Decision",
    ),
) -> None:
    """Semantic search over Graphiti episodic entities (same as HTTP /query/search and MCP get_project_context)."""
    if second is None:
        query = first
        project_id = _project_id_or_git(None)
    else:
        project_id = first
        query = second
    settings = EnvContextEngineSettings()
    if not settings.is_enabled():
        typer.echo(
            "Context graph is disabled (CONTEXT_GRAPH_ENABLED=false or empty).",
            err=True,
        )
        raise typer.Exit(code=1)
    episodic = GraphitiEpisodicAdapter(settings)
    labels = None
    if node_labels:
        labels = [x.strip() for x in node_labels.split(",") if x.strip()]
    rows = search_project_context(
        episodic,
        project_id,
        query,
        limit=limit,
        node_labels=labels,
    )
    typer.echo(json.dumps(rows, indent=2))


def _parse_iso_datetime(value: str) -> datetime:
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


@app.command("ingest")
def ingest_cmd(
    project_id: Optional[str] = typer.Argument(
        None,
        help="Project UUID; omit to infer from git origin + CONTEXT_ENGINE_REPO_TO_PROJECT / CONTEXT_ENGINE_PROJECTS",
    ),
    name: str = typer.Option(..., "--name", "-n", help="Episode title"),
    episode_body: str = typer.Option(
        ...,
        "--episode-body",
        "-b",
        help="Main episode text for Graphiti",
    ),
    source_description: str = typer.Option(
        ...,
        "--source",
        "-s",
        help="Short source label for the episode",
    ),
    reference_time: Optional[str] = typer.Option(
        None,
        "--reference-time",
        "-t",
        help="ISO 8601 time (default: UTC now)",
    ),
) -> None:
    """Add a raw episode to the episodic graph (same as HTTP POST /ingest)."""
    settings = EnvContextEngineSettings()
    if not settings.is_enabled():
        typer.echo(
            "Context graph is disabled (CONTEXT_GRAPH_ENABLED=false or empty).",
            err=True,
        )
        raise typer.Exit(code=1)
    episodic = GraphitiEpisodicAdapter(settings)
    resolved_project = _project_id_or_git(project_id)
    ref: datetime
    if reference_time:
        try:
            ref = _parse_iso_datetime(reference_time)
        except ValueError as e:
            typer.echo(f"Invalid --reference-time: {e}", err=True)
            raise typer.Exit(code=1) from e
    else:
        ref = datetime.now(timezone.utc)
    out = ingest_episode(
        episodic,
        resolved_project,
        name,
        episode_body,
        source_description,
        ref,
    )
    if out.get("episode_uuid") is None:
        typer.echo(
            "Failed to add episode (Graphiti unavailable or ingestion error).",
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(json.dumps(out, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()

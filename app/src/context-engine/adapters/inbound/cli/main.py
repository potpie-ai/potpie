import importlib.metadata
import json
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from adapters.inbound.cli.agent_installer import AGENT_TYPES, install_agent_bundle
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
    configure_error_output,
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
    IngestRejectedError,
    PotpieContextApiClient,
    PotpieContextApiError,
)
from adapters.outbound.http.potpie_local_api_client import (
    PotpieLocalApiClient,
    PotpieLocalApiError,
)
from adapters.outbound.settings_env import EnvContextEngineSettings
from bootstrap.http_projects import pot_map_from_env

try:
    __version__ = importlib.metadata.version("context-engine")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

app = typer.Typer(
    name="potpie",
    help="Context graph CLI (configure via env; use HTTP API for full sync).",
    invoke_without_command=True,
    no_args_is_help=False,
)

pot_app = typer.Typer(help="Active pot and local pot helpers.")
pot_repo_app = typer.Typer(help="Repositories attached to a context pot (Potpie API).")
event_app = typer.Typer(help="Inspect and wait for ingestion events.")
conflict_app = typer.Typer(help="Predicate-family conflicts (QualityIssue).")

# Set by root callback; read by all subcommands (including nested `pot`).
_cli_state: dict[str, Any] = {"json": False, "verbose": False, "source": None}


def _flags() -> tuple[bool, bool]:
    return bool(_cli_state.get("json")), bool(_cli_state.get("verbose"))


def _resolved_source_label(explicit: str | None) -> str | None:
    """Subcommand --source wins over global `potpie --source …`."""
    if explicit and explicit.strip():
        return explicit.strip()
    g = _cli_state.get("source")
    if isinstance(g, str) and g.strip():
        return g.strip()
    return None


def _cli_client_name() -> str:
    """Identify the concrete CLI client (env override, else ``potpie-cli``)."""
    for var in ("POTPIE_CLIENT_NAME", "CONTEXT_ENGINE_CLIENT_NAME"):
        v = os.getenv(var)
        if v and v.strip():
            return v.strip()
    return "potpie-cli"


def _cli_client_or_exit(verbose: bool) -> PotpieContextApiClient:
    try:
        return PotpieContextApiClient(
            resolve_potpie_api_base_url(),
            resolve_potpie_api_key(),
            client_surface="cli",
            client_name=_cli_client_name(),
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


def _database_url_env_set() -> bool:
    """Return whether local DB env is configured without importing optional SQLAlchemy."""
    return bool(
        os.getenv("DATABASE_URL")
        or os.getenv("POSTGRES_URL")
        or os.getenv("CONTEXT_ENGINE_DATABASE_URL")
        or os.getenv("POSTGRES_SERVER")
    )


def _resolve_local_api_base_url() -> str:
    for var in (
        "POTPIE_LOCAL_API_URL",
        "POTPIE_API_URL",
        "POTPIE_BASE_URL",
        "BASE_URL",
    ):
        value = os.getenv(var)
        if value and value.strip():
            return value.strip().rstrip("/")
    host = os.getenv("POTPIE_HOST") or os.getenv("POTPIE_API_HOST") or "127.0.0.1"
    port = os.getenv("POTPIE_PORT") or os.getenv("POTPIE_API_PORT") or "8001"
    scheme = os.getenv("POTPIE_SCHEME") or "http"
    return f"{scheme}://{host}:{port}"


def _local_client_or_exit(verbose: bool) -> PotpieLocalApiClient:
    try:
        return PotpieLocalApiClient(
            _resolve_local_api_base_url(),
            bearer_token=os.getenv("POTPIE_BEARER_TOKEN"),
        )
    except ValueError as exc:
        emit_error("Potpie local API not configured", str(exc), verbose=verbose)
        raise typer.Exit(code=1) from exc


def _format_local_api_error(exc: PotpieLocalApiError) -> str:
    d = exc.detail
    if isinstance(d, dict):
        inner = d.get("detail")
        if isinstance(inner, dict):
            return json.dumps(inner)
        if isinstance(inner, str):
            return inner
        return json.dumps(d)
    return str(d)


def _find_potpie_repo_root(start: Path | None = None) -> Path | None:
    cur = (start or Path.cwd()).resolve()
    for _ in range(24):
        if (cur / "pyproject.toml").is_file() and (cur / "scripts").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _local_repo_root_or_exit(verbose: bool) -> Path:
    root = _find_potpie_repo_root()
    if root is not None:
        return root
    emit_error(
        "Potpie checkout not found",
        "Run this command inside a Potpie repository checkout.",
        verbose=verbose,
    )
    raise typer.Exit(code=1)


def _powershell_executable() -> str:
    return shutil.which("powershell") or "powershell.exe"


def _start_command(repo_root: Path) -> list[str]:
    if os.name == "nt":
        return [
            _powershell_executable(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(repo_root / "scripts" / "start.ps1"),
        ]
    return ["bash", str(repo_root / "scripts" / "start.sh")]


def _stop_command(repo_root: Path) -> list[str]:
    if os.name == "nt":
        return [
            _powershell_executable(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(repo_root / "scripts" / "stop.ps1"),
        ]
    return ["bash", str(repo_root / "scripts" / "stop.sh")]


def _run_cli_subprocess_or_exit(
    command: list[str], *, cwd: Path, verbose: bool
) -> None:
    try:
        subprocess.run(command, cwd=str(cwd), check=True)
    except FileNotFoundError as exc:
        emit_error(
            "Required command not found",
            f"Could not run {command[0]!r}. Install the missing dependency and try again.",
            verbose=verbose,
        )
        raise typer.Exit(code=1) from exc
    except subprocess.CalledProcessError as exc:
        emit_error(
            "Command failed",
            f"{command[0]!r} exited with status {exc.returncode}.",
            verbose=verbose,
        )
        raise typer.Exit(code=exc.returncode or 1) from exc


def _launch_background_process_or_exit(
    command: list[str], *, cwd: Path, log_path: Path, verbose: bool
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    creationflags = 0
    if os.name == "nt":
        creationflags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    try:
        with log_path.open("ab") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                creationflags=creationflags,
            )
        return process.pid
    except FileNotFoundError as exc:
        emit_error(
            "Required command not found",
            f"Could not run {command[0]!r}. Install the missing dependency and try again.",
            verbose=verbose,
        )
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        emit_error("Could not start Potpie", str(exc), verbose=verbose)
        raise typer.Exit(code=1) from exc


def _wait_for_local_health(client: PotpieLocalApiClient, *, timeout_seconds: float) -> bool:
    started = time.monotonic()
    while (time.monotonic() - started) < timeout_seconds:
        try:
            code, _payload = client.get_health()
        except OSError:
            code = 0
        if code == 200:
            return True
        time.sleep(2.0)
    return False


def _git_or_exit(verbose: bool) -> str:
    git = shutil.which("git")
    if git:
        return git
    emit_error("Git is required", "Install Git and try again.", verbose=verbose)
    raise typer.Exit(code=1)


def _git_capture(repo_path: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_git_or_exit(False), "-C", str(repo_path), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _resolve_repo_path_or_exit(raw_path: str, verbose: bool) -> Path:
    repo_path = Path(raw_path).expanduser().resolve()
    if not repo_path.exists():
        emit_error("Invalid repo path", f"{repo_path} does not exist.", verbose=verbose)
        raise typer.Exit(code=1)
    if not repo_path.is_dir():
        emit_error(
            "Invalid repo path", f"{repo_path} is not a directory.", verbose=verbose
        )
        raise typer.Exit(code=1)
    completed = _git_capture(repo_path, ["rev-parse", "--is-inside-work-tree"])
    if completed.returncode != 0 or completed.stdout.strip().lower() != "true":
        emit_error(
            "Invalid repository",
            f"{repo_path} is not inside a git working tree.",
            verbose=verbose,
        )
        raise typer.Exit(code=1)
    return repo_path


def _resolve_branch_or_exit(
    repo_path: Path, branch: str | None, verbose: bool
) -> str | None:
    if branch and branch.strip():
        wanted = branch.strip()
        completed = _git_capture(repo_path, ["rev-parse", "--verify", wanted])
        if completed.returncode != 0:
            emit_error(
                "Invalid branch",
                f"Could not resolve {wanted!r} in {repo_path}.",
                verbose=verbose,
            )
            raise typer.Exit(code=1)
        return wanted

    current = _git_capture(repo_path, ["branch", "--show-current"]).stdout.strip()
    if current:
        return current

    fallback = _git_capture(repo_path, ["rev-parse", "--abbrev-ref", "HEAD"])
    current = fallback.stdout.strip()
    if current and current != "HEAD":
        return current
    return None


def _wait_for_project_ready(
    client: PotpieLocalApiClient,
    project_id: str,
    *,
    poll_interval: float,
    timeout_seconds: float,
    show_progress: bool,
    verbose: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    last_status: str | None = None
    while True:
        try:
            payload = client.get_parsing_status(project_id)
        except PotpieLocalApiError as exc:
            emit_error(
                "Could not fetch parsing status",
                _format_local_api_error(exc),
                verbose=verbose,
            )
            raise typer.Exit(code=1) from exc
        status = str(payload.get("status") or "").strip().lower()
        latest = bool(payload.get("latest"))
        if show_progress and status and status != last_status:
            print_plain_line(f"parse status: {status}", as_json=False)
            last_status = status
        if status == "ready" and latest:
            return payload
        if status == "error":
            emit_error(
                "Parsing failed",
                f"Project {project_id} reached status=error.",
                verbose=verbose,
            )
            raise typer.Exit(code=1)
        if timeout_seconds > 0 and (time.monotonic() - started) >= timeout_seconds:
            emit_error(
                "Timed out waiting for parse",
                f"Project {project_id} did not become ready within {int(timeout_seconds)}s.",
                verbose=verbose,
            )
            raise typer.Exit(code=1)
        time.sleep(max(poll_interval, 0.1))


def _normalize_agent_ref(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _resolve_agent_id(rows: list[dict[str, Any]], wanted: str) -> str | None:
    target = wanted.strip().lower()
    normalized_target = _normalize_agent_ref(wanted)
    for row in rows:
        row_id = str(row.get("id") or "").strip()
        row_name = str(row.get("name") or "").strip()
        if row_id.lower() == target or row_name.lower() == target:
            return row_id
        if _normalize_agent_ref(row_id) == normalized_target:
            return row_id
        if row_name and _normalize_agent_ref(row_name) == normalized_target:
            return row_id
    return None


def _ingest_result_from_http(status_code: int, data: dict[str, Any]) -> dict[str, Any]:
    if status_code == 202:
        return {
            "status": "queued",
            "episode_uuid": data.get("episode_uuid"),
            "event_id": data.get("event_id"),
            "job_id": data.get("job_id"),
            "downgrades": list(data.get("downgrades") or []),
        }
    if "event_id" in data or "job_id" in data:
        return {
            "status": data.get("status") or "applied",
            "episode_uuid": data.get("episode_uuid"),
            "event_id": data.get("event_id"),
            "job_id": data.get("job_id"),
            "downgrades": list(data.get("downgrades") or []),
            "errors": list(data.get("errors") or []),
        }
    return {
        "status": "applied",
        "episode_uuid": data.get("episode_uuid"),
        "event_id": data.get("event_id"),
        "job_id": data.get("job_id"),
        "downgrades": list(data.get("downgrades") or []),
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
    source: str | None = typer.Option(
        None,
        "--source",
        "-s",
        help="Default source label for ingest and optional search filter (e.g. cli, mcp). "
        "May appear before the subcommand, e.g. `potpie --source cli search …`.",
    ),
) -> None:
    if not _version:
        _cli_state["json"] = json_output
        _cli_state["verbose"] = verbose
        _cli_state["source"] = source.strip() if source and source.strip() else None
        configure_error_output(as_json=json_output)
        load_cli_env()
        configure_cli_logging(verbose)


@app.command("login")
def login_cmd(
    token: str = typer.Argument(..., help="Potpie API key (from the app)"),
    url: str | None = typer.Option(
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
    db = _database_url_env_set()
    gh = bool(os.getenv("CONTEXT_ENGINE_GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN"))

    maps_set = bool(
        os.getenv("CONTEXT_ENGINE_REPO_TO_POT") or os.getenv("CONTEXT_ENGINE_POTS")
    )

    health_ok: bool | None = None
    health_msg: str | None = None
    auth_ok: bool | None = None
    auth_msg: str | None = None
    try:
        c = PotpieContextApiClient(
            resolve_potpie_api_base_url(),
            resolve_potpie_api_key(),
            timeout=15.0,
            client_surface="cli",
            client_name=_cli_client_name(),
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


@app.command("start")
def start_cmd() -> None:
    """Start the local Potpie development stack in the background."""
    j, v = _flags()
    repo_root = _local_repo_root_or_exit(v)
    command = _start_command(repo_root)
    log_path = repo_root / ".potpie" / "cli" / "start.log"
    pid = _launch_background_process_or_exit(
        command, cwd=repo_root, log_path=log_path, verbose=v
    )
    healthy = _wait_for_local_health(_local_client_or_exit(v), timeout_seconds=60.0)
    payload = {
        "ok": healthy,
        "pid": pid,
        "cwd": str(repo_root),
        "log_path": str(log_path),
        "command": command,
    }
    if j:
        print_json_blob(payload, as_json=True)
        if healthy:
            return
        raise typer.Exit(code=1)
    if healthy:
        print_plain_line("Potpie start requested.", as_json=False)
        print_plain_line(f"log: {log_path}", as_json=False)
        print_plain_line(f"health: {_resolve_local_api_base_url()}/health", as_json=False)
        return
    emit_error(
        "Potpie did not become healthy in time",
        f"Check {log_path} for startup logs.",
        verbose=v,
    )
    raise typer.Exit(code=1)


@app.command("stop")
def stop_cmd() -> None:
    """Stop the local Potpie development stack using the repository scripts."""
    j, v = _flags()
    repo_root = _local_repo_root_or_exit(v)
    command = _stop_command(repo_root)
    _run_cli_subprocess_or_exit(command, cwd=repo_root, verbose=v)
    if j:
        print_json_blob({"ok": True, "cwd": str(repo_root)}, as_json=True)


@app.command("parse")
def parse_cmd(
    repo_path: str = typer.Argument(..., help="Local git repository path to parse."),
    branch: str | None = typer.Option(
        None,
        "--branch",
        help="Branch to parse. Defaults to the current checked-out branch when available.",
    ),
    poll_interval: float = typer.Option(
        2.0,
        "--poll-interval",
        min=0.1,
        help="Polling interval in seconds while waiting for parse completion.",
    ),
    timeout_seconds: float = typer.Option(
        1800.0,
        "--timeout",
        min=1.0,
        help="Max seconds to wait for the project to become ready.",
    ),
) -> None:
    """Parse a local repository, then block until the project is ready."""
    j, v = _flags()
    client = _local_client_or_exit(v)
    resolved_repo = _resolve_repo_path_or_exit(repo_path, v)
    resolved_branch = _resolve_branch_or_exit(resolved_repo, branch, v)
    try:
        payload = client.parse_directory(
            repo_path=str(resolved_repo),
            branch_name=resolved_branch,
            repo_name=resolved_repo.name,
        )
    except PotpieLocalApiError as exc:
        emit_error("Parse failed", _format_local_api_error(exc), verbose=v)
        raise typer.Exit(code=1) from exc

    project_id = str(payload.get("project_id") or "").strip()
    if not project_id:
        emit_error(
            "Parse failed",
            "Server response did not include a project_id.",
            verbose=v,
        )
        raise typer.Exit(code=1)

    if not j:
        branch_label = resolved_branch or "current checkout"
        print_plain_line(
            f"Submitted parse for {resolved_repo} (branch {branch_label}).",
            as_json=False,
        )
        print_plain_line(f"project_id: {project_id}", as_json=False)

    status_payload = _wait_for_project_ready(
        client,
        project_id,
        poll_interval=poll_interval,
        timeout_seconds=timeout_seconds,
        show_progress=not j,
        verbose=v,
    )
    final_payload = {
        "project_id": project_id,
        "repo_path": str(resolved_repo),
        "branch": resolved_branch,
        **status_payload,
    }
    if j:
        print_json_blob(final_payload, as_json=True)
        return
    print_plain_line(f"Parse ready: {project_id}", as_json=False)


@app.command("chat")
def chat_cmd(
    project_id: str = typer.Argument(..., help="Parsed Potpie project id."),
    agent: str = typer.Option(
        ...,
        "--agent",
        help="Agent id or display name (for example: codebase_qna_agent).",
    ),
    branch: str | None = typer.Option(
        None,
        "--branch",
        help="Optional branch to validate against the selected project.",
    ),
    conversation_id: str | None = typer.Option(
        None,
        "--conversation-id",
        help="Resume an existing conversation instead of creating a new one.",
    ),
) -> None:
    """Open an interactive local Potpie chat session for one parsed project."""
    j, v = _flags()
    client = _local_client_or_exit(v)
    try:
        projects = client.list_projects()
    except PotpieLocalApiError as exc:
        emit_error("Could not list projects", _format_local_api_error(exc), verbose=v)
        raise typer.Exit(code=1) from exc

    project_row = next(
        (row for row in projects if str(row.get("id") or "") == project_id), None
    )
    if project_row is None:
        emit_error(
            "Unknown project",
            f"{project_id!r} was not found in the local Potpie project list.",
            verbose=v,
        )
        raise typer.Exit(code=1)
    if branch and str(project_row.get("branch_name") or "") != branch:
        emit_error(
            "Project branch mismatch",
            f"Project {project_id} belongs to branch {project_row.get('branch_name')!r}, not {branch!r}.",
            verbose=v,
        )
        raise typer.Exit(code=1)

    try:
        status_payload = client.get_parsing_status(project_id)
    except PotpieLocalApiError as exc:
        emit_error(
            "Could not fetch parsing status",
            _format_local_api_error(exc),
            verbose=v,
        )
        raise typer.Exit(code=1) from exc
    if str(status_payload.get("status") or "").lower() != "ready" or not bool(
        status_payload.get("latest")
    ):
        emit_error(
            "Project not ready",
            f"Project {project_id} is not ready for chat yet.",
            verbose=v,
        )
        raise typer.Exit(code=1)

    try:
        agents = client.list_available_agents()
    except PotpieLocalApiError as exc:
        emit_error("Could not list agents", _format_local_api_error(exc), verbose=v)
        raise typer.Exit(code=1) from exc
    agent_id = _resolve_agent_id(agents, agent)
    if agent_id is None:
        emit_error(
            "Unknown agent",
            f"{agent!r} is not an available agent.",
            verbose=v,
        )
        raise typer.Exit(code=1)

    active_conversation_id = conversation_id
    if not active_conversation_id:
        try:
            created = client.create_conversation(
                project_id=project_id,
                agent_id=agent_id,
                title=f"CLI Chat {project_id}",
            )
        except PotpieLocalApiError as exc:
            emit_error(
                "Could not create conversation",
                _format_local_api_error(exc),
                verbose=v,
            )
            raise typer.Exit(code=1) from exc
        active_conversation_id = str(created.get("conversation_id") or "").strip()
        if not active_conversation_id:
            emit_error(
                "Could not create conversation",
                "Server response did not include a conversation_id.",
                verbose=v,
            )
            raise typer.Exit(code=1)

    if j:
        print_json_blob(
            {
                "conversation_id": active_conversation_id,
                "project_id": project_id,
                "agent_id": agent_id,
            },
            as_json=True,
        )
    else:
        print_plain_line(
            f"conversation_id: {active_conversation_id}", as_json=False
        )
        print_plain_line("Type /exit to end the chat session.", as_json=False)

    while True:
        try:
            prompt = "" if j else "you> "
            user_input = input(prompt)
        except EOFError:
            break
        except KeyboardInterrupt:
            print_plain_line("", as_json=False)
            break
        message = user_input.strip()
        if not message:
            continue
        if message.lower() in {"/exit", "exit", "quit"}:
            break
        try:
            response = client.send_message(active_conversation_id, message)
        except PotpieLocalApiError as exc:
            emit_error("Chat failed", _format_local_api_error(exc), verbose=v)
            raise typer.Exit(code=1) from exc
        if j:
            print_json_blob(
                {
                    "conversation_id": active_conversation_id,
                    "reply": response,
                },
                as_json=True,
            )
            continue
        reply = str(response.get("message") or "").strip() or "(no response)"
        print_plain_line(f"potpie> {reply}", as_json=False)


@app.command("init-agent")
def init_agent_cmd(
    agent: str | None = typer.Argument(
        None,
        help=(
            "Agent type: 'claude' installs CLAUDE.md section + .claude/commands; "
            "'codex' or omitted installs AGENTS.md + .agents/skills bundle. "
            f"Choices: {', '.join(AGENT_TYPES)}"
        ),
    ),
    path: str = typer.Argument(
        ".",
        help="Repository path or a subdirectory inside it (default: current directory)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing files when contents differ (CLAUDE.md: replace Potpie context section).",
    ),
) -> None:
    """Install agent instructions and Potpie context skills into a repository.

    Default / codex: installs AGENTS.md and .agents/skills bundle.
    Claude: merges a Potpie context section into CLAUDE.md and installs .claude/commands.

    Examples:
      potpie init-agent              # codex / default bundle
      potpie init-agent claude       # CLAUDE.md + slash commands
      potpie init-agent codex .      # explicit codex bundle
      potpie init-agent claude /path/to/repo
    """
    j, v = _flags()

    # If agent looks like a path (contains / or . and is not a known type), treat it as path
    resolved_agent = "default"
    resolved_path = path
    if agent is not None:
        normalized = agent.strip().lower()
        if normalized in AGENT_TYPES:
            resolved_agent = normalized
        else:
            # Treat it as a path — user omitted the agent type
            resolved_path = agent
            resolved_agent = "default"

    try:
        result = install_agent_bundle(resolved_path, agent=resolved_agent, force=force)
    except ValueError as exc:
        emit_error("Invalid install path or agent type", str(exc), verbose=v)
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
        resolved, rerr = _resolve_cli_pot_ref_with_api(explicit)
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


def _resolve_cli_pot_ref_with_api(ref: str) -> tuple[str | None, str]:
    """Resolve UUID/local alias first, then an accessible server-side slug."""
    resolved, rerr = resolve_cli_pot_ref(ref)
    if resolved and not rerr:
        return resolved, ""
    slug = (ref or "").strip().lower()
    if not slug:
        return None, rerr
    try:
        client = PotpieContextApiClient(
            resolve_potpie_api_base_url(),
            resolve_potpie_api_key(),
            timeout=30.0,
            client_surface="cli",
            client_name=_cli_client_name(),
        )
        row = client.find_context_pot_by_slug(slug)
    except (ValueError, OSError, PotpieContextApiError):
        return None, rerr
    if not row:
        return None, rerr
    pid = row.get("id")
    if not pid:
        return None, f"Pot slug {slug!r} resolved to a row without an id."
    try:
        uuid.UUID(str(pid))
    except ValueError:
        return None, f"Pot slug {slug!r} resolved to invalid id {pid!r}."
    register_pot_alias(slug, str(pid))
    return str(pid), ""


@pot_app.command("use")
def pot_use(
    pot_id: str = typer.Argument(
        ...,
        help="Pot UUID, slug, or a name from `pot alias` (see `pot pots`).",
    ),
) -> None:
    """Remember the active pot for CLI commands (stored in credentials.json).

    This is the global default for `search` and `ingest` when no pot UUID is passed.
    It takes precedence over CONTEXT_ENGINE_* repo maps. Use `pot unset` to fall back to maps + git.
    """
    j, v = _flags()
    resolved, rerr = _resolve_cli_pot_ref_with_api(pot_id)
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
            "No context pots yet. Run `potpie pot create my-scope` to create one on the server.",
            as_json=False,
        )
        return
    for p in rows:
        pid = p.get("id", "")
        slug = p.get("slug", "")
        repo = p.get("primary_repo_name", "")
        print_plain_line(f"{slug}\t{pid}\t{repo}", as_json=False)
    print_plain_line(
        "[dim]Use: potpie pot use <slug-or-id>  or  pot alias <name> <id>[/dim]",
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
            f"{pot_uuid!r} is not a UUID. Run `potpie pot pots` to list pots.",
            verbose=v,
        )
        raise typer.Exit(code=1)
    resolved, rerr = _resolve_cli_pot_ref_with_api(pot_uuid)
    if rerr or not resolved:
        emit_error("Invalid pot id", rerr or pot_uuid, verbose=v)
        raise typer.Exit(code=1)
    register_pot_alias(name, resolved)
    print_plain_line(
        f"Alias {name!r} -> {resolved}",
        as_json=j,
        json_payload={"ok": True, "name": name, "pot_id": resolved},
    )


@pot_app.command("slug-available")
def pot_slug_available_cmd(
    slug: str = typer.Argument(..., help="Candidate globally unique pot slug."),
) -> None:
    """Check whether a pot slug is available on the configured Potpie server."""
    j, v = _flags()
    client = _cli_client_or_exit(v)
    try:
        out = client.get_context_pot_slug_availability(slug)
    except PotpieContextApiError as exc:
        emit_error("Could not check slug", _format_api_error(exc), verbose=v)
        raise typer.Exit(code=1) from exc
    if j:
        print_json_blob(out, as_json=True)
        return
    available = bool(out.get("available"))
    normalized = out.get("slug") or slug
    print_plain_line(
        f"{normalized}: {'available' if available else 'taken'}",
        as_json=False,
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
    slug: str = typer.Argument(
        ...,
        help="Globally unique pot slug; registers it as a local alias to the new server pot id.",
    ),
) -> None:
    """Create a user-owned context pot on Potpie (POST /api/v2/context/pots) and save the slug locally."""
    j, v = _flags()
    client = _cli_client_or_exit(v)
    try:
        availability = client.get_context_pot_slug_availability(slug)
        if not availability.get("available"):
            emit_error(
                "Slug is already taken",
                str(availability.get("slug") or slug),
                verbose=v,
            )
            raise typer.Exit(code=1)
        row = client.create_context_pot(slug=str(availability.get("slug") or slug))
    except PotpieContextApiError as exc:
        emit_error("Could not create context pot", _format_api_error(exc), verbose=v)
        raise typer.Exit(code=1) from exc
    pid = row.get("id")
    if not pid:
        emit_error("Unexpected response", str(row), verbose=v)
        raise typer.Exit(code=1)
    row_slug = str(row.get("slug") or slug)
    register_pot_alias(row_slug, str(pid))
    data = {
        "id": pid,
        "slug": row_slug,
        "primary_repo_name": row.get("primary_repo_name"),
        "alias": row_slug,
        "message": f"Run `potpie pot use {row_slug}` or `potpie pot use {pid}`.",
    }
    if j:
        print_json_blob(data, as_json=True)
    else:
        print_plain_line(
            f"Created context pot {row_slug} ({pid}). Try: potpie pot use {row_slug}",
            as_json=False,
        )


@pot_repo_app.command("list")
def pot_repo_list_cmd(
    pot_opt: str | None = typer.Option(
        None,
        "--pot",
        help="Pot UUID or alias (default: active pot / git cwd, same as ingest).",
    ),
    cwd: str | None = typer.Option(
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
            f"No repositories for pot {pid}. Try `potpie pot repo add owner/repo`.",
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
    pot_opt: str | None = typer.Option(
        None,
        "--pot",
        help="Pot UUID or alias (default: active pot / git cwd).",
    ),
    cwd: str | None = typer.Option(
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
    pot_id: str | None = typer.Argument(
        None,
        help=(
            "Pot id to reset. Strongly recommended to pass explicitly; omit only "
            "when you are inside the pot's git working tree and have passed --yes."
        ),
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
    assume_yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the destructive-action confirmation prompt.",
    ),
) -> None:
    """[DESTRUCTIVE] Delete ALL context-graph data for this pot.

    Operator/admin command. Calls Potpie POST ``/api/v2/context/reset`` and
    triggers a ``context_engine.operator_audit`` log entry on the server.
    Prompts for confirmation unless ``--yes`` is passed.
    """
    j, v = _flags()
    cwd_resolved = str(Path(cwd).resolve())
    resolved = _pot_id_or_git(pot_id, cwd=cwd_resolved)
    pot_was_inferred = pot_id is None
    if not assume_yes:
        scope_note = (
            f"pot {resolved} (inferred from {cwd_resolved})"
            if pot_was_inferred
            else f"pot {resolved}"
        )
        try:
            confirmed = typer.confirm(
                f"This will DELETE all context-graph data for {scope_note}. "
                "This cannot be undone. Continue?",
                default=False,
            )
        except typer.Abort:
            confirmed = False
        if not confirmed:
            emit_error(
                "Hard reset aborted",
                "Confirmation declined; re-run with --yes to skip the prompt.",
                verbose=v,
            )
            raise typer.Exit(code=1)
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


def _event_terminal(status: Any) -> bool:
    return str(status or "").strip().lower() in {"done", "error"}


def _event_lifecycle(event: dict[str, Any]) -> str:
    return str(event.get("lifecycle_status") or event.get("status") or "").strip()


def _print_event(event: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print_json_blob(event, as_json=True)
        return
    status = _event_lifecycle(event) or "(unknown)"
    stage = event.get("stage") or event.get("raw_status") or ""
    print_plain_line(
        f"{event.get('event_id') or event.get('id')}\t{status}\t{stage}",
        as_json=False,
    )
    for key in (
        "pot_id",
        "ingestion_kind",
        "source_channel",
        "repo_name",
        "job_id",
        "status",
        "lifecycle_status",
        "submitted_at",
        "received_at",
        "started_at",
        "completed_at",
        "error",
    ):
        value = event.get(key)
        if value not in (None, "", []):
            print_plain_line(f"{key}: {value}", as_json=False)
    steps = event.get("episode_steps")
    if isinstance(steps, list) and steps:
        print_plain_line("steps:", as_json=False)
        for step in steps:
            if isinstance(step, dict):
                print_plain_line(
                    "  "
                    f"{step.get('sequence')}: {step.get('step_kind')} "
                    f"{step.get('status')} attempts={step.get('attempt_count')} "
                    f"error={step.get('error') or ''}",
                    as_json=False,
                )
    runs = event.get("reconciliation_runs")
    if isinstance(runs, list) and runs:
        print_plain_line("reconciliation runs:", as_json=False)
        for run in runs:
            if isinstance(run, dict):
                work_events = run.get("work_events")
                count = len(work_events) if isinstance(work_events, list) else 0
                print_plain_line(
                    "  "
                    f"attempt={run.get('attempt_number')} status={run.get('status')} "
                    f"agent={run.get('agent_name') or ''} work_events={count} "
                    f"error={run.get('error') or ''}",
                    as_json=False,
                )


@event_app.command("show")
def event_show_cmd(
    event_id: str = typer.Argument(
        ..., help="Ingestion event id from `potpie ingest`."
    ),
) -> None:
    """Fetch one persisted ingestion event."""
    j, v = _flags()
    client = _cli_client_or_exit(v)
    try:
        event = client.get_event(event_id)
    except PotpieContextApiError as exc:
        emit_error("Could not fetch event", _format_api_error(exc), verbose=v)
        raise typer.Exit(code=1) from exc
    _print_event(event, as_json=j)


@event_app.command("list")
def event_list_cmd(
    pot_id: str | None = typer.Argument(
        None,
        help="Pot id / slug / alias. Omit to infer from `potpie pot use` or git.",
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Max events to show (1-200)."),
    status: str | None = typer.Option(
        None,
        "--status",
        help="Filter lifecycle: queued, processing, done, or error.",
    ),
    ingestion_kind: str | None = typer.Option(
        None,
        "--kind",
        help="Filter ingestion kind, e.g. raw_episode or agent_reconciliation.",
    ),
    cwd: str = typer.Option(
        ".",
        "--cwd",
        help="Git working tree used to infer pot when POT_ID is omitted.",
    ),
) -> None:
    """List recent ingestion events for a pot."""
    j, v = _flags()
    resolved = _pot_id_or_git(pot_id, cwd=str(Path(cwd).resolve()))
    client = _cli_client_or_exit(v)
    try:
        page = client.list_events(
            resolved,
            limit=limit,
            status=status,
            ingestion_kind=ingestion_kind,
        )
    except PotpieContextApiError as exc:
        emit_error("Could not list events", _format_api_error(exc), verbose=v)
        raise typer.Exit(code=1) from exc
    if j:
        print_json_blob(page, as_json=True)
        return
    items = page.get("items")
    if not isinstance(items, list) or not items:
        print_plain_line(f"No events for pot {resolved}.", as_json=False)
        return
    print_plain_line(
        "event_id\tstatus\tlifecycle\tkind\tsource_channel\tsource_system\tsubmitted_at",
        as_json=False,
    )
    for event in items:
        if not isinstance(event, dict):
            continue
        print_plain_line(
            f"{event.get('event_id') or event.get('id')}\t{event.get('status')}\t"
            f"{event.get('lifecycle_status')}\t{event.get('ingestion_kind')}\t"
            f"{event.get('source_channel') or ''}\t{event.get('source_system') or ''}\t"
            f"{event.get('submitted_at') or event.get('received_at')}",
            as_json=False,
        )


@event_app.command("wait")
def event_wait_cmd(
    event_id: str = typer.Argument(
        ..., help="Ingestion event id from `potpie ingest`."
    ),
    timeout: float = typer.Option(
        60.0,
        "--timeout",
        help="Maximum seconds to wait for done/error.",
    ),
    interval: float = typer.Option(
        2.0,
        "--interval",
        help="Polling interval in seconds.",
    ),
) -> None:
    """Poll one event until it reaches done or error."""
    j, v = _flags()
    client = _cli_client_or_exit(v)
    deadline = time.monotonic() + max(timeout, 0.0)
    last: dict[str, Any] = {}
    while True:
        try:
            last = client.get_event(event_id)
        except PotpieContextApiError as exc:
            emit_error("Could not fetch event", _format_api_error(exc), verbose=v)
            raise typer.Exit(code=1) from exc
        if _event_terminal(_event_lifecycle(last)):
            _print_event(last, as_json=j)
            if _event_lifecycle(last).lower() == "error":
                raise typer.Exit(code=1)
            return
        if time.monotonic() >= deadline:
            if j:
                print_json_blob(
                    {
                        "ok": False,
                        "timeout": True,
                        "event": last,
                    },
                    as_json=True,
                )
            else:
                status = _event_lifecycle(last) or "(unknown)"
                stage = last.get("stage") or last.get("raw_status") or ""
                print_plain_line(
                    f"Timed out waiting for {event_id}; current status={status} stage={stage}. "
                    f"Run `potpie event show {event_id}` to check again.",
                    as_json=False,
                )
            raise typer.Exit(code=1)
        if not j:
            status = _event_lifecycle(last) or "(unknown)"
            stage = last.get("stage") or last.get("raw_status") or ""
            print_plain_line(f"event {event_id}: {status} {stage}", as_json=False)
        time.sleep(max(interval, 0.1))


app.add_typer(event_app, name="event")


@conflict_app.command("list")
def conflict_list_cmd(
    cwd: str = typer.Option(
        ".",
        "--cwd",
        help="Git working tree used when inferring pot from remote / env",
    ),
) -> None:
    """List open predicate-family conflicts for the active pot."""
    load_cli_env()
    j, v = _flags()
    cwd_resolved = str(Path(cwd).resolve())
    pot_id = _pot_id_or_git(None, cwd=cwd_resolved)
    client = _cli_client_or_exit(v)
    try:
        out = client.conflicts_list(pot_id)
    except PotpieContextApiError as exc:
        emit_error("Could not list conflicts", _format_api_error(exc), verbose=v)
        raise typer.Exit(code=1) from exc
    print_json_blob(out, as_json=j)


@conflict_app.command("resolve")
def conflict_resolve_cmd(
    issue_uuid: str = typer.Argument(..., help="QualityIssue uuid"),
    action: str = typer.Option(
        "supersede_older",
        "--action",
        "-a",
        help="Resolution strategy (supersede_older stamps invalid_at on the older edge)",
    ),
    cwd: str = typer.Option(
        ".",
        "--cwd",
        help="Git working tree used when inferring pot from remote / env",
    ),
) -> None:
    """Resolve one open conflict (default: supersede the older episodic edge)."""
    load_cli_env()
    j, v = _flags()
    cwd_resolved = str(Path(cwd).resolve())
    pot_id = _pot_id_or_git(None, cwd=cwd_resolved)
    client = _cli_client_or_exit(v)
    try:
        out = client.conflicts_resolve(pot_id, issue_uuid, action=action)
    except PotpieContextApiError as exc:
        emit_error("Could not resolve conflict", _format_api_error(exc), verbose=v)
        raise typer.Exit(code=1) from exc
    print_json_blob(out, as_json=j)


app.add_typer(conflict_app, name="conflict")


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
        "hint": "Map this repo to a pot with CONTEXT_ENGINE_REPO_TO_POT or `potpie pot use`.",
    }
    print_json_blob(data, as_json=j)


@app.command("search")
def search(
    first: str = typer.Argument(
        ...,
        help="Either pot UUID + query, or only the query when pot is inferred (pot use / env / git)",
    ),
    second: str | None = typer.Argument(
        None,
        help="Natural-language query when the first argument is the pot UUID",
    ),
    limit: int = typer.Option(8, "--limit", "-n", help="Max results (1–50)"),
    node_labels: str | None = typer.Option(
        None,
        "--node-labels",
        help="Optional comma-separated Graphiti labels, e.g. PullRequest,Decision",
    ),
    repo_name: str | None = typer.Option(
        None,
        "--repo",
        "-r",
        help="Optional owner/repo to narrow results inside a multi-repo pot",
    ),
    source_description: str | None = typer.Option(
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
        help="Plain output: also show created_at (valid/invalid are compact by default; JSON includes temporal_flag when present).",
    ),
    as_of: str | None = typer.Option(
        None,
        "--as-of",
        help="ISO 8601 instant; restrict results to edges valid at that time (valid_at/invalid_at window).",
    ),
    episode_uuid: str | None = typer.Option(
        None,
        "--episode",
        "-e",
        help="Only facts linked to this ingested episode UUID (server-side filter).",
    ),
    no_provenance: bool = typer.Option(
        False,
        "--no-provenance",
        help="Hide source / reference_time / episode line in plain (non-JSON) output.",
    ),
    cwd: str = typer.Option(
        ".",
        "--cwd",
        help="Git working tree used to read `origin` when inferring pot (ignored when pot UUID is explicit)",
    ),
) -> None:
    """Semantic search through the unified context graph query endpoint."""
    j, v = _flags()
    cwd_resolved = str(Path(cwd).resolve())
    if second is None:
        query = first
        pot_id = _pot_id_or_git(None, cwd=cwd_resolved)
    else:
        pot_id = _pot_id_or_git(first, cwd=cwd_resolved)
        query = second
    labels: list[str] = []
    if node_labels:
        labels = [x.strip() for x in node_labels.split(",") if x.strip()]
    as_of_dt: datetime | None = None
    if as_of:
        try:
            as_of_dt = _parse_iso_datetime(as_of)
        except ValueError as e:
            emit_error("Invalid --as-of", str(e), verbose=v)
            raise typer.Exit(code=1) from e
    body: dict[str, Any] = {
        "pot_id": pot_id,
        "query": query,
        "goal": "retrieve",
        "strategy": "semantic",
        "limit": limit,
        "node_labels": labels,
        "scope": {"repo_name": repo_name} if repo_name else {},
        "source_descriptions": [_resolved_source_label(source_description)]
        if _resolved_source_label(source_description)
        else [],
        "episode_uuids": [episode_uuid.strip()]
        if episode_uuid and episode_uuid.strip()
        else [],
        "include_invalidated": include_invalidated,
        "as_of": as_of_dt,
    }
    client = _cli_client_or_exit(v)
    try:
        response = client.context_graph_query(body)
    except PotpieContextApiError as exc:
        emit_error(
            "Search failed",
            _format_api_error(exc),
            verbose=v,
            exc=exc if v else None,
        )
        raise typer.Exit(code=1) from exc
    rows = response.get("result")
    if not isinstance(rows, list):
        emit_error(
            "Search failed", f"Unexpected response: {type(rows).__name__}", verbose=v
        )
        raise typer.Exit(code=1)
    print_search_results(
        rows,
        as_json=j,
        with_temporal=with_temporal,
        show_provenance=not no_provenance,
    )


def _resolve_scope_body(
    *,
    repo_name: str | None = None,
    file_path: str | None = None,
    function_name: str | None = None,
    pr_number: int | None = None,
    services: list[str] | None = None,
    features: list[str] | None = None,
    environment: str | None = None,
    ticket_ids: list[str] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if repo_name:
        out["repo_name"] = repo_name
    if file_path:
        out["file_path"] = file_path
    if function_name:
        out["function_name"] = function_name
    if pr_number is not None:
        out["pr_number"] = pr_number
    if services:
        out["services"] = services
    if features:
        out["features"] = features
    if environment:
        out["environment"] = environment
    if ticket_ids:
        out["ticket_ids"] = ticket_ids
    return out


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


@app.command("status")
def status_cmd(
    pot: str | None = typer.Argument(
        None,
        help="Pot UUID or alias; inferred from `pot use` / env / git when omitted.",
    ),
    intent: str | None = typer.Option(
        None,
        "--intent",
        help="Optional task intent; surfaces the recommended context_resolve recipe.",
    ),
    repo_name: str | None = typer.Option(None, "--repo", "-r"),
    file_path: str | None = typer.Option(None, "--file", "-f"),
    pr_number: int | None = typer.Option(None, "--pr"),
    cwd: str = typer.Option(".", "--cwd"),
) -> None:
    """Fetch readiness / capability envelope (POST /status)."""
    j, v = _flags()
    pot_id = _pot_id_or_git(pot, cwd=str(Path(cwd).resolve()))
    body: dict[str, Any] = {
        "pot_id": pot_id,
        "scope": _resolve_scope_body(
            repo_name=repo_name, file_path=file_path, pr_number=pr_number
        ),
    }
    if intent:
        body["intent"] = intent
    client = _cli_client_or_exit(v)
    try:
        response = client.status(body)
    except PotpieContextApiError as exc:
        emit_error(
            "Status failed", _format_api_error(exc), verbose=v, exc=exc if v else None
        )
        raise typer.Exit(code=1) from exc
    print_json_blob(response, as_json=j)


@app.command("resolve")
def resolve_cmd(
    first: str = typer.Argument(
        ...,
        help="Either pot UUID + query, or only the query when pot is inferred (pot use / env / git).",
    ),
    second: str | None = typer.Argument(
        None,
        help="Natural-language query when the first argument is the pot UUID.",
    ),
    repo_name: str | None = typer.Option(None, "--repo", "-r"),
    file_path: str | None = typer.Option(None, "--file", "-f"),
    function_name: str | None = typer.Option(None, "--function"),
    pr_number: int | None = typer.Option(None, "--pr"),
    services: str | None = typer.Option(
        None, "--services", help="Comma-separated service names."
    ),
    features: str | None = typer.Option(
        None, "--features", help="Comma-separated feature names."
    ),
    environment: str | None = typer.Option(None, "--env"),
    ticket_ids: str | None = typer.Option(
        None, "--tickets", help="Comma-separated ticket IDs."
    ),
    intent: str | None = typer.Option(None, "--intent"),
    include: str | None = typer.Option(
        None, "--include", help="Comma-separated recipe include list."
    ),
    limit: int = typer.Option(12, "--limit", "-n"),
    cwd: str = typer.Option(".", "--cwd"),
) -> None:
    """Answer a query with synthesized summary + bundled evidence (goal=answer)."""
    j, v = _flags()
    cwd_resolved = str(Path(cwd).resolve())
    if second is None:
        query = first
        pot_id = _pot_id_or_git(None, cwd=cwd_resolved)
    else:
        pot_id = _pot_id_or_git(first, cwd=cwd_resolved)
        query = second
    body: dict[str, Any] = {
        "pot_id": pot_id,
        "query": query,
        "goal": "answer",
        "strategy": "auto",
        "limit": limit,
        "scope": _resolve_scope_body(
            repo_name=repo_name,
            file_path=file_path,
            function_name=function_name,
            pr_number=pr_number,
            services=_split_csv(services),
            features=_split_csv(features),
            environment=environment,
            ticket_ids=_split_csv(ticket_ids),
        ),
        "include": _split_csv(include),
    }
    if intent:
        body["intent"] = intent
    client = _cli_client_or_exit(v)
    try:
        response = client.context_graph_query(body)
    except PotpieContextApiError as exc:
        emit_error(
            "Resolve failed", _format_api_error(exc), verbose=v, exc=exc if v else None
        )
        raise typer.Exit(code=1) from exc
    if j:
        print_json_blob(response, as_json=True)
        return
    result = response.get("result") or {}
    answer = result.get("answer") or {}
    summary = answer.get("summary") or "(no summary)"
    print_plain_line(summary, as_json=False)


@app.command("overview")
def overview_cmd(
    pot: str | None = typer.Argument(
        None,
        help="Pot UUID or alias; inferred from `pot use` / env / git when omitted.",
    ),
    repo_name: str | None = typer.Option(None, "--repo", "-r"),
    limit: int = typer.Option(12, "--limit", "-n"),
    cwd: str = typer.Option(".", "--cwd"),
) -> None:
    """Fetch graph-wide readiness / activity snapshot (goal=aggregate, include=[graph_overview])."""
    j, v = _flags()
    pot_id = _pot_id_or_git(pot, cwd=str(Path(cwd).resolve()))
    body: dict[str, Any] = {
        "pot_id": pot_id,
        "goal": "aggregate",
        "strategy": "auto",
        "include": ["graph_overview"],
        "limit": limit,
        "scope": _resolve_scope_body(repo_name=repo_name),
    }
    client = _cli_client_or_exit(v)
    try:
        response = client.context_graph_query(body)
    except PotpieContextApiError as exc:
        emit_error(
            "Overview failed", _format_api_error(exc), verbose=v, exc=exc if v else None
        )
        raise typer.Exit(code=1) from exc
    print_json_blob(response, as_json=j)


@app.command("record")
def record_cmd(
    record_type: str = typer.Option(
        ..., "--type", "-t", help="Record type (e.g. decision, fix, incident)."
    ),
    summary: str = typer.Option(
        ..., "--summary", "-s", help="One-line record summary (required)."
    ),
    pot: str | None = typer.Option(
        None, "--pot", help="Pot UUID/alias; inferred when omitted."
    ),
    details: str | None = typer.Option(
        None,
        "--details",
        help="JSON object with structured record details.",
    ),
    source_refs: str | None = typer.Option(
        None, "--source-refs", help="Comma-separated source references."
    ),
    confidence: float = typer.Option(0.7, "--confidence", min=0.0, max=1.0),
    visibility: str = typer.Option("project", "--visibility"),
    idempotency_key: str | None = typer.Option(None, "--idempotency-key"),
    occurred_at: str | None = typer.Option(
        None, "--occurred-at", help="ISO-8601 timestamp; defaults to now."
    ),
    repo_name: str | None = typer.Option(None, "--repo", "-r"),
    file_path: str | None = typer.Option(None, "--file", "-f"),
    pr_number: int | None = typer.Option(None, "--pr"),
    sync: bool = typer.Option(
        False, "--sync", help="Wait for synchronous reconciliation."
    ),
    cwd: str = typer.Option(".", "--cwd"),
) -> None:
    """Record a durable context fact (POST /record)."""
    j, v = _flags()
    pot_id = _pot_id_or_git(pot, cwd=str(Path(cwd).resolve()))
    details_dict: dict[str, Any] = {}
    if details:
        try:
            parsed = json.loads(details)
        except json.JSONDecodeError as exc:
            emit_error("Invalid --details", f"Expected JSON object: {exc}", verbose=v)
            raise typer.Exit(code=1) from exc
        if not isinstance(parsed, dict):
            emit_error("Invalid --details", "Expected a JSON object.", verbose=v)
            raise typer.Exit(code=1)
        details_dict = parsed
    occurred_dt: datetime | None = None
    if occurred_at:
        try:
            occurred_dt = _parse_iso_datetime(occurred_at)
        except ValueError as exc:
            emit_error("Invalid --occurred-at", str(exc), verbose=v)
            raise typer.Exit(code=1) from exc
    body: dict[str, Any] = {
        "pot_id": pot_id,
        "record": {
            "type": record_type,
            "summary": summary,
            "details": details_dict,
            "source_refs": _split_csv(source_refs),
            "confidence": confidence,
            "visibility": visibility,
        },
        "scope": _resolve_scope_body(
            repo_name=repo_name, file_path=file_path, pr_number=pr_number
        ),
    }
    if idempotency_key:
        body["idempotency_key"] = idempotency_key
    if occurred_dt is not None:
        body["occurred_at"] = occurred_dt
    client = _cli_client_or_exit(v)
    try:
        response = client.record(body, sync=sync)
    except PotpieContextApiError as exc:
        emit_error(
            "Record failed", _format_api_error(exc), verbose=v, exc=exc if v else None
        )
        raise typer.Exit(code=1) from exc
    print_json_blob(response, as_json=j)


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
    first: str | None = typer.Argument(
        None,
        help="Episode text, or pot UUID when a second argument is the text",
    ),
    second: str | None = typer.Argument(
        None,
        help="Episode text when the first argument is a pot UUID",
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help="Episode title (default: first line of body, truncated)",
    ),
    episode_body: str | None = typer.Option(
        None,
        "--episode-body",
        "-b",
        help="Episode text (optional if you pass text as a positional argument)",
    ),
    episode_file: Path | None = typer.Option(
        None,
        "--file",
        "-f",
        help="Read episode body from this file (UTF-8). Conflicts with inline body / -b.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    source_description: str | None = typer.Option(
        None,
        "--source",
        "-s",
        help="Short source label (default: cli)",
    ),
    reference_time: str | None = typer.Option(
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
    idempotency_key: str | None = typer.Option(
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

    Quick form: potpie ingest "Your text" (pot from pot use, env maps, or git --cwd).

    With no positional pot id, scope is: global pot use, then env repo maps, then git origin under --cwd.

    Use --file / -f to load the episode body from a UTF-8 file (optionally with an explicit pot UUID only).
    """
    j, v = _flags()
    file_body: str | None = None
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
    except IngestRejectedError as exc:
        pl = dict(exc.body)
        if pl.get("status") is None:
            pl["status"] = "reconciliation_rejected"
        print_ingest_result(pl, as_json=j)
        raise typer.Exit(code=2) from None
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

    if status_code == 409:
        ev = data.get("event_id") if isinstance(data, dict) else None
        emit_error(
            "Duplicate ingest",
            f"event_id={ev!r}" if ev else "Server reported a duplicate ingest.",
            verbose=v,
        )
        raise typer.Exit(code=1)

    out = _ingest_result_from_http(status_code, data if isinstance(data, dict) else {})
    if out.get("errors"):
        out["status"] = "reconciliation_rejected"
        print_ingest_result(out, as_json=j)
        raise typer.Exit(code=2)
    if (
        out["status"] == "applied"
        and not out.get("episode_uuid")
        and not out.get("event_id")
    ):
        emit_error(
            "Ingest failed",
            "Server returned no episode UUID or event_id (check Potpie logs).",
            verbose=v,
        )
        raise typer.Exit(code=1)

    print_ingest_result(out, as_json=j)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

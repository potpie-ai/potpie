"""Local development CLI for Potpie."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import typer
from dotenv import load_dotenv

APP_NAME = "potpie"
DEFAULT_API_BASE_URL = "http://127.0.0.1:8001"
READY_STATUS = "ready"
ERROR_STATUS = "error"
LOCAL_COMMANDS = {"start", "stop", "parse", "chat"}

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = Path(os.getenv("POTPIE_CLI_STATE_DIR", REPO_ROOT / ".potpie"))
PID_FILE = STATE_DIR / "server.pid"
LOG_FILE = STATE_DIR / "server.log"

app = typer.Typer(
    name=APP_NAME,
    no_args_is_help=True,
    help="Run and use a local Potpie server.",
)


class CLIError(Exception):
    """Expected CLI failure with a user-facing message."""


def _load_env() -> None:
    load_dotenv(_repo_root() / ".env", override=False)


def _echo_error(message: str) -> None:
    typer.echo(typer.style(f"Error: {message}", fg=typer.colors.RED), err=True)


def _repo_root() -> Path:
    return Path(os.getenv("POTPIE_REPO_ROOT", str(REPO_ROOT))).resolve()


def _state_dir() -> Path:
    return Path(os.getenv("POTPIE_CLI_STATE_DIR", str(STATE_DIR))).expanduser()


def _pid_file() -> Path:
    return _state_dir() / "server.pid"


def _log_file() -> Path:
    return _state_dir() / "server.log"


def _process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_pid_metadata() -> dict[str, Any] | None:
    try:
        data = json.loads(_pid_file().read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _metadata_pid(metadata: dict[str, Any] | None) -> int | None:
    if not metadata:
        return None
    pid = metadata.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    return pid


def _write_pid(pid: int, command: list[str]) -> None:
    state_dir = _state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "pid": pid,
        "command": command,
        "started_at": datetime.now().isoformat(),
    }
    _pid_file().write_text(json.dumps(metadata, indent=2) + "\n")


def _clear_pid() -> None:
    try:
        _pid_file().unlink()
    except FileNotFoundError:
        pass


def _resolve_api_base_url(api_url: str | None = None) -> str:
    explicit = (api_url or "").strip()
    if explicit:
        return explicit.rstrip("/")

    for key in ("POTPIE_API_URL", "POTPIE_BASE_URL"):
        value = os.getenv(key, "").strip()
        if value:
            return value.rstrip("/")

    port = (os.getenv("POTPIE_PORT") or os.getenv("POTPIE_API_PORT") or "").strip()
    if port:
        return f"http://127.0.0.1:{port}"

    return DEFAULT_API_BASE_URL


def _resolve_api_key() -> str:
    api_key = (os.getenv("POTPIE_API_KEY") or "").strip()
    if api_key:
        return api_key

    internal_secret = (os.getenv("INTERNAL_ADMIN_SECRET") or "").strip()
    if internal_secret:
        return internal_secret

    raise CLIError(
        "Potpie API key missing. Set POTPIE_API_KEY, or set INTERNAL_ADMIN_SECRET "
        "in your local .env for development."
    )


def _resolve_user_id() -> str | None:
    for key in ("POTPIE_USER_ID", "X_USER_ID", "defaultUsername"):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return None


def _format_http_detail(payload: Any) -> str:
    if isinstance(payload, dict):
        detail = payload.get("detail", payload)
        if isinstance(detail, str):
            return detail
        return json.dumps(detail)
    if isinstance(payload, str):
        return payload
    return str(payload)


class PotpieApiClient:
    """Small HTTP client for the existing Potpie API-key endpoints."""

    def __init__(self, base_url: str | None = None, timeout: float = 60.0):
        _load_env()
        self.base_url = _resolve_api_base_url(base_url)
        self.timeout = timeout
        self.api_key = _resolve_api_key()
        self.user_id = _resolve_user_id()

    def _headers(self) -> dict[str, str]:
        headers = {
            "X-API-Key": self.api_key,
            "User-Agent": "potpie-local-cli/1.0",
        }
        if self.user_id:
            headers["X-User-Id"] = self.user_id
        return headers

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(
                    method,
                    url,
                    headers=self._headers(),
                    **kwargs,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                detail = _format_http_detail(exc.response.json())
            except ValueError:
                detail = exc.response.text
            raise CLIError(
                f"{method} {path} failed with HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.RequestError as exc:
            raise CLIError(f"Could not reach Potpie at {self.base_url}: {exc}") from exc

        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise CLIError(f"{method} {path} returned non-JSON response") from exc

    def _request_object(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        data = self._request(method, path, **kwargs)
        if not isinstance(data, dict):
            raise CLIError(f"{method} {path} response was not an object.")
        return data

    def submit_parse(self, repo_path: Path, branch: str) -> dict[str, Any]:
        return self._request_object(
            "POST",
            "/api/v2/parse",
            json={"repo_path": str(repo_path), "branch_name": branch},
        )

    def get_parsing_status(self, project_id: str) -> dict[str, Any]:
        return self._request_object("GET", f"/api/v2/parsing-status/{project_id}")

    def list_projects(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/api/v2/projects/list")
        if not isinstance(data, list):
            raise CLIError("Project list response was not a list.")
        return data

    def create_conversation(self, project_id: str, agent_name: str) -> str:
        data = self._request_object(
            "POST",
            "/api/v2/conversations/",
            json={"project_ids": [project_id], "agent_ids": [agent_name]},
        )
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            raise CLIError("Conversation response did not include conversation_id.")
        return str(conversation_id)

    def send_message(self, conversation_id: str, content: str) -> dict[str, Any]:
        data = self._request(
            "POST",
            f"/api/v2/conversations/{conversation_id}/message/",
            params={"stream": "false"},
            json={"content": content},
        )
        if not isinstance(data, dict):
            raise CLIError("Message response was not an object.")
        return data


def _run_git(repo_path: Path, args: list[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise CLIError(detail or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _validate_git_repo(repo_path: Path) -> Path:
    resolved = repo_path.expanduser().resolve()
    if not resolved.exists():
        raise CLIError(f"Repository path does not exist: {resolved}")
    if not resolved.is_dir():
        raise CLIError(f"Repository path is not a directory: {resolved}")

    try:
        inside = _run_git(resolved, ["rev-parse", "--is-inside-work-tree"])
    except CLIError as exc:
        raise CLIError(f"Not a git repository: {resolved}") from exc

    if inside.lower() != "true":
        raise CLIError(f"Not a git repository: {resolved}")
    return resolved


def _current_branch(repo_path: Path) -> str:
    branch = _run_git(repo_path, ["rev-parse", "--abbrev-ref", "HEAD"])
    if branch == "HEAD":
        raise CLIError(
            "Repository is in detached HEAD state. Pass --branch explicitly."
        )
    return branch


def _validate_branch(repo_path: Path, branch: str) -> None:
    checks = [
        ["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        ["show-ref", "--verify", "--quiet", f"refs/remotes/origin/{branch}"],
    ]
    for args in checks:
        result = subprocess.run(
            ["git", "-C", str(repo_path), *args],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return
    raise CLIError(f"Branch not found in repository: {branch}")


def _status_value(status_response: dict[str, Any]) -> str:
    return str(status_response.get("status", "")).strip().lower()


def _extract_message_text(response: dict[str, Any]) -> str:
    for key in ("message", "response", "content", "answer"):
        value = response.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return json.dumps(response, indent=2, default=str)


def _client_or_exit(api_url: str | None) -> PotpieApiClient:
    try:
        return PotpieApiClient(api_url)
    except CLIError as exc:
        _echo_error(str(exc))
        raise typer.Exit(code=1) from exc


@app.command("start")
def start_cmd() -> None:
    """Start the local Potpie server using scripts/start.sh."""
    _load_env()
    root = _repo_root()
    script = root / "scripts" / "start.sh"
    env_file = root / ".env"

    if not script.is_file():
        _echo_error(f"Startup script not found: {script}")
        raise typer.Exit(code=1)
    if not env_file.is_file():
        _echo_error(f"Missing {env_file}. Copy .env.template to .env and configure it.")
        raise typer.Exit(code=1)

    pid_metadata = _read_pid_metadata()
    pid = _metadata_pid(pid_metadata)
    if pid and _process_running(pid):
        typer.echo(f"Potpie appears to be running already (pid {pid}).")
        typer.echo(f"Log file: {_log_file()}")
        return
    if _pid_file().exists():
        _clear_pid()

    _state_dir().mkdir(parents=True, exist_ok=True)
    log_handle = _log_file().open("a")
    log_handle.write(f"\n--- potpie start {datetime.now().isoformat()} ---\n")
    log_handle.flush()

    command = ["bash", str(script)]
    process = subprocess.Popen(
        command,
        cwd=root,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    _write_pid(process.pid, command)

    time.sleep(1)
    if process.poll() is not None:
        _clear_pid()
        _echo_error(
            f"Potpie start exited with code {process.returncode}. See {_log_file()}."
        )
        raise typer.Exit(code=process.returncode or 1)

    typer.echo(f"Potpie start launched (pid {process.pid}).")
    typer.echo(f"Log file: {_log_file()}")
    typer.echo("Use `potpie stop` to stop the local services.")


@app.command("stop")
def stop_cmd() -> None:
    """Stop local Potpie services using the local PID and scripts/stop.sh."""
    _load_env()
    root = _repo_root()
    script = root / "scripts" / "stop.sh"

    pid_metadata = _read_pid_metadata()
    pid = _metadata_pid(pid_metadata)
    if pid and _process_running(pid):
        try:
            os.killpg(pid, signal.SIGTERM)
            typer.echo(f"Sent SIGTERM to Potpie process group {pid}.")
        except ProcessLookupError:
            pass
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
                typer.echo(f"Sent SIGTERM to Potpie process {pid}.")
            except OSError:
                pass
    elif pid:
        typer.echo(f"No running Potpie process found for recorded pid {pid}.")
    elif _pid_file().exists():
        typer.echo("PID metadata missing or invalid.")
    _clear_pid()

    if script.is_file():
        result = subprocess.run(["bash", str(script)], cwd=root, check=False)
        if result.returncode != 0:
            _echo_error(f"Stop script exited with code {result.returncode}.")
            raise typer.Exit(code=result.returncode)
    else:
        typer.echo("No scripts/stop.sh found; only the recorded PID was stopped.")

    typer.echo("Potpie stop completed.")


@app.command("parse")
def parse_cmd(
    repo_path: Path = typer.Argument(..., help="Path to a local git repository."),
    branch: str | None = typer.Option(
        None,
        "--branch",
        "-b",
        help="Branch name to parse. Defaults to the current branch.",
    ),
    api_url: str | None = typer.Option(
        None,
        "--api-url",
        help="Potpie API base URL. Defaults to POTPIE_API_URL, POTPIE_BASE_URL, or localhost:8001.",
    ),
    poll_interval: float = typer.Option(
        5.0,
        "--poll-interval",
        help="Seconds between parsing status checks.",
    ),
    timeout: float = typer.Option(
        0.0,
        "--timeout",
        help="Maximum seconds to wait. 0 waits indefinitely.",
    ),
) -> None:
    """Submit a local git repository for parsing and wait until it is ready."""
    _load_env()
    try:
        resolved_repo = _validate_git_repo(repo_path)
        selected_branch = branch.strip() if branch else _current_branch(resolved_repo)
        _validate_branch(resolved_repo, selected_branch)
    except CLIError as exc:
        _echo_error(str(exc))
        raise typer.Exit(code=1) from exc

    client = _client_or_exit(api_url)

    try:
        typer.echo(f"Submitting {resolved_repo} (branch {selected_branch})...")
        submission = client.submit_parse(resolved_repo, selected_branch)
        project_id = submission.get("project_id")
        if not project_id:
            raise CLIError("Parse submission did not include project_id.")
        typer.echo(f"Project: {project_id}")

        last_status: str | None = None
        start_time = time.monotonic()
        while True:
            status_response = client.get_parsing_status(str(project_id))
            status = _status_value(status_response)
            if status != last_status:
                typer.echo(f"Status: {status or 'unknown'}")
                last_status = status

            if status == READY_STATUS:
                typer.echo("Parsing complete.")
                return
            if status == ERROR_STATUS:
                raise CLIError("Parsing failed.")
            if timeout and (time.monotonic() - start_time) >= timeout:
                raise CLIError(f"Timed out waiting for parsing after {timeout:g}s.")
            time.sleep(max(poll_interval, 0.1))
    except CLIError as exc:
        _echo_error(str(exc))
        raise typer.Exit(code=1) from exc


@app.command("chat")
def chat_cmd(
    project_id: str = typer.Argument(..., help="Parsed Potpie project ID."),
    agent_name: str = typer.Option(
        ...,
        "--agent",
        "-a",
        help="Agent ID/name, for example codebase_qna_agent.",
    ),
    branch: str | None = typer.Option(
        None,
        "--branch",
        "-b",
        help="Optional branch check for the project before chatting.",
    ),
    conversation_id: str | None = typer.Option(
        None,
        "--conversation-id",
        help="Existing conversation ID to resume. A new hidden conversation is created when omitted.",
    ),
    api_url: str | None = typer.Option(
        None,
        "--api-url",
        help="Potpie API base URL. Defaults to POTPIE_API_URL, POTPIE_BASE_URL, or localhost:8001.",
    ),
) -> None:
    """Open an interactive chat with a parsed project."""
    _load_env()
    client = _client_or_exit(api_url)

    try:
        status_response = client.get_parsing_status(project_id)
        status = _status_value(status_response)
        if status != READY_STATUS:
            raise CLIError(f"Project {project_id} is not ready (status: {status}).")

        if branch:
            projects = client.list_projects()
            match = next((p for p in projects if str(p.get("id")) == project_id), None)
            if not match:
                raise CLIError(
                    f"Project {project_id} was not found in the project list; "
                    f"cannot verify branch {branch}."
                )
            if str(match.get("branch_name")) != branch:
                raise CLIError(
                    f"Project {project_id} is for branch {match.get('branch_name')}, not {branch}."
                )

        active_conversation_id = conversation_id or client.create_conversation(
            project_id, agent_name
        )
    except CLIError as exc:
        _echo_error(str(exc))
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"Chatting with {agent_name} on project {project_id} "
        f"(conversation {active_conversation_id})."
    )
    typer.echo("Type `exit` or `quit` to leave.")

    while True:
        try:
            message = input("potpie> ").strip()
        except (KeyboardInterrupt, EOFError):
            typer.echo("\nExiting chat.")
            return

        if not message:
            continue
        if message.lower() in {"exit", "quit"}:
            typer.echo("Exiting chat.")
            return

        try:
            response = client.send_message(active_conversation_id, message)
            typer.echo(f"\nPotpie:\n{_extract_message_text(response)}\n")
        except CLIError as exc:
            _echo_error(str(exc))
            raise typer.Exit(code=1) from exc


def _should_use_local_cli(argv: list[str]) -> bool:
    if len(argv) <= 1:
        return True
    first = argv[1]
    return first in LOCAL_COMMANDS or first in {"--help", "-h"}


def main() -> None:
    if _should_use_local_cli(sys.argv):
        app()
        return

    from adapters.inbound.cli.main import main as context_cli_main

    context_cli_main()


if __name__ == "__main__":
    main()

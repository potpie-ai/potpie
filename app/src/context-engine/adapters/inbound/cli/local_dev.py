from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import typer


app = typer.Typer(
    help="Local development CLI for starting Potpie, parsing repositories, and chatting with agents."
)

READY_STATUSES = {"ready"}
FAILED_STATUSES = {"error"}
IN_PROGRESS_STATUSES = {
    "created",
    "submitted",
    "cloned",
    "parsed",
    "processing",
    "inferring",
}


def _normalize_status(value: Any) -> str:
    return str(value or "").strip().lower() or "unknown"


def _repo_root() -> Path:
    env_root = os.getenv("POTPIE_REPO_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if (root / "scripts" / "start.sh").exists():
            return root

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "scripts" / "start.sh").exists():
            return candidate

    source = Path(__file__).resolve()
    for candidate in source.parents:
        if (candidate / "scripts" / "start.sh").exists():
            return candidate

    typer.secho(
        "Could not find the Potpie repository root. Run this command from the repo root "
        "or set POTPIE_REPO_ROOT.",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(code=1)


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _headers(api_key: str | None, user_id: str | None) -> dict[str, str]:
    if not api_key:
        typer.secho(
            "Missing API key. Pass --api-key or set POTPIE_API_KEY.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    headers = {"X-API-Key": api_key}
    if user_id:
        headers["X-User-Id"] = user_id
    return headers


def _response_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text

    if isinstance(data, dict):
        detail = data.get("detail")
        if detail:
            return str(detail)
    return json.dumps(data, indent=2)


def _request(
    method: str,
    path: str,
    *,
    base_url: str,
    api_key: str | None,
    user_id: str | None,
    json_body: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    try:
        with httpx.Client(
            base_url=_normalize_base_url(base_url), timeout=timeout
        ) as client:
            response = client.request(
                method,
                path,
                headers=_headers(api_key, user_id),
                json=json_body,
            )
            response.raise_for_status()
    except httpx.ConnectError as exc:
        typer.secho(
            f"Could not connect to Potpie at {base_url}. Run `potpie start` first or pass --url.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except httpx.HTTPStatusError as exc:
        typer.secho(
            f"API error {exc.response.status_code}: {_response_detail(exc.response)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        typer.secho(f"Request failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if not response.content:
        return {}
    try:
        data = response.json()
    except ValueError as exc:
        typer.secho("API returned a non-JSON response.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    if not isinstance(data, dict):
        return {"data": data}
    return data


def _run_project_command(command: list[str]) -> None:
    exit_code = subprocess.call(command, cwd=_repo_root())
    raise typer.Exit(code=exit_code)


def _script_command(
    script_name: str, windows_script_name: str | None = None
) -> list[str]:
    root = _repo_root()
    is_windows = platform.system().lower().startswith("win")
    if is_windows and windows_script_name:
        script_path = root / "scripts" / windows_script_name
        if script_path.exists():
            return [
                "powershell",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
            ]

    bash = shutil.which("bash")
    if not bash:
        typer.secho(
            "Could not find bash. Install Git Bash or run the scripts manually.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    return [bash, str(root / "scripts" / script_name)]


@app.command()
def start() -> None:
    """Start Docker services, the API server, and the Celery worker."""
    _run_project_command(_script_command("start.sh"))


@app.command()
def stop() -> None:
    """Stop the API server, Celery worker, and Docker services."""
    _run_project_command(_script_command("stop.sh", "stop.ps1"))


@app.command()
def parse(
    repo_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Local repository path to submit for parsing.",
    ),
    branch: str | None = typer.Option(
        None, "--branch", "-b", help="Branch name to parse."
    ),
    wait: bool = typer.Option(
        True, "--wait/--no-wait", help="Poll until the project is ready."
    ),
    interval: float = typer.Option(
        5.0, "--interval", min=0.1, help="Polling interval in seconds."
    ),
    timeout: float = typer.Option(
        1800.0, "--timeout", min=1.0, help="Maximum time to wait."
    ),
    base_url: str = typer.Option(
        "http://localhost:8001",
        "--url",
        envvar="POTPIE_BASE_URL",
        help="Potpie API base URL.",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        envvar="POTPIE_API_KEY",
        help="Potpie API key.",
    ),
    user_id: str | None = typer.Option(
        None,
        "--user-id",
        envvar="POTPIE_USER_ID",
        help="User id for INTERNAL_ADMIN_SECRET impersonation.",
    ),
) -> None:
    """Submit a local repository for parsing and wait until it is ready."""
    payload: dict[str, Any] = {"repo_path": str(repo_path)}
    if branch:
        payload["branch_name"] = branch

    typer.echo(f"Submitting {repo_path} for parsing...")
    result = _request(
        "POST",
        "/api/v1/parse",
        base_url=base_url,
        api_key=api_key,
        user_id=user_id,
        json_body=payload,
    )

    project_id = result.get("project_id") or result.get("id")
    status = _normalize_status(result.get("status", "unknown"))
    if not project_id:
        typer.secho(
            "Parse response did not include a project id.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"Project: {project_id}")
    typer.echo(f"Status: {status}")

    if not wait:
        return

    deadline = time.monotonic() + timeout
    last_status = status
    while status not in READY_STATUSES:
        if status in FAILED_STATUSES:
            typer.secho(
                f"Parsing failed with status: {status}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        if status not in IN_PROGRESS_STATUSES and status != "unknown":
            typer.secho(f"Unexpected parsing status: {status}", fg=typer.colors.YELLOW)
        if time.monotonic() >= deadline:
            typer.secho(
                "Timed out waiting for parsing to finish.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        time.sleep(interval)
        status_result = _request(
            "GET",
            f"/api/v1/parsing-status/{project_id}",
            base_url=base_url,
            api_key=api_key,
            user_id=user_id,
            timeout=15.0,
        )
        status = _normalize_status(status_result.get("status", "unknown"))
        if status != last_status:
            latest = status_result.get("latest")
            suffix = f" (latest={latest})" if latest is not None else ""
            typer.echo(f"Status: {status}{suffix}")
            last_status = status

    typer.secho("Parsing complete. Project is ready.", fg=typer.colors.GREEN)


def _ensure_project_ready(
    project_id: str,
    *,
    base_url: str,
    api_key: str | None,
    user_id: str | None,
) -> None:
    status_result = _request(
        "GET",
        f"/api/v1/parsing-status/{project_id}",
        base_url=base_url,
        api_key=api_key,
        user_id=user_id,
        timeout=15.0,
    )
    status = _normalize_status(status_result.get("status", "unknown"))
    if status not in READY_STATUSES:
        typer.secho(
            f"Project {project_id} is not ready yet (status: {status}).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


def _create_conversation(
    project_id: str,
    agent: str,
    *,
    base_url: str,
    api_key: str | None,
    user_id: str | None,
) -> str:
    result = _request(
        "POST",
        "/api/v1/conversations/",
        base_url=base_url,
        api_key=api_key,
        user_id=user_id,
        json_body={"project_ids": [project_id], "agent_ids": [agent]},
    )
    conversation_id = result.get("conversation_id")
    if not conversation_id:
        typer.secho(
            "Conversation response did not include a conversation id.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    return str(conversation_id)


def _send_message(
    conversation_id: str,
    content: str,
    *,
    base_url: str,
    api_key: str | None,
    user_id: str | None,
) -> None:
    result = _request(
        "POST",
        f"/api/v1/conversations/{conversation_id}/message/",
        base_url=base_url,
        api_key=api_key,
        user_id=user_id,
        json_body={"content": content},
        timeout=300.0,
    )
    message = result.get("message")
    if message:
        typer.echo(message)
        return
    typer.echo(json.dumps(result, indent=2))


@app.command()
def chat(
    project_id: str = typer.Argument(..., help="Parsed Potpie project id."),
    agent: str = typer.Option(..., "--agent", "-a", help="Agent name to chat with."),
    branch: str | None = typer.Option(
        None,
        "--branch",
        "-b",
        help="Accepted for local workflow compatibility; project readiness is checked by id.",
    ),
    message: str | None = typer.Option(
        None,
        "--message",
        "-m",
        help="Send one message and exit instead of starting an interactive session.",
    ),
    base_url: str = typer.Option(
        "http://localhost:8001",
        "--url",
        envvar="POTPIE_BASE_URL",
        help="Potpie API base URL.",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        envvar="POTPIE_API_KEY",
        help="Potpie API key.",
    ),
    user_id: str | None = typer.Option(
        None,
        "--user-id",
        envvar="POTPIE_USER_ID",
        help="User id for INTERNAL_ADMIN_SECRET impersonation.",
    ),
) -> None:
    """Validate a project, open a conversation, and chat with an agent."""
    _ensure_project_ready(
        project_id, base_url=base_url, api_key=api_key, user_id=user_id
    )
    if branch:
        typer.echo(f"Using project {project_id}; branch option received: {branch}")

    conversation_id = _create_conversation(
        project_id,
        agent,
        base_url=base_url,
        api_key=api_key,
        user_id=user_id,
    )
    typer.echo(f"Conversation: {conversation_id}")

    if message:
        _send_message(
            conversation_id,
            message,
            base_url=base_url,
            api_key=api_key,
            user_id=user_id,
        )
        return

    typer.echo("Type /exit or /quit to end the session.")
    while True:
        try:
            user_message = typer.prompt("you")
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            return
        if user_message.strip().lower() in {"/exit", "/quit"}:
            return
        if not user_message.strip():
            continue
        _send_message(
            conversation_id,
            user_message,
            base_url=base_url,
            api_key=api_key,
            user_id=user_id,
        )

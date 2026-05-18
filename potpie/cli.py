"""Command-line interface for local Potpie development."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer

from potpie import PotpieRuntime
from potpie.agents import ChatContext
from potpie.types.project import ProjectStatus

app = typer.Typer(
    help="Run and use Potpie locally.",
    no_args_is_help=True,
)

STATE_DIR = Path(".potpie")
PID_FILE = STATE_DIR / "potpie.pid"
LOG_FILE = STATE_DIR / "potpie.log"


@dataclass(frozen=True)
class TrackedProcess:
    pid: int
    command: list[str]


def _run(coro):
    return asyncio.run(coro)


def _ensure_repo_path(repo_path: Path) -> Path:
    resolved = repo_path.expanduser().resolve()
    if not resolved.exists():
        raise typer.BadParameter(f"Repository path does not exist: {resolved}")
    if not resolved.is_dir():
        raise typer.BadParameter(f"Repository path is not a directory: {resolved}")
    if not (resolved / ".git").exists():
        raise typer.BadParameter(f"Repository path is not a Git repository: {resolved}")
    return resolved


def _repo_name_from_path(repo_path: Path) -> str:
    parent = repo_path.parent.name or "local"
    return f"{parent}/{repo_path.name}"


def _read_tracked_process() -> Optional[TrackedProcess]:
    try:
        content = PID_FILE.read_text().strip()
    except FileNotFoundError:
        return None

    try:
        data = json.loads(content)
        return TrackedProcess(pid=int(data["pid"]), command=list(data["command"]))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        try:
            return TrackedProcess(pid=int(content), command=["make", "dev"])
        except ValueError:
            return None


def _read_pid() -> Optional[int]:
    tracked_process = _read_tracked_process()
    if tracked_process is None:
        return None
    return tracked_process.pid


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _matches_tracked_process(tracked_process: TrackedProcess) -> bool:
    if not _is_running(tracked_process.pid):
        return False

    try:
        command = subprocess.check_output(
            ["ps", "-p", str(tracked_process.pid), "-o", "command="],
            text=True,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return False

    expected = " ".join(tracked_process.command)
    return command == expected or command.startswith(expected + " ")


def _is_process_group_running(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_process_group_running(pid):
            return True
        time.sleep(0.1)
    return not _is_process_group_running(pid)


@app.command()
def start(
    sandbox: Optional[str] = typer.Option(
        None,
        "--sandbox",
        help="Sandbox mode passed to make dev: local, docker, or daytona.",
    ),
):
    """Start local Potpie services in the background."""
    tracked_process = _read_tracked_process()
    if tracked_process and _matches_tracked_process(tracked_process):
        typer.echo(f"Potpie is already running with PID {tracked_process.pid}.")
        return

    STATE_DIR.mkdir(exist_ok=True)
    command = ["make", "dev"]
    if sandbox:
        command.append(f"SANDBOX={sandbox}")

    with LOG_FILE.open("a") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    PID_FILE.write_text(
        json.dumps({"pid": process.pid, "command": command}) + "\n"
    )

    typer.echo(f"Started Potpie with PID {process.pid}.")
    typer.echo("Logs: .potpie/potpie.log")


@app.command()
def stop():
    """Stop local Potpie services started by `potpie start`."""
    tracked_process = _read_tracked_process()
    if tracked_process is None:
        typer.echo("No tracked Potpie process is running.")
        PID_FILE.unlink(missing_ok=True)
        subprocess.run(["make", "infra-down"], check=False)
        return

    elif not _matches_tracked_process(tracked_process):
        typer.echo("No tracked Potpie process is running.")
        PID_FILE.unlink(missing_ok=True)
        subprocess.run(["make", "infra-down"], check=False)
        return

    else:
        try:
            os.killpg(tracked_process.pid, signal.SIGTERM)
        except ProcessLookupError:
            typer.echo("No tracked Potpie process is running.")
        except PermissionError:
            typer.echo(
                f"Permission denied while stopping PID {tracked_process.pid}.",
                err=True,
            )
            raise typer.Exit(1)
        else:
            if not _wait_for_process_group_exit(tracked_process.pid):
                typer.echo(
                    f"Timed out while stopping PID {tracked_process.pid}.",
                    err=True,
                )
                raise typer.Exit(1)
            typer.echo(f"Stopped Potpie process group for PID {tracked_process.pid}.")

    PID_FILE.unlink(missing_ok=True)
    subprocess.run(["make", "infra-down"], check=False)


@app.command()
def status():
    """Show whether a tracked local Potpie process is running."""
    tracked_process = _read_tracked_process()
    if tracked_process and _matches_tracked_process(tracked_process):
        typer.echo(f"Potpie is running with PID {tracked_process.pid}.")
        typer.echo("API: http://localhost:8001")
        return
    typer.echo("Potpie is not running.")


@app.command(name="agents")
def list_agents():
    """List available system agents."""

    async def _list_agents():
        async with PotpieRuntime.from_env() as runtime:
            for agent in runtime.agents.list_agents():
                typer.echo(f"{agent.id}\t{agent.name}")

    _run(_list_agents())


@app.command()
def parse(
    repo_path: Path = typer.Argument(..., help="Path to the local Git repository."),
    branch: str = typer.Option("main", "--branch", "-b", help="Branch to parse."),
    user_id: Optional[str] = typer.Option(
        None,
        "--user-id",
        help="User id for the local runtime. Defaults to config default_user_id.",
    ),
    user_email: Optional[str] = typer.Option(
        None,
        "--user-email",
        help="User email for the local runtime. Defaults to config default_user_email.",
    ),
):
    """Register and parse a local repository, then print the project id."""
    resolved_path = _ensure_repo_path(repo_path)

    async def _parse():
        async with PotpieRuntime.from_env() as runtime:
            effective_user_id = user_id or runtime.config.default_user_id
            effective_user_email = user_email or runtime.config.default_user_email
            repo_name = _repo_name_from_path(resolved_path)

            existing = await runtime.projects.get_by_repo(
                repo_name=repo_name,
                branch_name=branch,
                user_id=effective_user_id,
                repo_path=str(resolved_path),
            )
            if existing is None:
                project_id = await runtime.projects.register(
                    repo_name=repo_name,
                    branch_name=branch,
                    user_id=effective_user_id,
                    repo_path=str(resolved_path),
                )
                typer.echo(f"Registered project: {project_id}")
            else:
                project_id = existing.id
                typer.echo(f"Using existing project: {project_id}")

            typer.echo("Parsing repository...")
            result = await runtime.parsing.parse_project(
                project_id,
                user_id=effective_user_id,
                user_email=effective_user_email,
            )
            if not result.success:
                typer.echo(f"Parsing failed: {result.error_message}", err=True)
                raise typer.Exit(1)

            typer.echo(f"Ready: {project_id}")

    _run(_parse())


@app.command()
def chat(
    project_id: str = typer.Argument(
        ...,
        help="Project id returned by `potpie parse`.",
    ),
    agent: str = typer.Option(
        "codebase_qna_agent",
        "--agent",
        "-a",
        help="Agent id to chat with.",
    ),
    branch: Optional[str] = typer.Option(
        None,
        "--branch",
        "-b",
        help="Optional branch override. Defaults to the project's branch.",
    ),
    user_id: Optional[str] = typer.Option(
        None,
        "--user-id",
        help="User id for the local runtime. Defaults to config default_user_id.",
    ),
):
    """Open an interactive chat with a Potpie agent."""

    async def _chat():
        async with PotpieRuntime.from_env() as runtime:
            project = await runtime.projects.get(project_id)
            if project is None:
                raise typer.BadParameter(f"Project not found: {project_id}")

            project_branch = project.branch_name or branch or "main"
            if branch and project.branch_name and branch != project.branch_name:
                raise typer.BadParameter(
                    f"Branch {branch!r} does not match project branch "
                    f"{project.branch_name!r}."
                )

            effective_user_id = user_id or runtime.config.default_user_id
            agent_handle = runtime.agents.get(agent)
            history: list[str] = []

            typer.echo("Enter a message. Use Ctrl-D or /exit to quit.")
            while True:
                try:
                    query = typer.prompt("you")
                except typer.Abort:
                    typer.echo()
                    break

                if query.strip() in {"/exit", "/quit"}:
                    break
                if not query.strip():
                    continue

                ctx = ChatContext(
                    project_id=project.id,
                    project_name=project.repo_name,
                    curr_agent_id=agent,
                    history=history,
                    query=query,
                    project_status=(
                        project.status.value
                        if isinstance(project.status, ProjectStatus)
                        else str(project.status)
                    ),
                    user_id=effective_user_id,
                    repository=project.repo_name,
                    branch=project_branch,
                    local_mode=True,
                )
                response = await agent_handle.query(ctx)
                typer.echo(response.response)
                history.extend([f"user: {query}", f"assistant: {response.response}"])

    _run(_chat())


def main():
    app()


if __name__ == "__main__":
    main()

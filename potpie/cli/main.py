"""Potpie CLI — main entry point.

Usage:
    potpie start          Start the Potpie server (Docker + app)
    potpie stop           Stop all Potpie services
    potpie parse <path>   Parse a repository
    potpie chat <id>      Start an interactive chat session
    potpie projects       List parsed projects
    potpie agents         List available agents
    potpie status <id>    Check parsing status
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from potpie.cli.client import PotpieClient

console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Walk up from cwd to find the Potpie project root (has compose.yaml)."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "compose.yaml").exists() and (parent / "scripts" / "start.sh").exists():
            return parent
    # Fallback: check if POTPIE_ROOT is set
    root = os.environ.get("POTPIE_ROOT")
    if root and Path(root).exists():
        return Path(root)
    return current


def _get_client(args: argparse.Namespace) -> PotpieClient:
    """Create a PotpieClient from CLI args."""
    url = getattr(args, "url", None) or os.environ.get("POTPIE_URL", "http://localhost:8001")
    return PotpieClient(base_url=url)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_start(args: argparse.Namespace) -> None:
    """Start the Potpie server."""
    root = _find_project_root()
    start_script = root / "scripts" / "start.sh"

    if not start_script.exists():
        console.print(
            "[red]Error:[/red] Could not find scripts/start.sh. "
            "Run this command from the Potpie project directory, "
            "or set POTPIE_ROOT."
        )
        sys.exit(1)

    console.print(Panel.fit("🚀 Starting Potpie", style="bold green"))
    console.print(f"  Project root: {root}")
    console.print()

    try:
        process = subprocess.Popen(
            ["bash", str(start_script)],
            cwd=str(root),
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        process.wait()
        if process.returncode != 0:
            console.print(f"\n[red]Start script exited with code {process.returncode}[/red]")
            sys.exit(process.returncode)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)

    # Wait for health
    console.print("\n  Waiting for server to be ready...", end="")
    client = PotpieClient()
    for _ in range(30):
        if client.is_alive():
            console.print(" [green]✓ Ready![/green]")
            console.print(f"\n  Server running at [bold]http://localhost:8001[/bold]")
            return
        time.sleep(2)
        console.print(".", end="")

    console.print("\n[yellow]  Server did not respond within 60s. Check logs.[/yellow]")


def cmd_stop(args: argparse.Namespace) -> None:
    """Stop all Potpie services."""
    root = _find_project_root()
    stop_script = root / "scripts" / "stop.sh"

    if not stop_script.exists():
        console.print("[red]Error:[/red] Could not find scripts/stop.sh.")
        sys.exit(1)

    console.print(Panel.fit("🛑 Stopping Potpie", style="bold red"))

    try:
        subprocess.run(
            ["bash", str(stop_script)],
            cwd=str(root),
            check=True,
        )
        console.print("[green]✓ All services stopped.[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error stopping services (exit {e.returncode})[/red]")
        sys.exit(e.returncode)


def cmd_parse(args: argparse.Namespace) -> None:
    """Parse a repository."""
    client = _get_client(args)
    repo_path = args.repo_path
    branch = args.branch

    if not client.is_alive():
        console.print("[red]Error:[/red] Potpie server is not running. Run [bold]potpie start[/bold] first.")
        sys.exit(1)

    console.print(Panel.fit("📦 Parsing Repository", style="bold blue"))
    console.print(f"  Path:   [bold]{repo_path}[/bold]")
    console.print(f"  Branch: [bold]{branch}[/bold]")
    console.print()

    # Submit parse request
    try:
        result = client.parse(repo_path, branch)
    except Exception as e:
        console.print(f"[red]Error submitting parse request:[/red] {e}")
        sys.exit(1)

    project_id = result.get("project_id")
    if not project_id:
        console.print(f"[red]Unexpected response:[/red] {result}")
        sys.exit(1)

    console.print(f"  Project ID: [bold cyan]{project_id}[/bold cyan]")
    console.print()

    # Poll until complete
    console.print("  Parsing", end="")
    try:
        start_time = time.monotonic()
        while True:
            status = client.get_parsing_status(project_id)
            state = status.get("status", "").lower()

            if state in ("ready", "completed", "parsed"):
                elapsed = time.monotonic() - start_time
                console.print(f" [green]✓ Complete![/green] ({elapsed:.0f}s)")
                console.print()
                console.print(f"  You can now chat with this project:")
                console.print(f"  [bold]potpie chat {project_id}[/bold]")
                return

            if state in ("error", "failed"):
                console.print(f" [red]✗ Failed![/red]")
                console.print(f"  Status: {status}")
                sys.exit(1)

            console.print(".", end="", highlight=False)
            time.sleep(3)

    except TimeoutError:
        console.print(f"\n[yellow]  Timed out waiting for parsing to complete.[/yellow]")
        console.print(f"  Check status with: [bold]potpie status {project_id}[/bold]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print(f"\n[yellow]  Interrupted. Parsing continues in background.[/yellow]")
        console.print(f"  Check status with: [bold]potpie status {project_id}[/bold]")
        sys.exit(130)


def cmd_status(args: argparse.Namespace) -> None:
    """Check parsing status for a project."""
    client = _get_client(args)
    project_id = args.project_id

    if not client.is_alive():
        console.print("[red]Error:[/red] Server not running.")
        sys.exit(1)

    try:
        status = client.get_parsing_status(project_id)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    state = status.get("status", "unknown")
    style = "green" if state in ("ready", "completed", "parsed") else "yellow" if state == "processing" else "red"

    console.print(f"  Project: [bold]{project_id}[/bold]")
    console.print(f"  Status:  [{style}]{state}[/{style}]")

    for key, val in status.items():
        if key != "status":
            console.print(f"  {key}: {val}")


def cmd_projects(args: argparse.Namespace) -> None:
    """List all parsed projects."""
    client = _get_client(args)

    if not client.is_alive():
        console.print("[red]Error:[/red] Server not running.")
        sys.exit(1)

    try:
        projects = client.list_projects()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    if not projects:
        console.print("  No projects found. Parse a repository first:")
        console.print("  [bold]potpie parse /path/to/repo[/bold]")
        return

    table = Table(title="📁 Projects")
    table.add_column("ID", style="cyan", max_width=40)
    table.add_column("Name", style="bold")
    table.add_column("Branch")
    table.add_column("Status")

    for p in projects:
        pid = str(p.get("project_id", p.get("id", "?")))
        name = p.get("project_name", p.get("repo_name", "?"))
        branch = p.get("branch_name", "?")
        status = p.get("status", "?")
        status_style = "green" if status in ("ready", "parsed") else "yellow"
        table.add_row(pid, name, branch, f"[{status_style}]{status}[/{status_style}]")

    console.print(table)


def cmd_agents(args: argparse.Namespace) -> None:
    """List available agents."""
    client = _get_client(args)

    if not client.is_alive():
        console.print("[red]Error:[/red] Server not running.")
        sys.exit(1)

    try:
        agents = client.list_agents()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    table = Table(title="🤖 Available Agents")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Description", max_width=60)

    for a in agents:
        aid = a.get("id", "?")
        name = a.get("name", aid)
        desc = a.get("description", "")
        table.add_row(aid, name, desc)

    console.print(table)


def cmd_chat(args: argparse.Namespace) -> None:
    """Start an interactive chat session."""
    client = _get_client(args)
    project_id = args.project_id
    agent = args.agent

    if not client.is_alive():
        console.print("[red]Error:[/red] Server not running. Run [bold]potpie start[/bold] first.")
        sys.exit(1)

    # Validate project is ready
    try:
        status = client.get_parsing_status(project_id)
        state = status.get("status", "").lower()
        if state not in ("ready", "completed", "parsed"):
            console.print(f"[red]Error:[/red] Project is not ready (status: {state}).")
            console.print("  Wait for parsing to complete or run:")
            console.print(f"  [bold]potpie status {project_id}[/bold]")
            sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error checking project status:[/red] {e}")
        sys.exit(1)

    # Create conversation
    try:
        conv = client.create_conversation(project_id, agent)
        conversation_id = conv.get("conversation_id")
        if not conversation_id:
            console.print(f"[red]Error creating conversation:[/red] {conv}")
            sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error creating conversation:[/red] {e}")
        sys.exit(1)

    console.print(Panel.fit("💬 Potpie Chat", style="bold magenta"))
    console.print(f"  Project: [bold]{project_id}[/bold]")
    console.print(f"  Agent:   [bold]{agent}[/bold]")
    console.print(f"  Session: [dim]{conversation_id}[/dim]")
    console.print()
    console.print("  Type your message and press Enter. Use [bold]Ctrl+C[/bold] or [bold]/quit[/bold] to exit.")
    console.print()

    # Interactive loop
    try:
        while True:
            try:
                user_input = console.input("[bold green]You>[/bold green] ")
            except EOFError:
                break

            if not user_input.strip():
                continue

            if user_input.strip().lower() in ("/quit", "/exit", "/q"):
                break

            # Stream response
            console.print("[bold blue]Potpie>[/bold blue] ", end="")
            try:
                full_response = ""
                for chunk in client.send_message(conversation_id, user_input):
                    console.print(chunk, end="", highlight=False)
                    full_response += chunk
                console.print()  # newline after response
            except Exception as e:
                console.print(f"\n[red]Error:[/red] {e}")

            console.print()

    except KeyboardInterrupt:
        pass

    console.print("\n[dim]Chat session ended.[/dim]")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="potpie",
        description="🥧 Potpie CLI — Local development interface for Potpie",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  potpie start                          Start the server
  potpie stop                           Stop all services
  potpie parse /path/to/repo            Parse a local repo
  potpie parse owner/repo               Parse a GitHub repo
  potpie parse . --branch develop       Parse current dir, specific branch
  potpie chat <project-id>              Chat with default agent
  potpie chat <project-id> --agent qna_agent
  potpie projects                       List all projects
  potpie agents                         List available agents
  potpie status <project-id>            Check parsing status
""",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Potpie server URL (default: http://localhost:8001 or $POTPIE_URL)",
    )

    sub = parser.add_subparsers(dest="command")

    # start
    sub.add_parser("start", help="Start the Potpie server")

    # stop
    sub.add_parser("stop", help="Stop all Potpie services")

    # parse
    parse_cmd = sub.add_parser("parse", help="Parse a repository")
    parse_cmd.add_argument("repo_path", help="Repository path (local or owner/repo)")
    parse_cmd.add_argument("--branch", default="main", help="Branch name (default: main)")

    # status
    status_cmd = sub.add_parser("status", help="Check parsing status")
    status_cmd.add_argument("project_id", help="Project ID")

    # projects
    sub.add_parser("projects", help="List parsed projects")

    # agents
    sub.add_parser("agents", help="List available agents")

    # chat
    chat_cmd = sub.add_parser("chat", help="Start interactive chat")
    chat_cmd.add_argument("project_id", help="Project ID to chat about")
    chat_cmd.add_argument("--agent", default="codebase_qna_agent", help="Agent to use (default: codebase_qna_agent)")

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "start": cmd_start,
        "stop": cmd_stop,
        "parse": cmd_parse,
        "status": cmd_status,
        "projects": cmd_projects,
        "agents": cmd_agents,
        "chat": cmd_chat,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Entry point for the Potpie CLI (``potpie`` command)."""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="potpie",
        description="Potpie CLI – manage and interact with a local Potpie server.",
    )
    parser.add_argument(
        "--url",
        metavar="URL",
        default=None,
        help="Base URL of the Potpie server (default: http://localhost:8001).",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ---- start ----
    subparsers.add_parser(
        "start",
        help="Start the local Potpie server.",
        description="Start the Potpie server (gunicorn + Celery worker).",
    )

    # ---- stop ----
    subparsers.add_parser(
        "stop",
        help="Stop the local Potpie server.",
        description="Stop all running Potpie server processes.",
    )

    # ---- parse ----
    parse_parser = subparsers.add_parser(
        "parse",
        help="Submit a repository for parsing.",
        description="Submit a local repository for parsing and poll until complete.",
    )
    parse_parser.add_argument(
        "repo_path",
        help="Path to the local repository directory.",
    )
    parse_parser.add_argument(
        "--branch",
        default="main",
        metavar="BRANCH",
        help="Branch name to parse (default: main).",
    )

    # ---- chat ----
    chat_parser = subparsers.add_parser(
        "chat",
        help="Start an interactive chat session with an agent.",
        description="Open an interactive chat session for the given project.",
    )
    chat_parser.add_argument(
        "project_id",
        help="Project ID to chat about.",
    )
    chat_parser.add_argument(
        "--agent",
        required=True,
        metavar="AGENT_ID",
        help="Agent ID (or name) to use for the conversation.",
    )

    # ---- list-projects ----
    subparsers.add_parser(
        "list-projects",
        help="List all projects registered on the server.",
        description="Display all projects for the current user.",
    )

    # ---- list-agents ----
    subparsers.add_parser(
        "list-agents",
        help="List all available agents.",
        description="Display all system and custom agents available on the server.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the ``potpie`` CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    base_url: str | None = args.url  # may be None (each command uses its default)

    if args.command == "start":
        from potpie.cli.commands.start import start_server
        start_server()

    elif args.command == "stop":
        from potpie.cli.commands.stop import stop_server
        stop_server()

    elif args.command == "parse":
        from potpie.cli.commands.parse import parse_repo
        parse_repo(args.repo_path, branch=args.branch, base_url=base_url)

    elif args.command == "chat":
        from potpie.cli.commands.chat import start_chat
        start_chat(args.project_id, agent_id=args.agent, base_url=base_url)

    elif args.command == "list-projects":
        from potpie.cli.commands.list_projects import list_projects
        list_projects(base_url=base_url)

    elif args.command == "list-agents":
        from potpie.cli.commands.list_agents import list_agents
        list_agents(base_url=base_url)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

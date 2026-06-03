"""CLI for exercising the sandbox providers end-to-end.

Goes through ``build_sandbox_container``, so it picks up the same provider
config the worker uses. Useful for reproducing cross-process persistence
issues without dragging the full agent stack along — each invocation is a
new Python process, exactly like a worker restart.

Configuration (env vars, same as the worker):

    SANDBOX_WORKSPACE_PROVIDER=daytona|local
    SANDBOX_RUNTIME_PROVIDER=daytona|local_subprocess|docker
    DAYTONA_API_URL, DAYTONA_API_KEY, DAYTONA_ORGANIZATION_ID, DAYTONA_SNAPSHOT
    GITHUB_TOKEN  (optional — used as auth_token when --token isn't passed)

Subcommands:

    workspace create        Create or reuse a workspace; prints workspace info.
    workspace info          Same as create — useful for "what's my path?".
    workspace destroy       Tear down the worktree (sandbox stays alive).
    exec  -- <cmd>          Run a command in the workspace.

Workspace identity flags are repeated on every subcommand because each CLI
invocation is a fresh process — there's no cross-call session state. That's
the point: if you can ``create`` then ``exec`` the same identity in two
separate processes and see your work, persistence is working.

Examples (against a local Daytona stack)::

    export $(grep -v '^#' .env.daytona.local | xargs)
    export GITHUB_TOKEN=ghp_xxx

    # First process: clone + write a marker file
    python -m scripts.sandbox_cli exec \\
        --user u1 --project p1 --repo octocat/Hello-World \\
        --conversation demo-1 --base-ref master \\
        -- bash -c 'echo hi > /tmp/marker'

    # Second process: read it back. Same conversation_id → same worktree.
    python -m scripts.sandbox_cli exec \\
        --user u1 --project p1 --repo octocat/Hello-World \\
        --conversation demo-1 --base-ref master \\
        -- cat /tmp/marker
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Allow ``python scripts/sandbox_cli.py`` from inside the sandbox dir without
# the ``-m`` flag — keeps the example commands in --help copy-pasteable.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from sandbox.bootstrap.container import build_sandbox_container  # noqa: E402
from sandbox.domain.models import (  # noqa: E402
    ExecRequest,
    RepoIdentity,
    RuntimeRequest,
    WorkspaceMode,
    WorkspaceRequest,
)


def _add_workspace_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--user", required=True, help="Tenant / user id.")
    p.add_argument("--project", required=True, help="Project id within that user.")
    p.add_argument(
        "--repo",
        required=True,
        help="owner/repo — used as the canonical repo name and to derive the "
        "default GitHub URL.",
    )
    p.add_argument(
        "--repo-url",
        default=None,
        help="Override the clone URL (default: https://github.com/<repo>.git).",
    )
    p.add_argument("--base-ref", default="main", help="Ref to branch off from.")
    p.add_argument(
        "--conversation",
        default=None,
        help="Conversation id — drives the agent branch name and the workspace "
        "key for EDIT mode. Persistence hangs off this.",
    )
    p.add_argument(
        "--task",
        default=None,
        help="Task id — same role as --conversation but for TASK mode.",
    )
    p.add_argument(
        "--branch",
        default=None,
        help="Explicit branch name. Defaults to agent/edits-<conversation> "
        "for EDIT mode.",
    )
    p.add_argument(
        "--mode",
        default="edit",
        choices=[m.value for m in WorkspaceMode],
        help="Workspace mode.",
    )
    p.add_argument(
        "--token",
        default=None,
        help="Auth token for the repo (default: $GITHUB_TOKEN).",
    )
    p.add_argument(
        "--no-create-branch",
        action="store_true",
        help="Don't create a new branch in the worktree (use base_ref directly, "
        "detached for ANALYSIS).",
    )


def _build_request(args: argparse.Namespace) -> WorkspaceRequest:
    return WorkspaceRequest(
        user_id=args.user,
        project_id=args.project,
        repo=RepoIdentity(repo_name=args.repo, repo_url=args.repo_url),
        base_ref=args.base_ref,
        mode=WorkspaceMode(args.mode),
        conversation_id=args.conversation,
        task_id=args.task,
        branch_name=args.branch,
        create_branch=not args.no_create_branch,
        auth_token=args.token or os.getenv("GITHUB_TOKEN"),
    )


def _print_workspace(workspace) -> None:
    loc = workspace.location
    print(f"workspace_id      = {workspace.id}")
    print(f"  key             = {workspace.key}")
    print(f"  backend         = {workspace.backend_kind}")
    print(f"  state           = {workspace.state.value}")
    if loc.local_path:
        print(f"  local_path      = {loc.local_path}")
    if loc.remote_path:
        print(f"  remote_path     = {loc.remote_path}")
    if loc.backend_workspace_id:
        print(f"  sandbox_id      = {loc.backend_workspace_id}")
    branch = workspace.metadata.get("branch") if workspace.metadata else None
    if branch:
        print(f"  branch          = {branch}")


async def cmd_create(args: argparse.Namespace) -> int:
    container = build_sandbox_container()
    workspace = await container.service.get_or_create_workspace(_build_request(args))
    _print_workspace(workspace)
    return 0


async def cmd_exec(args: argparse.Namespace) -> int:
    if not args.cmd:
        sys.stderr.write("error: no command provided after `--`\n")
        return 2
    container = build_sandbox_container()
    workspace = await container.service.get_or_create_workspace(_build_request(args))
    await container.service.get_or_create_runtime(RuntimeRequest(workspace.id))
    if args.show_workspace:
        _print_workspace(workspace)
        print("---")
    result = await container.service.exec(
        workspace.id,
        ExecRequest(cmd=tuple(args.cmd), shell=False, timeout_s=args.timeout),
    )
    if result.stdout:
        sys.stdout.buffer.write(result.stdout)
        if not result.stdout.endswith(b"\n"):
            sys.stdout.write("\n")
    if result.stderr:
        sys.stderr.buffer.write(result.stderr)
    return result.exit_code


async def cmd_destroy(args: argparse.Namespace) -> int:
    container = build_sandbox_container()
    workspace = await container.service.get_or_create_workspace(_build_request(args))
    await container.service.destroy_workspace(workspace.id)
    print(f"destroyed workspace {workspace.id} ({workspace.key})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sandbox-cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="Create or reuse a workspace.")
    _add_workspace_args(p_create)
    p_create.set_defaults(func=cmd_create)

    p_info = sub.add_parser("info", help="Alias for `create` — prints workspace info.")
    _add_workspace_args(p_info)
    p_info.set_defaults(func=cmd_create)

    p_exec = sub.add_parser(
        "exec",
        help="Run a command inside the workspace. Use `--` to separate the "
        "command from CLI flags.",
    )
    _add_workspace_args(p_exec)
    p_exec.add_argument(
        "--show-workspace",
        action="store_true",
        help="Print workspace info before running the command.",
    )
    p_exec.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-exec timeout in seconds.",
    )
    p_exec.add_argument("cmd", nargs=argparse.REMAINDER)
    p_exec.set_defaults(func=cmd_exec)

    p_destroy = sub.add_parser(
        "destroy", help="Remove the workspace's worktree (the sandbox stays)."
    )
    _add_workspace_args(p_destroy)
    p_destroy.set_defaults(func=cmd_destroy)

    args = parser.parse_args(argv)
    # `argparse.REMAINDER` keeps the leading `--` if present; strip it.
    if getattr(args, "cmd", None) and args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    return asyncio.run(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

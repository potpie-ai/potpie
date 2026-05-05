"""End-to-end tests for the Daytona runtime backend against a real local stack.

These tests require the docker-compose Daytona stack from
`/Users/nandan/Desktop/Dev/daytona/docker/docker-compose.yaml` to be reachable
at http://localhost:3000. When `DAYTONA_API_KEY` is not set the conftest
fixture mints a dev key by scripting the dex OIDC login flow.

Each test creates a *real* Daytona sandbox, clones a small public repo into it,
runs commands inside the managed runtime, and tears the sandbox back down so we
do not leak resources.

The Phase 3 model uses ``git worktree`` inside the Daytona sandbox to share a
single sandbox across many branches. ``git worktree`` requires the git CLI in
the sandbox image — the tests skip if the configured snapshot doesn't ship it.
The Phase 1 ``potpie/agent-sandbox`` snapshot installs everything needed.
"""

from __future__ import annotations

import os

import pytest

from sandbox.application.services.sandbox_service import SandboxService
from sandbox.domain.models import (
    CommandKind,
    ExecRequest,
    RepoIdentity,
    RuntimeRequest,
    WorkspaceMode,
    WorkspaceRequest,
)


# A small public repo used so the test does not depend on github auth tokens.
PUBLIC_REPO_URL = os.getenv(
    "DAYTONA_TEST_REPO_URL",
    "https://github.com/octocat/Hello-World.git",
)
PUBLIC_REPO_NAME = os.getenv("DAYTONA_TEST_REPO_NAME", "octocat/Hello-World")
PUBLIC_REPO_REF = os.getenv("DAYTONA_TEST_REPO_REF", "master")


def _request(*, conversation_id: str, project_id: str = "potpie-e2e") -> WorkspaceRequest:
    return WorkspaceRequest(
        user_id="potpie-e2e",
        project_id=project_id,
        repo=RepoIdentity(repo_name=PUBLIC_REPO_NAME, repo_url=PUBLIC_REPO_URL),
        base_ref=PUBLIC_REPO_REF,
        mode=WorkspaceMode.EDIT,
        conversation_id=conversation_id,
        create_branch=True,
    )


async def _safe_destroy(service: SandboxService, workspace_id: str) -> None:
    try:
        await service.destroy_workspace(workspace_id)
    except Exception as exc:  # noqa: BLE001 -- best-effort teardown
        print(f"warning: destroy_workspace failed: {exc!r}")


async def _require_git_cli(service: SandboxService, workspace_id: str) -> None:
    """Skip the rest of this test if git CLI is missing in the sandbox.

    The Phase 3 worktree-per-branch model needs git CLI; the slim Daytona
    image (Phase 0) doesn't ship one. Phase 1's `potpie/agent-sandbox` does.
    """
    result = await service.exec(
        workspace_id,
        ExecRequest(cmd=("git", "--version"), command_kind=CommandKind.READ),
    )
    if result.exit_code != 0:
        pytest.skip(
            "Daytona sandbox image lacks `git` CLI required by the Phase 3 "
            "worktree model. Set DAYTONA_SNAPSHOT to potpie/agent-sandbox or "
            "another image with git installed."
        )


@pytest.mark.asyncio
async def test_daytona_creates_sandbox_clones_repo_and_execs(
    daytona_service: SandboxService,
) -> None:
    workspace = await daytona_service.get_or_create_workspace(
        _request(conversation_id="daytona-clone")
    )
    try:
        assert workspace.location.backend_workspace_id
        assert workspace.location.remote_path
        # Worktree path: <root>/<repo_safe>/worktrees/<branch_safe>
        assert "/worktrees/" in (workspace.location.remote_path or "")

        runtime = await daytona_service.get_or_create_runtime(
            RuntimeRequest(workspace_id=workspace.id)
        )
        assert runtime.backend_runtime_id == workspace.location.backend_workspace_id

        result = await daytona_service.exec(
            workspace.id,
            ExecRequest(cmd=("ls", "-1"), command_kind=CommandKind.READ),
        )
        assert result.exit_code == 0
        assert b"README" in result.stdout
    finally:
        await _safe_destroy(daytona_service, workspace.id)


@pytest.mark.asyncio
async def test_daytona_writes_persist_within_session(
    daytona_service: SandboxService,
) -> None:
    workspace = await daytona_service.get_or_create_workspace(
        _request(conversation_id="daytona-write")
    )
    try:
        write = await daytona_service.exec(
            workspace.id,
            ExecRequest(
                cmd=("sh", "-c", "printf 'potpie-was-here' > marker.txt"),
                command_kind=CommandKind.WRITE,
            ),
        )
        assert write.exit_code == 0

        verify = await daytona_service.exec(
            workspace.id,
            ExecRequest(cmd=("cat", "marker.txt"), command_kind=CommandKind.READ),
        )
        assert verify.exit_code == 0
        assert b"potpie-was-here" in verify.stdout

        refreshed = await daytona_service.get_workspace(workspace.id)
        assert refreshed.dirty is True
    finally:
        await _safe_destroy(daytona_service, workspace.id)


@pytest.mark.asyncio
async def test_daytona_workspace_lookup_is_idempotent(
    daytona_service: SandboxService,
) -> None:
    request = _request(conversation_id="daytona-idempotent")
    workspace_a = await daytona_service.get_or_create_workspace(request)
    workspace_b = await daytona_service.get_or_create_workspace(request)
    try:
        assert workspace_a.id == workspace_b.id
        assert (
            workspace_a.location.backend_workspace_id
            == workspace_b.location.backend_workspace_id
        )
        assert workspace_a.location.remote_path == workspace_b.location.remote_path
    finally:
        await _safe_destroy(daytona_service, workspace_a.id)


@pytest.mark.asyncio
async def test_daytona_branch_is_checked_out(
    daytona_service: SandboxService,
) -> None:
    """The worktree's HEAD points at the agent branch."""
    workspace = await daytona_service.get_or_create_workspace(
        _request(conversation_id="branch-check")
    )
    try:
        await _require_git_cli(daytona_service, workspace.id)
        result = await daytona_service.exec(
            workspace.id,
            ExecRequest(
                cmd=("git", "rev-parse", "--abbrev-ref", "HEAD"),
                command_kind=CommandKind.READ,
            ),
        )
        assert result.exit_code == 0
        assert b"agent/edits-branch-check" in result.stdout
        assert workspace.metadata.get("branch") == "agent/edits-branch-check"
    finally:
        await _safe_destroy(daytona_service, workspace.id)


@pytest.mark.asyncio
async def test_daytona_two_branches_share_one_sandbox(
    daytona_service: SandboxService,
) -> None:
    """The Phase 3 invariant: two branches on the same project map to the
    same Daytona sandbox id and different worktree paths."""
    ws_a = await daytona_service.get_or_create_workspace(
        _request(conversation_id="share-a")
    )
    ws_b = await daytona_service.get_or_create_workspace(
        _request(conversation_id="share-b")
    )
    try:
        await _require_git_cli(daytona_service, ws_a.id)
        assert ws_a.id != ws_b.id
        assert (
            ws_a.location.backend_workspace_id
            == ws_b.location.backend_workspace_id
        )
        assert ws_a.location.remote_path != ws_b.location.remote_path

        # Each worktree exec runs in its own dir.
        pwd_a = await daytona_service.exec(
            ws_a.id, ExecRequest(cmd=("pwd",), command_kind=CommandKind.READ)
        )
        pwd_b = await daytona_service.exec(
            ws_b.id, ExecRequest(cmd=("pwd",), command_kind=CommandKind.READ)
        )
        assert pwd_a.stdout.strip() != pwd_b.stdout.strip()
        assert b"share-a" in pwd_a.stdout or b"agent_edits-share-a" in pwd_a.stdout
        assert b"share-b" in pwd_b.stdout or b"agent_edits-share-b" in pwd_b.stdout
    finally:
        await _safe_destroy(daytona_service, ws_a.id)
        await _safe_destroy(daytona_service, ws_b.id)

"""End-to-end tests for the Docker runtime backend.

These run the full SandboxService flow with a real long-lived container so
we exercise create, exec, hibernate/restart, and persistence-across-restart.
Skipped when the Docker daemon is unavailable.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from sandbox.application.services.sandbox_service import SandboxService
from sandbox.domain.models import (
    CommandKind,
    ExecRequest,
    NetworkMode,
    RepoIdentity,
    RuntimeRequest,
    RuntimeState,
    WorkspaceMode,
    WorkspaceRequest,
)


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=20, check=False
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not _docker_available(), reason="Docker daemon is not available"
)


def _request(source: Path, *, conversation_id: str) -> WorkspaceRequest:
    return WorkspaceRequest(
        user_id="u1",
        project_id="p1",
        repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
        base_ref="main",
        mode=WorkspaceMode.EDIT,
        conversation_id=conversation_id,
        create_branch=True,
    )


@pytest.fixture
def image() -> str:
    return os.getenv("SANDBOX_DOCKER_IMAGE", "busybox:latest")


@pytest.mark.asyncio
async def test_docker_full_flow_read_write_destroy(
    source_repo: Path, docker_service: SandboxService, image: str
) -> None:
    workspace = await docker_service.get_or_create_workspace(
        _request(source_repo, conversation_id="docker-flow")
    )
    runtime = await docker_service.get_or_create_runtime(
        RuntimeRequest(workspace_id=workspace.id, image=image, writable=True)
    )
    try:
        read = await docker_service.exec(
            workspace.id,
            ExecRequest(cmd=("cat", "README.md"), command_kind=CommandKind.READ),
        )
        write = await docker_service.exec(
            workspace.id,
            ExecRequest(
                cmd=("sh", "-c", "printf docker-wrote > generated.txt"),
                command_kind=CommandKind.WRITE,
            ),
        )
        verify = await docker_service.exec(
            workspace.id,
            ExecRequest(cmd=("cat", "generated.txt"), command_kind=CommandKind.READ),
        )
    finally:
        await docker_service.destroy_runtime(runtime.id)

    assert read.exit_code == 0
    assert read.stdout == b"hello e2e\n"
    assert write.exit_code == 0
    assert verify.stdout == b"docker-wrote"

    # Mount points back at the host worktree, so the file should also exist there
    # — this is the key persistence guarantee for the local Docker provider.
    host_path = Path(workspace.location.local_path) / "generated.txt"
    assert host_path.read_bytes() == b"docker-wrote"


@pytest.mark.asyncio
async def test_docker_runtime_survives_hibernate_and_restart(
    source_repo: Path, docker_service: SandboxService, image: str
) -> None:
    workspace = await docker_service.get_or_create_workspace(
        _request(source_repo, conversation_id="docker-hibernate")
    )
    runtime = await docker_service.get_or_create_runtime(
        RuntimeRequest(workspace_id=workspace.id, image=image, writable=True)
    )
    try:
        await docker_service.exec(
            workspace.id,
            ExecRequest(
                cmd=("sh", "-c", "printf one > step.txt"),
                command_kind=CommandKind.WRITE,
            ),
        )
        await docker_service.hibernate_runtime(runtime.id)
        stopped = await docker_service._store.get_runtime(runtime.id)
        assert stopped.state is RuntimeState.STOPPED

        # Next exec must transparently restart the same container; the file
        # written before hibernate must still be readable through the same mount.
        result = await docker_service.exec(
            workspace.id,
            ExecRequest(cmd=("cat", "step.txt"), command_kind=CommandKind.READ),
        )
        assert result.stdout == b"one"
        running = await docker_service._store.get_runtime(runtime.id)
        assert running.state is RuntimeState.RUNNING
    finally:
        await docker_service.destroy_runtime(runtime.id)


@pytest.mark.asyncio
async def test_docker_destroy_runtime_keeps_workspace_files(
    source_repo: Path, docker_service: SandboxService, image: str
) -> None:
    workspace = await docker_service.get_or_create_workspace(
        _request(source_repo, conversation_id="docker-keep-ws")
    )
    runtime = await docker_service.get_or_create_runtime(
        RuntimeRequest(workspace_id=workspace.id, image=image, writable=True)
    )
    await docker_service.exec(
        workspace.id,
        ExecRequest(
            cmd=("sh", "-c", "printf survives > artifact.txt"),
            command_kind=CommandKind.WRITE,
        ),
    )
    await docker_service.destroy_runtime(runtime.id)

    artifact = Path(workspace.location.local_path) / "artifact.txt"
    assert artifact.read_bytes() == b"survives"

    # Re-attach a fresh runtime to the same workspace and confirm the file is
    # still visible inside the new container.
    new_runtime = await docker_service.get_or_create_runtime(
        RuntimeRequest(workspace_id=workspace.id, image=image, writable=True)
    )
    try:
        result = await docker_service.exec(
            workspace.id,
            ExecRequest(cmd=("cat", "artifact.txt"), command_kind=CommandKind.READ),
        )
        assert result.stdout == b"survives"
        assert new_runtime.id != runtime.id
    finally:
        await docker_service.destroy_runtime(new_runtime.id)


@pytest.mark.asyncio
async def test_docker_network_none_blocks_outbound(
    source_repo: Path, docker_service: SandboxService, image: str
) -> None:
    workspace = await docker_service.get_or_create_workspace(
        _request(source_repo, conversation_id="docker-net-none")
    )
    runtime = await docker_service.get_or_create_runtime(
        RuntimeRequest(
            workspace_id=workspace.id,
            image=image,
            writable=True,
            network=NetworkMode.NONE,
        )
    )
    try:
        # busybox `nslookup` against an unresolvable host should fail when the
        # container has no network. We use --net=none and expect a non-zero exit.
        result = await docker_service.exec(
            workspace.id,
            ExecRequest(
                cmd=("sh", "-c", "nslookup example.com 2>/dev/null; echo exit=$?"),
                command_kind=CommandKind.READ,
                timeout_s=10,
            ),
        )
    finally:
        await docker_service.destroy_runtime(runtime.id)

    assert b"exit=0" not in result.stdout, result.stdout

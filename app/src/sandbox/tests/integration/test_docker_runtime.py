from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from sandbox.adapters.outbound.docker.runtime import DockerRuntimeProvider
from sandbox.adapters.outbound.file.json_store import JsonSandboxStore
from sandbox.adapters.outbound.local.git_workspace import LocalGitWorkspaceProvider
from sandbox.adapters.outbound.memory.locks import InMemoryLockManager
from sandbox.application.services.sandbox_service import SandboxService
from sandbox.domain.models import (
    CommandKind,
    ExecRequest,
    RepoIdentity,
    RuntimeRequest,
    WorkspaceMode,
    WorkspaceRequest,
)


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0


def _test_root(tmp_path: Path) -> Path:
    override = os.getenv("SANDBOX_TEST_TMPDIR")
    if not override:
        return tmp_path
    path = Path(override)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def _make_repo(root: Path) -> Path:
    repo = root / "docker-source"
    if repo.exists():
        subprocess.run(["rm", "-rf", str(repo)], check=False)
    repo.mkdir(parents=True)
    _run(["git", "init", "-b", "main"], repo)
    _run(["git", "config", "user.email", "test@example.com"], repo)
    _run(["git", "config", "user.name", "Test User"], repo)
    (repo / "README.md").write_text("hello docker sandbox\n", encoding="utf-8")
    _run(["git", "add", "README.md"], repo)
    _run(["git", "commit", "-m", "initial"], repo)
    return repo


@pytest.mark.skipif(not _docker_available(), reason="Docker daemon is not available")
@pytest.mark.asyncio
async def test_docker_runtime_execs_inside_local_workspace(tmp_path: Path) -> None:
    root = _test_root(tmp_path)
    source = _make_repo(root)
    service = SandboxService(
        workspace_provider=LocalGitWorkspaceProvider(root / ".repos"),
        runtime_provider=DockerRuntimeProvider(name_prefix="potpie-sandbox-test"),
        store=JsonSandboxStore(root / "metadata.json"),
        locks=InMemoryLockManager(),
    )
    workspace = await service.get_or_create_workspace(
        WorkspaceRequest(
            user_id="u1",
            project_id="p1",
            repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="docker-smoke",
            create_branch=True,
        )
    )
    runtime = await service.get_or_create_runtime(
        RuntimeRequest(
            workspace_id=workspace.id,
            image=os.getenv("SANDBOX_DOCKER_IMAGE", "busybox:latest"),
            writable=True,
        )
    )
    try:
        read = await service.exec(
            workspace.id,
            ExecRequest(cmd=("cat", "README.md"), command_kind=CommandKind.READ),
        )
        write = await service.exec(
            workspace.id,
            ExecRequest(
                cmd=("sh", "-c", "printf changed > generated.txt"),
                command_kind=CommandKind.WRITE,
            ),
        )
        verify = await service.exec(
            workspace.id,
            ExecRequest(cmd=("cat", "generated.txt"), command_kind=CommandKind.READ),
        )
    finally:
        await service.destroy_runtime(runtime.id)

    assert read.exit_code == 0
    assert read.stdout == b"hello docker sandbox\n"
    assert write.exit_code == 0
    assert verify.stdout == b"changed"

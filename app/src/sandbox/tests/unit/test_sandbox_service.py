from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sandbox.adapters.outbound.file.json_store import JsonSandboxStore
from sandbox.adapters.outbound.local.git_workspace import LocalGitWorkspaceProvider
from sandbox.adapters.outbound.local.subprocess_runtime import LocalSubprocessRuntimeProvider
from sandbox.adapters.outbound.memory.locks import InMemoryLockManager
from sandbox.application.services.sandbox_service import SandboxService
from sandbox.domain.errors import RuntimeCommandRejected
from sandbox.domain.models import (
    CommandKind,
    ExecRequest,
    RepoIdentity,
    RuntimeRequest,
    WorkspaceMode,
    WorkspaceRequest,
)


def _run(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    repo.mkdir()
    _run(["git", "init", "-b", "main"], repo)
    _run(["git", "config", "user.email", "test@example.com"], repo)
    _run(["git", "config", "user.name", "Test User"], repo)
    (repo / "README.md").write_text("hello sandbox\n", encoding="utf-8")
    _run(["git", "add", "README.md"], repo)
    _run(["git", "commit", "-m", "initial"], repo)
    return repo


@pytest.mark.asyncio
async def test_local_workspace_and_runtime_read(tmp_path: Path) -> None:
    source = _make_repo(tmp_path)
    store = JsonSandboxStore(tmp_path / "metadata.json")
    service = SandboxService(
        workspace_provider=LocalGitWorkspaceProvider(tmp_path / ".repos"),
        runtime_provider=LocalSubprocessRuntimeProvider(),
        store=store,
        locks=InMemoryLockManager(),
    )
    workspace = await service.get_or_create_workspace(
        WorkspaceRequest(
            user_id="u1",
            project_id="p1",
            repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="c1",
            create_branch=True,
        )
    )
    runtime = await service.get_or_create_runtime(
        RuntimeRequest(workspace_id=workspace.id, writable=False)
    )

    result = await service.exec(
        workspace.id,
        ExecRequest(cmd=("cat", "README.md"), command_kind=CommandKind.READ),
    )

    assert runtime.workspace_id == workspace.id
    assert result.exit_code == 0
    assert result.stdout == b"hello sandbox\n"


@pytest.mark.asyncio
async def test_local_runtime_rejects_write_when_disabled(tmp_path: Path) -> None:
    source = _make_repo(tmp_path)
    service = SandboxService(
        workspace_provider=LocalGitWorkspaceProvider(tmp_path / ".repos"),
        runtime_provider=LocalSubprocessRuntimeProvider(allow_write=False),
        store=JsonSandboxStore(tmp_path / "metadata.json"),
        locks=InMemoryLockManager(),
    )
    workspace = await service.get_or_create_workspace(
        WorkspaceRequest(
            user_id="u1",
            project_id="p1",
            repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="c1",
            create_branch=True,
        )
    )

    with pytest.raises(RuntimeCommandRejected):
        await service.exec(
            workspace.id,
            ExecRequest(
                cmd=("sh", "-c", "echo changed > README.md"),
                command_kind=CommandKind.WRITE,
            ),
        )


@pytest.mark.asyncio
async def test_stale_workspace_from_other_provider_is_dropped(tmp_path: Path) -> None:
    """Switching providers between runs must not crash mount_for_runtime.

    The store is provider-agnostic, but a workspace created by provider A has
    a ``WorkspaceLocation`` that's incompatible with provider B's
    ``mount_for_runtime``. The service should treat such entries as stale,
    drop them, and have the current provider create a fresh workspace.
    """
    from sandbox.domain.models import (
        Workspace,
        WorkspaceLocation,
        WorkspaceState,
        WorkspaceStorageKind,
        new_id,
    )

    source = _make_repo(tmp_path)
    metadata = tmp_path / "metadata.json"

    # First run: create a workspace with the local provider, persist it.
    local_provider = LocalGitWorkspaceProvider(tmp_path / ".repos")
    store = JsonSandboxStore(metadata)
    request = WorkspaceRequest(
        user_id="u1",
        project_id="p1",
        repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
        base_ref="main",
        mode=WorkspaceMode.EDIT,
        conversation_id="c1",
        create_branch=True,
    )
    stale = await local_provider.get_or_create_workspace(request)
    await store.save_workspace(stale)

    # Second run: a different provider with `kind="other-provider"`. The
    # service must invoke it (not silently hand back the stale entry) and the
    # stored workspace must be replaced.
    class _OtherProvider:
        kind = "other-provider"

        def __init__(self) -> None:
            self.calls = 0

        async def get_or_create_workspace(self, req: WorkspaceRequest) -> Workspace:
            self.calls += 1
            return Workspace(
                id=new_id("ws"),
                key=req.key(),
                repo_cache_id=None,
                request=req,
                location=WorkspaceLocation(kind=WorkspaceStorageKind.REMOTE_PATH, remote_path="/work"),
                backend_kind=self.kind,
                state=WorkspaceState.READY,
            )

        async def get_workspace(self, workspace_id: str) -> Workspace | None:
            return None

        async def delete_workspace(self, workspace: Workspace) -> None:
            return None

        async def mount_for_runtime(self, workspace: Workspace, *, writable: bool):
            from sandbox.domain.models import Mount
            return Mount(source="/work", target="/work", writable=writable)

    other = _OtherProvider()
    service = SandboxService(
        workspace_provider=other,  # type: ignore[arg-type]
        runtime_provider=LocalSubprocessRuntimeProvider(),
        store=JsonSandboxStore(metadata),
        locks=InMemoryLockManager(),
    )

    fresh = await service.get_or_create_workspace(request)

    assert other.calls == 1, "current provider must build the workspace"
    assert fresh.backend_kind == "other-provider"
    assert fresh.id != stale.id

    # The store now holds only the fresh workspace under that key.
    reload_store = JsonSandboxStore(metadata)
    found = await reload_store.find_workspace_by_key(request.key())
    assert found is not None
    assert found.id == fresh.id
    assert found.backend_kind == "other-provider"


@pytest.mark.asyncio
async def test_json_store_round_trips_workspace(tmp_path: Path) -> None:
    source = _make_repo(tmp_path)
    metadata = tmp_path / "metadata.json"
    provider = LocalGitWorkspaceProvider(tmp_path / ".repos")
    store = JsonSandboxStore(metadata)
    workspace = await provider.get_or_create_workspace(
        WorkspaceRequest(
            user_id="u1",
            project_id="p1",
            repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
            base_ref="main",
            mode=WorkspaceMode.ANALYSIS,
        )
    )
    await store.save_workspace(workspace)

    reloaded = JsonSandboxStore(metadata)
    found = await reloaded.find_workspace_by_key(workspace.key)

    assert found is not None
    assert found.id == workspace.id
    assert found.location.local_path == workspace.location.local_path


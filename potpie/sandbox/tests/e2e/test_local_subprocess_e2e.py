"""End-to-end coverage of `SandboxService` against the local subprocess runtime.

These tests do not need Docker or any managed sandbox backend, so they are the
fastest way to catch service-level regressions across:

- workspace creation, idempotent reuse, persistence in the JSON store
- runtime attach, exec, mutation, hibernate, restart
- destroy semantics for runtime vs workspace
- write rejection when a runtime is created read-only
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from sandbox.adapters.outbound.file.json_store import JsonSandboxStore
from sandbox.adapters.outbound.local.git_workspace import LocalGitWorkspaceProvider
from sandbox.adapters.outbound.local.subprocess_runtime import LocalSubprocessRuntimeProvider
from sandbox.adapters.outbound.memory.locks import InMemoryLockManager
from sandbox.application.services.sandbox_service import SandboxService
from sandbox.domain.errors import RuntimeCommandRejected, WorkspaceNotFound
from sandbox.domain.models import (
    CommandKind,
    ExecRequest,
    RepoIdentity,
    RuntimeRequest,
    RuntimeState,
    WorkspaceMode,
    WorkspaceRequest,
)


def _edit_request(source: Path, *, conversation_id: str = "c1") -> WorkspaceRequest:
    return WorkspaceRequest(
        user_id="u1",
        project_id="p1",
        repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
        base_ref="main",
        mode=WorkspaceMode.EDIT,
        conversation_id=conversation_id,
        create_branch=True,
    )


@pytest.mark.asyncio
async def test_full_flow_create_exec_mutate_destroy(
    source_repo: Path, local_service: SandboxService
) -> None:
    workspace = await local_service.get_or_create_workspace(_edit_request(source_repo))

    read = await local_service.exec(
        workspace.id, ExecRequest(cmd=("cat", "README.md"), command_kind=CommandKind.READ)
    )
    assert read.exit_code == 0
    assert read.stdout == b"hello e2e\n"

    write = await local_service.exec(
        workspace.id,
        ExecRequest(
            cmd=("sh", "-c", "printf changed > generated.txt"),
            command_kind=CommandKind.WRITE,
        ),
    )
    assert write.exit_code == 0

    refreshed = await local_service.get_workspace(workspace.id)
    assert refreshed.dirty is True

    verify = await local_service.exec(
        workspace.id,
        ExecRequest(cmd=("cat", "generated.txt"), command_kind=CommandKind.READ),
    )
    assert verify.stdout == b"changed"

    await local_service.destroy_workspace(workspace.id)
    with pytest.raises(WorkspaceNotFound):
        await local_service.get_workspace(workspace.id)
    assert not Path(workspace.location.local_path).exists()


@pytest.mark.asyncio
async def test_workspace_is_idempotent_and_persisted(
    source_repo: Path,
    workspace_provider: LocalGitWorkspaceProvider,
    metadata_path: Path,
) -> None:
    store = JsonSandboxStore(metadata_path)
    service = SandboxService(
        workspace_provider=workspace_provider,
        runtime_provider=LocalSubprocessRuntimeProvider(allow_write=True),
        store=store,
        locks=InMemoryLockManager(),
    )
    request = _edit_request(source_repo, conversation_id="persist-1")

    first = await service.get_or_create_workspace(request)
    second = await service.get_or_create_workspace(request)

    assert first.id == second.id

    reloaded = JsonSandboxStore(metadata_path)
    found = await reloaded.find_workspace_by_key(first.key)
    assert found is not None
    assert found.id == first.id
    assert found.location.local_path == first.location.local_path


@pytest.mark.asyncio
async def test_runtime_lifecycle_hibernate_then_resume(
    source_repo: Path, local_service: SandboxService
) -> None:
    workspace = await local_service.get_or_create_workspace(
        _edit_request(source_repo, conversation_id="lifecycle")
    )
    runtime = await local_service.get_or_create_runtime(
        RuntimeRequest(workspace_id=workspace.id, writable=True)
    )

    await local_service.hibernate_runtime(runtime.id)
    stopped = await local_service._store.get_runtime(runtime.id)
    assert stopped.state is RuntimeState.STOPPED

    result = await local_service.exec(
        workspace.id, ExecRequest(cmd=("cat", "README.md"), command_kind=CommandKind.READ)
    )
    assert result.exit_code == 0

    resumed = await local_service._store.get_runtime(runtime.id)
    assert resumed.state is RuntimeState.RUNNING

    await local_service.destroy_runtime(runtime.id)
    assert await local_service._store.get_runtime(runtime.id) is None


@pytest.mark.asyncio
async def test_destroy_runtime_keeps_workspace_alive(
    source_repo: Path, local_service: SandboxService
) -> None:
    workspace = await local_service.get_or_create_workspace(
        _edit_request(source_repo, conversation_id="keep-ws")
    )
    runtime = await local_service.get_or_create_runtime(
        RuntimeRequest(workspace_id=workspace.id)
    )

    await local_service.destroy_runtime(runtime.id)

    still_there = await local_service.get_workspace(workspace.id)
    assert still_there.id == workspace.id
    assert Path(still_there.location.local_path).exists()

    new_runtime = await local_service.get_or_create_runtime(
        RuntimeRequest(workspace_id=workspace.id)
    )
    assert new_runtime.id != runtime.id


@pytest.mark.asyncio
async def test_readonly_runtime_blocks_writes(
    source_repo: Path,
    workspace_provider: LocalGitWorkspaceProvider,
    metadata_path: Path,
) -> None:
    service = SandboxService(
        workspace_provider=workspace_provider,
        runtime_provider=LocalSubprocessRuntimeProvider(allow_write=False),
        store=JsonSandboxStore(metadata_path),
        locks=InMemoryLockManager(),
    )
    workspace = await service.get_or_create_workspace(
        _edit_request(source_repo, conversation_id="readonly")
    )

    ok = await service.exec(
        workspace.id, ExecRequest(cmd=("cat", "README.md"), command_kind=CommandKind.READ)
    )
    assert ok.exit_code == 0

    with pytest.raises(RuntimeCommandRejected):
        await service.exec(
            workspace.id,
            ExecRequest(
                cmd=("sh", "-c", "echo nope > nope.txt"),
                command_kind=CommandKind.WRITE,
            ),
        )


@pytest.mark.asyncio
async def test_concurrent_writes_serialize_per_workspace(
    source_repo: Path, local_service: SandboxService
) -> None:
    """Two parallel write commands must not interleave inside one workspace.

    The service holds a per-workspace lock for mutating commands. We make each
    write append-then-sleep-then-append; if the lock works the file ends with
    `AABB` (or `BBAA`), never with interleaved `ABAB`.
    """
    workspace = await local_service.get_or_create_workspace(
        _edit_request(source_repo, conversation_id="concurrent")
    )
    await local_service.exec(
        workspace.id,
        ExecRequest(
            cmd=("sh", "-c", "printf '' > log.txt"),
            command_kind=CommandKind.WRITE,
        ),
    )

    async def append(token: str) -> None:
        await local_service.exec(
            workspace.id,
            ExecRequest(
                cmd=(
                    "sh",
                    "-c",
                    f"printf {token} >> log.txt && sleep 0.2 && printf {token} >> log.txt",
                ),
                command_kind=CommandKind.WRITE,
            ),
        )

    await asyncio.gather(append("A"), append("B"))

    result = await local_service.exec(
        workspace.id, ExecRequest(cmd=("cat", "log.txt"), command_kind=CommandKind.READ)
    )
    assert result.stdout in {b"AABB", b"BBAA"}, result.stdout


@pytest.mark.asyncio
async def test_exec_timeout_returns_timed_out(
    source_repo: Path, local_service: SandboxService
) -> None:
    workspace = await local_service.get_or_create_workspace(
        _edit_request(source_repo, conversation_id="timeout")
    )
    result = await local_service.exec(
        workspace.id,
        ExecRequest(
            cmd=("sh", "-c", "sleep 5"),
            command_kind=CommandKind.READ,
            timeout_s=1,
        ),
    )
    assert result.timed_out is True
    assert result.exit_code == 124


@pytest.mark.asyncio
async def test_max_output_bytes_truncates(
    source_repo: Path, local_service: SandboxService
) -> None:
    workspace = await local_service.get_or_create_workspace(
        _edit_request(source_repo, conversation_id="truncate")
    )
    result = await local_service.exec(
        workspace.id,
        ExecRequest(
            cmd=("sh", "-c", "yes hello | head -c 1024"),
            command_kind=CommandKind.READ,
            max_output_bytes=64,
        ),
    )
    assert result.truncated is True
    assert len(result.stdout) <= 64

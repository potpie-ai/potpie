"""Tests for the Capabilities dataclass and the acquire/release session API.

The doc's Edit Flow says SandboxService should orchestrate "ensure
cache → fork worktree → attach runtime" as a single coherent action.
``acquire_session`` covers the cache + workspace half; runtime
attachment stays separate because read-only callers don't need one.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from sandbox.adapters.outbound.file.json_store import JsonSandboxStore
from sandbox.adapters.outbound.local.git_workspace import LocalGitWorkspaceProvider
from sandbox.adapters.outbound.local.repo_cache import LocalRepoCacheProvider
from sandbox.adapters.outbound.local.subprocess_runtime import (
    LocalSubprocessRuntimeProvider,
)
from sandbox.adapters.outbound.memory.locks import InMemoryLockManager
from sandbox.adapters.outbound.memory.store import InMemorySandboxStore
from sandbox.application.services.sandbox_service import SandboxService
from sandbox.domain.models import (
    Capabilities,
    RepoIdentity,
    RuntimeState,
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
    (repo / "README.md").write_text("hi\n", encoding="utf-8")
    _run(["git", "add", "README.md"], repo)
    _run(["git", "commit", "-m", "initial"], repo)
    return repo


def _build_service(tmp_path: Path) -> SandboxService:
    cache_provider = LocalRepoCacheProvider(tmp_path / ".repos")
    return SandboxService(
        workspace_provider=LocalGitWorkspaceProvider(
            tmp_path / ".repos", repo_cache_provider=cache_provider
        ),
        runtime_provider=LocalSubprocessRuntimeProvider(allow_write=True),
        store=InMemorySandboxStore(),
        locks=InMemoryLockManager(),
        repo_cache_provider=cache_provider,
    )


def test_capabilities_default_is_read_only_and_persistent() -> None:
    caps = Capabilities()
    assert caps.writable is False
    assert caps.isolated is False
    assert caps.persistent is True


def test_capabilities_from_mode_maps_each_mode() -> None:
    analysis = Capabilities.from_mode(WorkspaceMode.ANALYSIS)
    assert analysis == Capabilities(writable=False, isolated=False, persistent=True)

    edit = Capabilities.from_mode(WorkspaceMode.EDIT)
    assert edit == Capabilities(writable=True, isolated=True, persistent=True)

    task = Capabilities.from_mode(WorkspaceMode.TASK)
    assert task == Capabilities(writable=True, isolated=True, persistent=True)


@pytest.mark.asyncio
async def test_workspace_carries_capabilities_from_mode(tmp_path: Path) -> None:
    source = _make_repo(tmp_path)
    cache_provider = LocalRepoCacheProvider(tmp_path / ".repos")
    provider = LocalGitWorkspaceProvider(
        tmp_path / ".repos", repo_cache_provider=cache_provider
    )
    workspace = await provider.get_or_create_workspace(
        WorkspaceRequest(
            user_id="u1",
            project_id="p1",
            repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="c1",
        )
    )
    assert workspace.capabilities.writable is True
    assert workspace.capabilities.isolated is True


@pytest.mark.asyncio
async def test_capabilities_round_trip_through_json_store(tmp_path: Path) -> None:
    source = _make_repo(tmp_path)
    cache_provider = LocalRepoCacheProvider(tmp_path / ".repos")
    ws_provider = LocalGitWorkspaceProvider(
        tmp_path / ".repos", repo_cache_provider=cache_provider
    )
    saver = JsonSandboxStore(tmp_path / "metadata.json")

    workspace = await ws_provider.get_or_create_workspace(
        WorkspaceRequest(
            user_id="u1",
            project_id="p1",
            repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
            base_ref="main",
            mode=WorkspaceMode.ANALYSIS,
        )
    )
    await saver.save_workspace(workspace)

    reloader = JsonSandboxStore(tmp_path / "metadata.json")
    reloaded = await reloader.find_workspace_by_key(workspace.key)
    assert reloaded is not None
    assert reloaded.capabilities == workspace.capabilities
    assert reloaded.capabilities.writable is False  # ANALYSIS


@pytest.mark.asyncio
async def test_acquire_session_provisions_repo_cache_first(tmp_path: Path) -> None:
    """acquire_session must persist a RepoCache row before the workspace."""
    source = _make_repo(tmp_path)
    service = _build_service(tmp_path)

    workspace = await service.acquire_session(
        WorkspaceRequest(
            user_id="u1",
            project_id="p1",
            repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="c1",
        )
    )
    assert workspace.repo_cache_id is not None

    # Repo cache row must be in the store, keyed by (host, repo).
    cache = await service._store.find_repo_cache_by_key("github.com|owner/repo")
    assert cache is not None
    assert cache.id == workspace.repo_cache_id


@pytest.mark.asyncio
async def test_release_session_hibernates_runtime_keeps_workspace(
    tmp_path: Path,
) -> None:
    source = _make_repo(tmp_path)
    service = _build_service(tmp_path)
    workspace = await service.acquire_session(
        WorkspaceRequest(
            user_id="u1",
            project_id="p1",
            repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="c1",
        )
    )
    # Force a runtime so we have something to release.
    from sandbox.domain.models import RuntimeRequest

    runtime = await service.get_or_create_runtime(RuntimeRequest(workspace.id))
    assert runtime.state is RuntimeState.RUNNING

    await service.release_session(workspace.id)

    # Runtime is stopped; workspace row still in the store.
    after = await service._store.get_runtime(runtime.id)
    assert after is not None
    assert after.state is RuntimeState.STOPPED
    persisted_ws = await service._store.get_workspace(workspace.id)
    assert persisted_ws is not None


@pytest.mark.asyncio
async def test_release_session_with_destroy_drops_the_runtime(tmp_path: Path) -> None:
    source = _make_repo(tmp_path)
    service = _build_service(tmp_path)
    workspace = await service.acquire_session(
        WorkspaceRequest(
            user_id="u1",
            project_id="p1",
            repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="c1",
        )
    )
    from sandbox.domain.models import RuntimeRequest

    runtime = await service.get_or_create_runtime(RuntimeRequest(workspace.id))
    await service.release_session(workspace.id, destroy_runtime=True)

    # Runtime row is gone but workspace persists.
    assert await service._store.get_runtime(runtime.id) is None
    assert await service._store.get_workspace(workspace.id) is not None


@pytest.mark.asyncio
async def test_release_session_is_noop_when_no_runtime_attached(
    tmp_path: Path,
) -> None:
    """A session that hasn't yet created a runtime can still be released.

    Read-only callers (search/analysis) often grab a workspace and never
    attach a runtime. Releasing such a session must be a clean no-op
    rather than a NotFound error.
    """
    source = _make_repo(tmp_path)
    service = _build_service(tmp_path)
    workspace = await service.acquire_session(
        WorkspaceRequest(
            user_id="u1",
            project_id="p1",
            repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
            base_ref="main",
            mode=WorkspaceMode.ANALYSIS,
        )
    )

    # No runtime created. Release should silently succeed.
    await service.release_session(workspace.id)

    persisted = await service._store.get_workspace(workspace.id)
    assert persisted is not None


@pytest.mark.asyncio
async def test_two_conversations_get_independent_branches(tmp_path: Path) -> None:
    """The canonical adapter must not pin both convs to the same git branch.

    Production already passes ``branch=base_branch`` for EDIT mode in the
    legacy bridge; without conversation-scoped branch derivation the
    canonical adapter would refuse the second worktree (git rejects two
    worktrees on the same ref). Test the auto-derive path so the cutover
    is safe.
    """
    source = _make_repo(tmp_path)
    service = _build_service(tmp_path)

    common = dict(
        user_id="u1",
        project_id="p1",
        repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
        base_ref="main",
        mode=WorkspaceMode.EDIT,
    )
    a = await service.acquire_session(
        WorkspaceRequest(**common, conversation_id="conv-a")  # type: ignore[arg-type]
    )
    b = await service.acquire_session(
        WorkspaceRequest(**common, conversation_id="conv-b")  # type: ignore[arg-type]
    )
    assert a.id != b.id
    a_branch = a.metadata.get("branch")
    b_branch = b.metadata.get("branch")
    assert a_branch != b_branch
    assert a.location.local_path != b.location.local_path

"""Tests for `sandbox.api.SandboxClient`.

These exercise the client over the in-process local-fs / subprocess providers
so we cover both the local fast path (handle.local_path != None) and the
exec-based fallback by patching the handle to drop `local_path`.
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from sandbox import (
    SandboxClient,
    SandboxContainer,
    SandboxService,
    WorkspaceMode,
)
from sandbox.adapters.outbound.file.json_store import JsonSandboxStore
from sandbox.adapters.outbound.local.git_workspace import LocalGitWorkspaceProvider
from sandbox.adapters.outbound.local.repo_cache import LocalRepoCacheProvider
from sandbox.adapters.outbound.local.subprocess_runtime import (
    LocalSubprocessRuntimeProvider,
)
from sandbox.adapters.outbound.memory.eviction import NoOpEvictionPolicy
from sandbox.adapters.outbound.memory.locks import InMemoryLockManager
from sandbox.api.client import SandboxOpError
from sandbox.api.types import WorkspaceHandle
from sandbox.domain.errors import InvalidWorkspacePath


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
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("print('alive')\n", encoding="utf-8")
    _run(["git", "add", "-A"], repo)
    _run(["git", "commit", "-m", "initial"], repo)
    return repo


def _build_client(tmp_path: Path) -> SandboxClient:
    cache_provider = LocalRepoCacheProvider(tmp_path / ".repos")
    workspace_provider = LocalGitWorkspaceProvider(
        tmp_path / ".repos", repo_cache_provider=cache_provider
    )
    runtime_provider = LocalSubprocessRuntimeProvider(allow_write=True)
    store = JsonSandboxStore(tmp_path / "metadata.json")
    locks = InMemoryLockManager()
    service = SandboxService(
        workspace_provider=workspace_provider,
        runtime_provider=runtime_provider,
        store=store,
        locks=locks,
        repo_cache_provider=cache_provider,
    )
    container = SandboxContainer(
        workspace_provider=workspace_provider,
        runtime_provider=runtime_provider,
        store=store,
        locks=locks,
        service=service,
        eviction=NoOpEvictionPolicy(),
        repo_cache_provider=cache_provider,
    )
    return SandboxClient.from_container(container)


@pytest.mark.asyncio
async def test_get_workspace_returns_handle_with_local_path(tmp_path: Path) -> None:
    source = _make_repo(tmp_path)
    client = _build_client(tmp_path)
    handle = await client.get_workspace(
        user_id="u1",
        project_id="p1",
        repo="owner/repo",
        repo_url=str(source),
        branch="feat/x",
        base_ref="main",
        create_branch=True,
        mode=WorkspaceMode.EDIT,
        conversation_id="c1",
    )
    assert isinstance(handle, WorkspaceHandle)
    assert handle.backend_kind == "local"
    assert handle.branch == "feat/x"
    assert handle.local_path is not None
    assert (Path(handle.local_path) / "README.md").exists()


@pytest.mark.asyncio
async def test_get_workspace_idempotent_per_key(tmp_path: Path) -> None:
    source = _make_repo(tmp_path)
    client = _build_client(tmp_path)
    kwargs = dict(
        user_id="u1",
        project_id="p1",
        repo="owner/repo",
        repo_url=str(source),
        branch="feat/x",
        base_ref="main",
        create_branch=True,
        mode=WorkspaceMode.EDIT,
        conversation_id="c1",
    )
    a = await client.get_workspace(**kwargs)
    b = await client.get_workspace(**kwargs)
    assert a.workspace_id == b.workspace_id


@pytest.mark.asyncio
async def test_read_file_local_path_fast_path(tmp_path: Path) -> None:
    source = _make_repo(tmp_path)
    client = _build_client(tmp_path)
    handle = await client.get_workspace(
        user_id="u1",
        project_id="p1",
        repo="owner/repo",
        repo_url=str(source),
        branch="main",
        base_ref="main",
        mode=WorkspaceMode.ANALYSIS,
    )
    data = await client.read_file(handle, "README.md")
    assert data == b"hello sandbox\n"


@pytest.mark.asyncio
async def test_read_file_via_exec_when_local_path_missing(tmp_path: Path) -> None:
    source = _make_repo(tmp_path)
    client = _build_client(tmp_path)
    handle = await client.get_workspace(
        user_id="u1",
        project_id="p1",
        repo="owner/repo",
        repo_url=str(source),
        branch="main",
        base_ref="main",
        mode=WorkspaceMode.ANALYSIS,
    )
    # Force the exec fallback by erasing local_path.
    exec_handle = dataclasses.replace(handle, local_path=None)
    data = await client.read_file(exec_handle, "README.md")
    assert data == b"hello sandbox\n"


@pytest.mark.asyncio
async def test_write_file_then_read_back(tmp_path: Path) -> None:
    source = _make_repo(tmp_path)
    client = _build_client(tmp_path)
    handle = await client.get_workspace(
        user_id="u1",
        project_id="p1",
        repo="owner/repo",
        repo_url=str(source),
        branch="feat/x",
        base_ref="main",
        create_branch=True,
        mode=WorkspaceMode.EDIT,
        conversation_id="c1",
    )
    await client.write_file(handle, "src/new.py", "print('new')\n")
    assert (Path(handle.local_path) / "src" / "new.py").read_text() == "print('new')\n"

    # Round-trip via exec backend too.
    exec_handle = dataclasses.replace(handle, local_path=None)
    data = await client.read_file(exec_handle, "src/new.py")
    assert data == b"print('new')\n"


@pytest.mark.asyncio
async def test_list_dir(tmp_path: Path) -> None:
    source = _make_repo(tmp_path)
    client = _build_client(tmp_path)
    handle = await client.get_workspace(
        user_id="u1",
        project_id="p1",
        repo="owner/repo",
        repo_url=str(source),
        branch="main",
        base_ref="main",
        mode=WorkspaceMode.ANALYSIS,
    )
    entries = await client.list_dir(handle, ".")
    names = {e.name: e.is_dir for e in entries}
    assert names.get("README.md") is False
    assert names.get("src") is True


@pytest.mark.asyncio
async def test_list_dir_via_exec(tmp_path: Path) -> None:
    source = _make_repo(tmp_path)
    client = _build_client(tmp_path)
    handle = await client.get_workspace(
        user_id="u1",
        project_id="p1",
        repo="owner/repo",
        repo_url=str(source),
        branch="main",
        base_ref="main",
        mode=WorkspaceMode.ANALYSIS,
    )
    exec_handle = dataclasses.replace(handle, local_path=None)
    entries = await client.list_dir(exec_handle, ".")
    names = {e.name: e.is_dir for e in entries}
    assert names.get("README.md") is False
    assert names.get("src") is True


@pytest.mark.asyncio
async def test_exec_runs_command(tmp_path: Path) -> None:
    source = _make_repo(tmp_path)
    client = _build_client(tmp_path)
    handle = await client.get_workspace(
        user_id="u1",
        project_id="p1",
        repo="owner/repo",
        repo_url=str(source),
        branch="main",
        base_ref="main",
        mode=WorkspaceMode.ANALYSIS,
    )
    result = await client.exec(handle, ["cat", "README.md"])
    assert result.exit_code == 0
    assert result.stdout == b"hello sandbox\n"


@pytest.mark.asyncio
async def test_status_clean_after_create(tmp_path: Path) -> None:
    source = _make_repo(tmp_path)
    client = _build_client(tmp_path)
    handle = await client.get_workspace(
        user_id="u1",
        project_id="p1",
        repo="owner/repo",
        repo_url=str(source),
        branch="feat/x",
        base_ref="main",
        create_branch=True,
        mode=WorkspaceMode.EDIT,
        conversation_id="c1",
    )
    status = await client.status(handle)
    assert status.is_clean is True
    assert status.branch == "feat/x"
    assert status.staged == ()
    assert status.unstaged == ()
    assert status.untracked == ()


@pytest.mark.asyncio
async def test_status_after_write(tmp_path: Path) -> None:
    source = _make_repo(tmp_path)
    client = _build_client(tmp_path)
    handle = await client.get_workspace(
        user_id="u1",
        project_id="p1",
        repo="owner/repo",
        repo_url=str(source),
        branch="feat/x",
        base_ref="main",
        create_branch=True,
        mode=WorkspaceMode.EDIT,
        conversation_id="c1",
    )
    await client.write_file(handle, "README.md", "changed\n")
    await client.write_file(handle, "src/new.py", "print('new')\n")
    status = await client.status(handle)
    assert status.is_clean is False
    assert "README.md" in status.unstaged
    assert "src/new.py" in status.untracked


@pytest.mark.asyncio
async def test_commit_returns_sha(tmp_path: Path) -> None:
    source = _make_repo(tmp_path)
    client = _build_client(tmp_path)
    handle = await client.get_workspace(
        user_id="u1",
        project_id="p1",
        repo="owner/repo",
        repo_url=str(source),
        branch="feat/x",
        base_ref="main",
        create_branch=True,
        mode=WorkspaceMode.EDIT,
        conversation_id="c1",
    )
    await client.write_file(handle, "src/new.py", "print('new')\n")
    sha = await client.commit(
        handle,
        "add new file",
        author=("Bot", "bot@example.com"),
    )
    assert len(sha) == 40
    status = await client.status(handle)
    assert status.is_clean


@pytest.mark.asyncio
async def test_diff_after_change(tmp_path: Path) -> None:
    source = _make_repo(tmp_path)
    client = _build_client(tmp_path)
    handle = await client.get_workspace(
        user_id="u1",
        project_id="p1",
        repo="owner/repo",
        repo_url=str(source),
        branch="feat/x",
        base_ref="main",
        create_branch=True,
        mode=WorkspaceMode.EDIT,
        conversation_id="c1",
    )
    await client.write_file(handle, "README.md", "changed\n")
    diff = await client.diff(handle)
    assert "-hello sandbox" in diff
    assert "+changed" in diff


@pytest.mark.asyncio
async def test_search_finds_match_when_rg_available(tmp_path: Path) -> None:
    if shutil.which("rg") is None:
        pytest.skip("ripgrep not on PATH; skipping search test")
    source = _make_repo(tmp_path)
    client = _build_client(tmp_path)
    handle = await client.get_workspace(
        user_id="u1",
        project_id="p1",
        repo="owner/repo",
        repo_url=str(source),
        branch="main",
        base_ref="main",
        mode=WorkspaceMode.ANALYSIS,
    )
    hits = await client.search(handle, "alive")
    assert any("app.py" in h.path and "alive" in h.snippet for h in hits)


@pytest.mark.asyncio
async def test_release_workspace_hibernates_runtime(tmp_path: Path) -> None:
    source = _make_repo(tmp_path)
    client = _build_client(tmp_path)
    handle = await client.get_workspace(
        user_id="u1",
        project_id="p1",
        repo="owner/repo",
        repo_url=str(source),
        branch="main",
        base_ref="main",
        mode=WorkspaceMode.ANALYSIS,
    )
    await client.exec(handle, ["true"])  # bring runtime up
    runtime = await client.container.store.find_runtime_by_workspace(
        handle.workspace_id
    )
    assert runtime is not None and runtime.state.value == "running"
    await client.release_workspace(handle)
    runtime_after = await client.container.store.find_runtime_by_workspace(
        handle.workspace_id
    )
    assert runtime_after is not None
    assert runtime_after.state.value == "stopped"


@pytest.mark.asyncio
async def test_destroy_workspace_removes_local_path(tmp_path: Path) -> None:
    source = _make_repo(tmp_path)
    client = _build_client(tmp_path)
    handle = await client.get_workspace(
        user_id="u1",
        project_id="p1",
        repo="owner/repo",
        repo_url=str(source),
        branch="feat/x",
        base_ref="main",
        create_branch=True,
        mode=WorkspaceMode.EDIT,
        conversation_id="c1",
    )
    assert handle.local_path is not None
    path = Path(handle.local_path)
    assert path.exists()
    await client.destroy_workspace(handle)
    assert not path.exists()


@pytest.mark.parametrize(
    "bad",
    ["/etc/passwd", "..", "../escape", "src/../../escape"],
)
def test_validate_relpath_rejects_unsafe(bad: str) -> None:
    from sandbox.api.client import _validate_relpath

    with pytest.raises(InvalidWorkspacePath):
        _validate_relpath(bad)


def test_validate_relpath_rejects_empty_unless_allow_dot() -> None:
    from sandbox.api.client import _validate_relpath

    with pytest.raises(InvalidWorkspacePath):
        _validate_relpath("")
    assert _validate_relpath("", allow_dot=True) == "."
    assert _validate_relpath(".", allow_dot=True) == "."


@pytest.mark.asyncio
async def test_read_file_rejects_absolute(tmp_path: Path) -> None:
    source = _make_repo(tmp_path)
    client = _build_client(tmp_path)
    handle = await client.get_workspace(
        user_id="u1",
        project_id="p1",
        repo="owner/repo",
        repo_url=str(source),
        branch="main",
        base_ref="main",
        mode=WorkspaceMode.ANALYSIS,
    )
    with pytest.raises(InvalidWorkspacePath):
        await client.read_file(handle, "/etc/passwd")


@pytest.mark.asyncio
async def test_read_file_missing_raises(tmp_path: Path) -> None:
    source = _make_repo(tmp_path)
    client = _build_client(tmp_path)
    handle = await client.get_workspace(
        user_id="u1",
        project_id="p1",
        repo="owner/repo",
        repo_url=str(source),
        branch="main",
        base_ref="main",
        mode=WorkspaceMode.ANALYSIS,
    )
    with pytest.raises(SandboxOpError):
        await client.read_file(handle, "does-not-exist.txt")


@pytest.mark.asyncio
async def test_ensure_repo_cache_persists_and_dedupes(tmp_path: Path) -> None:
    """First call clones + saves a row; second call hits the store and
    returns the same id without rebuilding the workspace state."""
    source = _make_repo(tmp_path)
    client = _build_client(tmp_path)

    cache = await client.ensure_repo_cache(
        user_id="u1",
        repo="owner/repo",
        base_ref="main",
        repo_url=str(source),
    )
    assert cache.id.startswith("rc_")
    assert cache.key == "github.com|owner/repo"
    assert cache.location.local_path is not None
    assert (Path(cache.location.local_path) / "HEAD").exists()

    again = await client.ensure_repo_cache(
        user_id="u1",
        repo="owner/repo",
        base_ref="main",
        repo_url=str(source),
    )
    assert again.id == cache.id, "service must dedupe by key"


@pytest.mark.asyncio
async def test_search_anchors_path_to_remote_root(tmp_path: Path) -> None:
    """``search(..., path=...)`` must prepend ``handle.remote_path``.

    All other helpers (``read_file``, ``write_file``, ``list_dir``) anchor
    relative paths to ``handle.remote_path`` via ``_posix_join``. ``search``
    historically did not — pinned here so it stays consistent and
    doesn't silently search the runtime's CWD if that ever diverges.

    We don't need ripgrep installed: we patch the client's ``exec`` to
    capture the actual command and assert the path arg.
    """
    from sandbox import Capabilities
    from sandbox.api.client import _posix_join
    from sandbox.domain.models import ExecResult

    source = _make_repo(tmp_path)
    client = _build_client(tmp_path)
    handle_local = await client.get_workspace(
        user_id="u1",
        project_id="p1",
        repo="owner/repo",
        repo_url=str(source),
        branch="main",
        base_ref="main",
        mode=WorkspaceMode.ANALYSIS,
    )
    # Force the exec backend by stripping local_path AND injecting a
    # remote_path so the helper has something to anchor against.
    handle = WorkspaceHandle(
        workspace_id=handle_local.workspace_id,
        branch=handle_local.branch,
        backend_kind=handle_local.backend_kind,
        local_path=None,
        remote_path="/remote/work",
        capabilities=Capabilities.from_mode(WorkspaceMode.ANALYSIS),
    )

    captured: dict[str, list[str]] = {}

    async def fake_exec(h: WorkspaceHandle, cmd: list[str], **kwargs):
        captured["cmd"] = list(cmd)
        return ExecResult(exit_code=1, stdout=b"", stderr=b"")  # 1 = no matches

    # Patch the bound method on this client instance only.
    client.exec = fake_exec  # type: ignore[assignment]

    await client.search(handle, "needle", path="src")

    assert "cmd" in captured, "search did not invoke exec"
    last_arg = captured["cmd"][-1]
    expected = _posix_join("/remote/work", "src")
    assert last_arg == expected, (
        f"search path arg should be anchored at remote_root: "
        f"expected {expected!r}, got {last_arg!r}"
    )


@pytest.mark.asyncio
async def test_write_file_after_destroy_raises(tmp_path: Path) -> None:
    """A handle held across ``destroy_workspace`` must NOT silently
    recreate the worktree on the next ``write_file`` call.

    The fix: the write fast path re-resolves the workspace through the
    service, so a destroyed workspace surfaces as ``WorkspaceNotFound``
    instead of letting ``_write_bytes`` mkdir+write into a stale path.
    """
    from sandbox.domain.errors import WorkspaceNotFound

    source = _make_repo(tmp_path)
    client = _build_client(tmp_path)
    handle = await client.get_workspace(
        user_id="u1",
        project_id="p1",
        repo="owner/repo",
        repo_url=str(source),
        branch="feat/x",
        base_ref="main",
        create_branch=True,
        mode=WorkspaceMode.EDIT,
        conversation_id="c1",
    )
    assert handle.local_path is not None
    worktree = Path(handle.local_path)
    assert worktree.exists()

    await client.destroy_workspace(handle)
    assert not worktree.exists()

    with pytest.raises(WorkspaceNotFound):
        await client.write_file(handle, "leaked.txt", b"should not land")

    assert not worktree.exists(), (
        "destroyed worktree must not be recreated by a stale handle write"
    )

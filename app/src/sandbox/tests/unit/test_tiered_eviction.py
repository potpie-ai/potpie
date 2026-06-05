from __future__ import annotations

import subprocess
from datetime import timedelta
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from sandbox.adapters.outbound.local.git_workspace import LocalGitWorkspaceProvider
from sandbox.adapters.outbound.local.repo_cache import LocalRepoCacheProvider
from sandbox.adapters.outbound.local.subprocess_runtime import (
    LocalSubprocessRuntimeProvider,
)
from sandbox.adapters.outbound.memory.eviction import TieredVolumeEvictionPolicy
from sandbox.adapters.outbound.memory.locks import InMemoryLockManager
from sandbox.adapters.outbound.memory.store import InMemorySandboxStore
from sandbox.application.services.sandbox_service import SandboxService
from sandbox.domain.models import (
    RepoCache,
    RepoIdentity,
    Workspace,
    WorkspaceLocation,
    WorkspaceMode,
    WorkspaceRequest,
    WorkspaceState,
    WorkspaceStorageKind,
    new_id,
    utc_now,
)
from sandbox.domain.ports.storage import StorageStatus


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeInspector:
    kind = "local"

    def __init__(self) -> None:
        self.statuses: list[StorageStatus] = []

    async def status(self, *, user_id: str | None = None):
        return list(self.statuses)


class _FakeReaper:
    """Records calls and mutates the in-memory store like the real service."""

    def __init__(self, store: InMemorySandboxStore) -> None:
        self._store = store
        self.destroyed_workspaces: list[str] = []
        self.deleted_caches: list[str] = []
        self.destroyed_pots: list[tuple[str, str]] = []

    async def destroy_workspace(
        self, workspace_id: str, *, destroy_runtime: bool = True
    ) -> None:
        self.destroyed_workspaces.append(workspace_id)
        await self._store.delete_workspace(workspace_id)

    async def delete_repo_cache_if_unreferenced(self, cache_id: str) -> bool:
        for ws in await self._store.list_workspaces():
            if ws.repo_cache_id == cache_id:
                return False
        self.deleted_caches.append(cache_id)
        await self._store.delete_repo_cache(cache_id)
        return True

    async def destroy_pot_container(
        self,
        *,
        user_id: str,
        project_id: str,
        delete_repo_caches: bool = False,
    ) -> dict[str, int]:
        self.destroyed_pots.append((user_id, project_id))
        n = 0
        for ws in list(await self._store.list_workspaces()):
            if ws.request.user_id == user_id and ws.request.project_id == project_id:
                await self._store.delete_workspace(ws.id)
                n += 1
        return {"workspaces": n, "repo_caches": 0}


async def _mk_ws(
    store: InMemorySandboxStore,
    *,
    mode: WorkspaceMode,
    dirty: bool = False,
    size: int = 10,
    age_s: float = 0.0,
    user: str = "u1",
    project: str = "p1",
    scope: str = "s",
    pinned: bool = False,
    repo_cache_id: str | None = None,
) -> Workspace:
    req = WorkspaceRequest(
        user_id=user,
        project_id=project,
        repo=RepoIdentity(repo_name="owner/repo"),
        base_ref="main",
        mode=mode,
        conversation_id=scope if mode is WorkspaceMode.EDIT else None,
        task_id=scope if mode is WorkspaceMode.TASK else None,
    )
    ws = Workspace(
        id=new_id("ws"),
        key=req.key(),
        repo_cache_id=repo_cache_id,
        request=req,
        location=WorkspaceLocation(
            kind=WorkspaceStorageKind.LOCAL_PATH, local_path="/tmp/x"
        ),
        backend_kind="local",
        state=WorkspaceState.READY,
        dirty=dirty,
        size_bytes=size,
        last_used_at=utc_now() - timedelta(seconds=age_s),
        pinned_until=(utc_now() + timedelta(hours=1)) if pinned else None,
    )
    await store.save_workspace(ws)
    return ws


def _host_status(used: int, limit: int) -> StorageStatus:
    return StorageStatus(
        backend_kind="local", scope="host", used_bytes=used, limit_bytes=limit
    )


def _policy(store, inspector, reaper, *, evict_dirty=True):
    p = TieredVolumeEvictionPolicy(
        store=store,
        inspector=inspector,
        high_water=0.85,
        low_water=0.70,
        evict_dirty=evict_dirty,
    )
    p.bind_reaper(reaper)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_breach_evicts_nothing() -> None:
    store = InMemorySandboxStore()
    inspector = _FakeInspector()
    inspector.statuses = [_host_status(used=10, limit=100)]  # 10% — fine
    reaper = _FakeReaper(store)
    await _mk_ws(store, mode=WorkspaceMode.ANALYSIS)

    result = await _policy(store, inspector, reaper).evict_if_needed()

    assert result.is_empty
    assert reaper.destroyed_workspaces == []


@pytest.mark.asyncio
async def test_tier_order_clean_analysis_task_edit_then_spares_dirty() -> None:
    store = InMemorySandboxStore()
    inspector = _FakeInspector()
    # used=100/limit=100 ⇒ over 0.85; reclaim to 0.70 ⇒ free 30.
    inspector.statuses = [_host_status(used=100, limit=100)]
    reaper = _FakeReaper(store)

    edit_dirty = await _mk_ws(
        store, mode=WorkspaceMode.EDIT, dirty=True, scope="d", age_s=999
    )
    edit = await _mk_ws(store, mode=WorkspaceMode.EDIT, scope="e", age_s=10)
    task = await _mk_ws(store, mode=WorkspaceMode.TASK, scope="t", age_s=10)
    analysis = await _mk_ws(store, mode=WorkspaceMode.ANALYSIS, scope="a", age_s=10)

    result = await _policy(store, inspector, reaper).evict_if_needed()

    # 3 clean (10 each) = 30 → exactly low-water; dirty spared despite
    # being the oldest, because it is the last-resort tier.
    assert reaper.destroyed_workspaces == [analysis.id, task.id, edit.id]
    assert edit_dirty.id not in reaper.destroyed_workspaces
    assert result.freed_bytes == 30


@pytest.mark.asyncio
async def test_lru_within_tier_and_pinned_and_exclude_key_are_spared() -> None:
    store = InMemorySandboxStore()
    inspector = _FakeInspector()
    inspector.statuses = [_host_status(used=100, limit=100)]  # free 30
    reaper = _FakeReaper(store)

    # EDIT mode so each gets a distinct key (keyed on conversation_id);
    # ANALYSIS keys collapse onto base_ref and would collide.
    old = await _mk_ws(store, mode=WorkspaceMode.EDIT, scope="old", age_s=900)
    pinned = await _mk_ws(
        store, mode=WorkspaceMode.EDIT, scope="pin", age_s=800, pinned=True
    )
    excluded = await _mk_ws(store, mode=WorkspaceMode.EDIT, scope="exc", age_s=700)
    newish = await _mk_ws(store, mode=WorkspaceMode.EDIT, scope="new", age_s=1)

    result = await _policy(store, inspector, reaper).evict_if_needed(
        exclude_key=excluded.key
    )

    # pinned + exclude_key never touched; remaining evicted LRU-first.
    assert pinned.id not in reaper.destroyed_workspaces
    assert excluded.id not in reaper.destroyed_workspaces
    assert reaper.destroyed_workspaces == [old.id, newish.id]
    assert result.freed_bytes == 20  # only 2 evictable, can't reach 30


@pytest.mark.asyncio
async def test_orphan_cache_evicted_but_referenced_cache_kept() -> None:
    store = InMemorySandboxStore()
    inspector = _FakeInspector()
    inspector.statuses = [_host_status(used=100, limit=100)]  # free 30
    reaper = _FakeReaper(store)

    referenced = RepoCache(
        id="rc_ref",
        key="github.com|owner/repo",
        repo=RepoIdentity(repo_name="owner/repo"),
        location=WorkspaceLocation(kind=WorkspaceStorageKind.LOCAL_PATH),
        backend_kind="local",
        size_bytes=100,
    )
    orphan = RepoCache(
        id="rc_orphan",
        key="github.com|owner/other",
        repo=RepoIdentity(repo_name="owner/other"),
        location=WorkspaceLocation(kind=WorkspaceStorageKind.LOCAL_PATH),
        backend_kind="local",
        size_bytes=50,
        last_used_at=utc_now() - timedelta(seconds=500),
    )
    await store.save_repo_cache(referenced)
    await store.save_repo_cache(orphan)
    # A live workspace pins ``referenced``; no clean workspace candidates
    # so tiers 1–3 free nothing and the policy falls to tier 4 (caches).
    await _mk_ws(store, mode=WorkspaceMode.EDIT, dirty=True, repo_cache_id="rc_ref")

    result = await _policy(
        store, inspector, reaper, evict_dirty=False
    ).evict_if_needed()

    assert reaper.deleted_caches == ["rc_orphan"]
    assert result.evicted_repo_cache_ids == ("rc_orphan",)
    assert result.freed_bytes == 50


@pytest.mark.asyncio
async def test_last_resort_dirty_only_when_enabled() -> None:
    inspector = _FakeInspector()
    inspector.statuses = [_host_status(used=100, limit=100)]  # free 30

    # evict_dirty disabled ⇒ nothing happens, no exception.
    store = InMemorySandboxStore()
    reaper = _FakeReaper(store)
    await _mk_ws(store, mode=WorkspaceMode.EDIT, dirty=True, size=50)
    result = await _policy(
        store, inspector, reaper, evict_dirty=False
    ).evict_if_needed()
    assert reaper.destroyed_workspaces == []
    assert result.is_empty

    # evict_dirty enabled ⇒ the dirty workspace is the last resort.
    store2 = InMemorySandboxStore()
    reaper2 = _FakeReaper(store2)
    d = await _mk_ws(store2, mode=WorkspaceMode.EDIT, dirty=True, size=50)
    result2 = await _policy(store2, inspector, reaper2).evict_if_needed()
    assert reaper2.destroyed_workspaces == [d.id]
    assert result2.freed_bytes == 50


@pytest.mark.asyncio
async def test_count_scope_destroys_lru_project_not_excluded() -> None:
    store = InMemorySandboxStore()
    inspector = _FakeInspector()
    # 3 sandboxes, cap 3, low-water 0.70 ⇒ drop 1 project.
    inspector.statuses = [
        StorageStatus(
            backend_kind="local",
            scope="count:user:u1",
            used_bytes=3,
            limit_bytes=3,
        )
    ]
    reaper = _FakeReaper(store)
    # p_old is least-recently-used → should be the one destroyed.
    await _mk_ws(store, mode=WorkspaceMode.EDIT, project="p_old", scope="c1", age_s=999)
    await _mk_ws(store, mode=WorkspaceMode.EDIT, project="p_mid", scope="c2", age_s=500)
    excl = await _mk_ws(
        store, mode=WorkspaceMode.EDIT, project="p_new", scope="c3", age_s=1
    )

    result = await _policy(store, inspector, reaper).evict_if_needed(
        exclude_key=excl.key
    )

    assert reaper.destroyed_pots == [("u1", "p_old")]
    assert len(result.evicted_workspace_ids) == 1


@pytest.mark.asyncio
async def test_unbound_reaper_is_safe_noop() -> None:
    store = InMemorySandboxStore()
    inspector = _FakeInspector()
    inspector.statuses = [_host_status(used=100, limit=100)]
    await _mk_ws(store, mode=WorkspaceMode.ANALYSIS)

    policy = TieredVolumeEvictionPolicy(store=store, inspector=inspector)
    # No bind_reaper() → must not raise, must evict nothing.
    result = await policy.evict_if_needed()
    assert result.is_empty


# ---------------------------------------------------------------------------
# Integration: eviction is safe because rebuild is transparent
# ---------------------------------------------------------------------------
def _run(cmd: list[str], cwd: Path) -> None:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    repo.mkdir()
    _run(["git", "init", "-b", "main"], repo)
    _run(["git", "config", "user.email", "t@e.com"], repo)
    _run(["git", "config", "user.name", "T"], repo)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _run(["git", "add", "README.md"], repo)
    _run(["git", "commit", "-m", "initial"], repo)
    return repo


@pytest.mark.asyncio
async def test_evicted_workspace_is_transparently_rebuilt(
    tmp_path: Path,
) -> None:
    source = _make_repo(tmp_path)
    repos = tmp_path / ".repos"
    store = InMemorySandboxStore()
    cache_provider = LocalRepoCacheProvider(repos)
    ws_provider = LocalGitWorkspaceProvider(repos, repo_cache_provider=cache_provider)
    inspector = _FakeInspector()
    policy = TieredVolumeEvictionPolicy(
        store=store, inspector=inspector, high_water=0.85, low_water=0.70
    )
    service = SandboxService(
        workspace_provider=ws_provider,
        runtime_provider=LocalSubprocessRuntimeProvider(),
        store=store,
        locks=InMemoryLockManager(),
        repo_cache_provider=cache_provider,
        eviction=policy,
    )
    policy.bind_reaper(service)

    req = WorkspaceRequest(
        user_id="u1",
        project_id="p1",
        repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
        base_ref="main",
        mode=WorkspaceMode.EDIT,
        conversation_id="c1",
    )
    ws = await service.get_or_create_workspace(req)
    worktree = Path(ws.location.local_path or "")
    assert worktree.exists()

    # Force severe pressure and evict everything *except* nothing — the
    # exclude_key only protects the key being acquired, and we are not
    # acquiring here, so the workspace is a fair candidate.
    inspector.statuses = [_host_status(used=1_000, limit=1_000)]
    result = await policy.evict_if_needed()
    assert ws.id in result.evicted_workspace_ids
    assert await store.get_workspace(ws.id) is None
    assert not worktree.exists()

    # Re-acquire the same logical workspace. Pressure relieved so the
    # create path doesn't immediately evict the rebuilt one.
    inspector.statuses = [_host_status(used=1, limit=1_000)]
    rebuilt = await service.get_or_create_workspace(req)

    assert rebuilt.id != ws.id
    assert rebuilt.key == ws.key
    rebuilt_path = Path(rebuilt.location.local_path or "")
    assert rebuilt_path.exists()
    assert (rebuilt_path / "README.md").read_text() == "hello\n"
    assert await service.is_workspace_alive(rebuilt.id)

"""Eviction policies.

* :class:`NoOpEvictionPolicy` — never evicts. Kept for tests and
  standalone use where disk pressure genuinely is not a concern.
* :class:`TieredVolumeEvictionPolicy` — the production default. Asks a
  :class:`StorageInspector` how full the backend is and, when a scope is
  over the high-water mark, deletes least-recently-used resources in
  safety-ordered tiers until the scope falls back to the low-water mark.

The policy owns *what* and *when*; the actual deletion goes through a
:class:`WorkspaceReaper` (implemented by ``SandboxService``) so runtime
teardown, store consistency, and the orphan-cache guard stay in one
place. Anything evicted is transparently rebuilt on next use — the
``is_alive`` → re-acquire path re-clones the bare repo and re-forks the
worktree, so committed/pushed work is never lost; only un-pushed edits
in a *dirty* workspace are at risk, and those are touched only as the
explicit last resort.
"""

from __future__ import annotations

import asyncio
import logging

from sandbox.domain.models import (
    RepoCache,
    Workspace,
    WorkspaceMode,
    WorkspaceState,
    utc_now,
)
from sandbox.domain.ports.eviction import EvictionResult, WorkspaceReaper
from sandbox.domain.ports.stores import SandboxStore
from sandbox.domain.ports.storage import StorageInspector, StorageStatus

logger = logging.getLogger(__name__)

DEFAULT_HIGH_WATER = 0.85
DEFAULT_LOW_WATER = 0.70

# Used when a candidate has no measured ``size_bytes`` yet (created
# before the sizing change, or never swept). Non-zero so a sweep still
# makes progress and accounts *something* as freed — the periodic
# sweeper refreshes real sizes so this is only ever a cold-start
# approximation, deliberately on the small side so we don't over-evict.
_FALLBACK_SIZE_BYTES = 64 * 1024 * 1024


class NoOpEvictionPolicy:
    """Eviction policy that never evicts."""

    async def evict_if_needed(
        self, *, user_id: str | None = None, exclude_key: str | None = None
    ) -> EvictionResult:
        _ = (user_id, exclude_key)
        return EvictionResult()


class TieredVolumeEvictionPolicy:
    """Disk/quota-aware LRU eviction with safety-ordered tiers.

    Tier order (stop as soon as the breached scope is back under the
    low-water mark):

      1. clean ``ANALYSIS`` workspaces (read-only, cheapest to rebuild)
      2. clean ``TASK`` workspaces
      3. clean ``EDIT`` workspaces
      4. orphan repo caches (no surviving workspace forks off them)
      5. dirty unpinned workspaces — **last resort**, only when
         ``evict_dirty`` is set and tiers 1–4 didn't free enough

    For ``count:`` scopes (Daytona per-user sandbox cap) the unit is
    *sandboxes*, not bytes: the policy destroys whole least-recently-used
    project sandboxes for the over-quota user instead of pruning
    individual worktrees.

    Never touched: pinned workspaces (``pinned_until`` in the future),
    ``CREATING`` workspaces, and the ``exclude_key`` the caller is about
    to (re)acquire.
    """

    def __init__(
        self,
        *,
        store: SandboxStore,
        inspector: StorageInspector,
        high_water: float = DEFAULT_HIGH_WATER,
        low_water: float = DEFAULT_LOW_WATER,
        evict_dirty: bool = True,
        reaper: WorkspaceReaper | None = None,
    ) -> None:
        if not 0.0 < low_water < high_water <= 1.0:
            raise ValueError(
                "require 0 < low_water < high_water <= 1; got "
                f"low={low_water} high={high_water}"
            )
        self._store = store
        self._inspector = inspector
        self._high_water = high_water
        self._low_water = low_water
        self._evict_dirty = evict_dirty
        self._reaper = reaper
        # One sweep at a time. evict_if_needed runs on the allocation
        # path and from the background sweeper; without this, two
        # concurrent sweeps would each plan against the pre-eviction
        # snapshot and over-delete.
        self._lock = asyncio.Lock()

    def bind_reaper(self, reaper: WorkspaceReaper) -> None:
        """Late-bind the deletion mechanism.

        The bootstrap constructs the policy before the service (the
        service depends on the policy), then calls this so the policy can
        actually delete. Until bound, the policy degrades to a no-op and
        logs — surfacing a wiring bug rather than silently never evicting.
        """
        self._reaper = reaper

    async def evict_if_needed(
        self, *, user_id: str | None = None, exclude_key: str | None = None
    ) -> EvictionResult:
        statuses = await self._inspector.status(user_id=user_id)
        breached = [
            s
            for s in statuses
            if s.over(self._high_water) and s.reclaim_target(self._low_water) > 0
        ]
        if not breached:
            return EvictionResult()

        if self._reaper is None:
            logger.error(
                "sandbox eviction needed (%s) but no reaper is bound — "
                "storage will keep growing. This is a bootstrap wiring bug.",
                ", ".join(b.detail or b.scope for b in breached),
            )
            return EvictionResult()

        async with self._lock:
            evicted_ws: list[str] = []
            evicted_rc: list[str] = []
            freed = 0
            for status in breached:
                logger.warning(
                    "sandbox storage over high-water on %s (%.0f%% — %s); "
                    "evicting to %.0f%%",
                    status.scope,
                    status.pressure * 100,
                    status.detail or status.scope,
                    self._low_water * 100,
                )
                if status.is_count_scope:
                    ws_ids = await self._evict_count_scope(
                        status, exclude_key=exclude_key
                    )
                    evicted_ws.extend(ws_ids)
                else:
                    ws_ids, rc_ids, bytes_freed = await self._evict_byte_scope(
                        status, exclude_key=exclude_key
                    )
                    evicted_ws.extend(ws_ids)
                    evicted_rc.extend(rc_ids)
                    freed += bytes_freed

            result = EvictionResult(
                evicted_workspace_ids=tuple(evicted_ws),
                evicted_repo_cache_ids=tuple(evicted_rc),
                freed_bytes=freed,
            )
            if not result.is_empty:
                logger.warning(
                    "sandbox eviction freed %.2fGiB: %d workspaces, %d caches",
                    freed / (1024**3),
                    len(evicted_ws),
                    len(evicted_rc),
                )
            return result

    # ------------------------------------------------------------------
    # Byte scopes (local host filesystem / per-user volume)
    # ------------------------------------------------------------------
    async def _evict_byte_scope(
        self, status: StorageStatus, *, exclude_key: str | None
    ) -> tuple[list[str], list[str], int]:
        reaper = self._reaper
        assert reaper is not None  # guarded in evict_if_needed
        need = status.reclaim_target(self._low_water)
        workspaces = await self._store.list_workspaces()
        now = utc_now()

        def in_scope(ws: Workspace) -> bool:
            if ws.backend_kind != status.backend_kind:
                return False
            if status.scope == "host":
                return True
            uid = status.user_id
            return uid is not None and ws.request.user_id == uid

        def evictable(ws: Workspace) -> bool:
            if ws.state in (WorkspaceState.CREATING, WorkspaceState.DELETED):
                return False
            if ws.key == exclude_key:
                return False
            if ws.pinned_until is not None and ws.pinned_until > now:
                return False
            return in_scope(ws)

        candidates = [w for w in workspaces if evictable(w)]

        def lru(ws_list: list[Workspace]) -> list[Workspace]:
            return sorted(ws_list, key=lambda w: w.last_used_at)

        clean = [w for w in candidates if not w.dirty]
        dirty = [w for w in candidates if w.dirty]
        tiers: list[list[Workspace]] = [
            lru([w for w in clean if w.request.mode is WorkspaceMode.ANALYSIS]),
            lru([w for w in clean if w.request.mode is WorkspaceMode.TASK]),
            lru([w for w in clean if w.request.mode is WorkspaceMode.EDIT]),
        ]

        evicted_ws: list[str] = []
        freed = 0
        for tier in tiers:
            for ws in tier:
                if freed >= need:
                    break
                if await self._destroy_workspace(ws):
                    evicted_ws.append(ws.id)
                    freed += ws.size_bytes or _FALLBACK_SIZE_BYTES

        # Tier 4: caches no surviving workspace forks off any more.
        evicted_rc: list[str] = []
        if freed < need:
            evicted_set = set(evicted_ws)
            referenced = {
                w.repo_cache_id
                for w in workspaces
                if w.id not in evicted_set and w.repo_cache_id
            }
            caches = await self._store.list_repo_caches()
            orphans = sorted(
                (
                    c
                    for c in caches
                    if c.backend_kind == status.backend_kind
                    and c.id not in referenced
                    and self._cache_in_scope(c, status)
                ),
                key=lambda c: c.last_used_at,
            )
            for cache in orphans:
                if freed >= need:
                    break
                if await reaper.delete_repo_cache_if_unreferenced(cache.id):
                    evicted_rc.append(cache.id)
                    freed += cache.size_bytes or _FALLBACK_SIZE_BYTES

        # Tier 5: last resort — dirty unpinned workspaces. Logged loudly
        # per workspace so dropped un-pushed edits are traceable.
        if freed < need and self._evict_dirty:
            for ws in lru(dirty):
                if freed >= need:
                    break
                logger.warning(
                    "sandbox eviction LAST RESORT: dropping DIRTY workspace "
                    "%s (key=%s, branch=%s) under severe storage pressure — "
                    "un-pushed edits will be lost; committed state rebuilds "
                    "from git on next use",
                    ws.id,
                    ws.key,
                    ws.metadata.get("branch", "?"),
                )
                if await self._destroy_workspace(ws):
                    evicted_ws.append(ws.id)
                    freed += ws.size_bytes or _FALLBACK_SIZE_BYTES

        if freed < need:
            logger.error(
                "sandbox eviction could not reach low-water on %s: freed "
                "%.2fGiB of %.2fGiB needed (everything left is pinned, "
                "in-use, or dirty with evict_dirty disabled)",
                status.scope,
                freed / (1024**3),
                need / (1024**3),
            )
        return evicted_ws, evicted_rc, freed

    @staticmethod
    def _cache_in_scope(cache: RepoCache, status: StorageStatus) -> bool:
        if status.scope == "host":
            return True
        uid = status.user_id
        # RepoCache rows are shared across users (keyed on host|repo) and
        # may carry an optional ``user_id``; only prune within a user
        # scope when we can prove ownership.
        return uid is not None and getattr(cache, "user_id", None) == uid

    # ------------------------------------------------------------------
    # Count scopes (Daytona per-user sandbox cap)
    # ------------------------------------------------------------------
    async def _evict_count_scope(
        self, status: StorageStatus, *, exclude_key: str | None
    ) -> list[str]:
        reaper = self._reaper
        assert reaper is not None  # guarded in evict_if_needed
        uid = status.user_id
        if uid is None:
            return []
        need = status.reclaim_target(self._low_water)  # sandboxes to drop
        now = utc_now()
        workspaces = await self._store.list_workspaces()

        # Group this user's workspaces (on this backend) by project. A
        # project is the eviction unit because Daytona hosts one sandbox
        # per (user, project); destroying it reclaims a count slot.
        projects: dict[str, list[Workspace]] = {}
        for ws in workspaces:
            if ws.backend_kind != status.backend_kind:
                continue
            if ws.request.user_id != uid:
                continue
            projects.setdefault(ws.request.project_id, []).append(ws)

        def project_evictable(ws_list: list[Workspace]) -> bool:
            for ws in ws_list:
                if ws.key == exclude_key:
                    return False
                if ws.state is WorkspaceState.CREATING:
                    return False
                if ws.pinned_until is not None and ws.pinned_until > now:
                    return False
            return True

        # Least-recently-used project first: rank by the most recent
        # activity across the project's workspaces, oldest project wins.
        ranked = sorted(
            (
                (pid, ws_list)
                for pid, ws_list in projects.items()
                if project_evictable(ws_list)
            ),
            key=lambda kv: max(w.last_used_at for w in kv[1]),
        )

        evicted_ws: list[str] = []
        dropped = 0
        for project_id, ws_list in ranked:
            if dropped >= need:
                break
            try:
                await reaper.destroy_pot_container(user_id=uid, project_id=project_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "sandbox count-eviction: destroy_pot_container %s/%s failed: %s",
                    uid,
                    project_id,
                    exc,
                )
                continue
            evicted_ws.extend(w.id for w in ws_list)
            dropped += 1
            logger.warning(
                "sandbox count-eviction: destroyed idle project sandbox "
                "%s/%s (%d/%d slots reclaimed)",
                uid,
                project_id,
                dropped,
                need,
            )
        return evicted_ws

    async def _destroy_workspace(self, ws: Workspace) -> bool:
        reaper = self._reaper
        assert reaper is not None  # guarded in evict_if_needed
        try:
            await reaper.destroy_workspace(ws.id)
            return True
        except Exception as exc:  # noqa: BLE001
            # Best-effort, mirroring destroy_pot_container: a missing
            # backend row must not abort the rest of the sweep.
            logger.warning(
                "sandbox eviction: destroy_workspace %s failed: %s",
                ws.id,
                exc,
            )
            return False

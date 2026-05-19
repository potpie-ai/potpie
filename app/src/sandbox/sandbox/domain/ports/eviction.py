"""Eviction policy port.

Decides when and what to evict to keep the sandbox under disk/quota
thresholds. Workspace providers call ``evict_if_needed`` on the path
that creates new workspaces; background sweepers can call it
proactively. Implementations encapsulate both the policy
(volume/age/pinning) and the mechanism (which adapter to delete from).

Splitting this out of the workspace provider lets us swap policy
without rewriting clone/worktree logic, and lets the local adapter
share the same interface that future Postgres-backed or
volume-aware policies will implement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class EvictionResult:
    """Summary of one eviction pass."""

    evicted_workspace_ids: tuple[str, ...] = field(default_factory=tuple)
    evicted_repo_cache_ids: tuple[str, ...] = field(default_factory=tuple)
    freed_bytes: int = 0

    @property
    def is_empty(self) -> bool:
        return not (self.evicted_workspace_ids or self.evicted_repo_cache_ids)


class WorkspaceReaper(Protocol):
    """The deletion mechanism the eviction policy drives.

    Split from the policy so the policy owns *what* and *when* while the
    service owns *how* — runtime teardown ordering, store consistency,
    and the orphan-cache guard (a cache can't be dropped while a
    workspace still forks worktrees off it). Implemented by
    ``SandboxService``; injected into the policy *after* the service is
    constructed (the bootstrap binds it late to break the
    policy↔service construction cycle).
    """

    async def destroy_workspace(
        self, workspace_id: str, *, destroy_runtime: bool = True
    ) -> None:
        ...

    async def delete_repo_cache_if_unreferenced(self, cache_id: str) -> bool:
        """Drop a bare repo iff no workspace still forks off it.

        Returns ``True`` when the cache was deleted, ``False`` when it
        was kept because a live workspace still references it (the policy
        treats a ``False`` as "no bytes freed, move to the next tier").
        """
        ...

    async def destroy_pot_container(
        self,
        *,
        user_id: str,
        project_id: str,
        delete_repo_caches: bool = False,
    ) -> dict[str, int]:
        """Tear down every workspace + the backing project sandbox.

        Used by the count-scope tier (Daytona): a per-user sandbox cap is
        relieved by destroying a whole idle project sandbox, not by
        pruning individual worktrees.
        """
        ...


class EvictionPolicy(Protocol):
    """Decide when/what to evict before workspace allocation."""

    async def evict_if_needed(
        self, *, user_id: str | None = None, exclude_key: str | None = None
    ) -> EvictionResult:
        """Run the policy; evict if thresholds are breached.

        ``user_id`` scopes the decision when the policy is multi-tenant
        aware (e.g. a per-user volume quota). Pass ``None`` for global
        eviction (the periodic sweeper does this).

        ``exclude_key`` is the ``WorkspaceRequest.key()`` the caller is
        about to (re)acquire. The policy must never evict the workspace
        it is making room for — without this an LRU sweep on the
        allocation path could delete the very worktree the caller is
        seconds away from reusing.

        Implementations must be safe to call concurrently with workspace
        creation — they should take their own locks if they delete
        resources that other callers may be reading.
        """
        ...

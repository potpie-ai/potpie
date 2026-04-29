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


class EvictionPolicy(Protocol):
    """Decide when/what to evict before workspace allocation."""

    async def evict_if_needed(
        self, *, user_id: str | None = None
    ) -> EvictionResult:
        """Run the policy; evict if thresholds are breached.

        ``user_id`` scopes the decision when the policy is multi-tenant
        aware (e.g. a per-user volume quota). Pass ``None`` for global
        eviction.

        Implementations must be safe to call concurrently with workspace
        creation — they should take their own locks if they delete
        resources that other callers may be reading.
        """
        ...

"""Storage-capacity inspection port.

Tells the eviction policy how much room a backend has and how close it
is to the ceiling that actually binds it. Kept separate from
``WorkspaceProvider`` so the policy can reason about disk pressure
without knowing how worktrees are checked out, and so each backend can
report the limit that is real *for that backend*:

* local — the host filesystem ``.repos`` lives on (optionally capped
  lower by an explicit budget),
* Daytona — the per-user sandbox-count cap and the aggregate per-sandbox
  disk the snapshot bakes in.

Adapters typically implement this alongside ``WorkspaceProvider`` (the
Daytona provider already owns the SDK client it needs); the port is kept
separate so it can be faked independently in tests and so the local
inspector can be a standalone, settings-only object with no provider
dependency (which keeps the bootstrap wiring acyclic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class StorageStatus:
    """One storage scope a backend enforces, with its current fill.

    ``scope`` is an opaque-ish label the policy uses only for logging and
    to decide which candidates a breached scope can evict from:

    * ``"host"`` — a single shared filesystem (local). Any workspace /
      cache on that backend is a candidate.
    * ``"user:<id>"`` — a per-user byte budget (e.g. a Daytona volume).
      Only that user's resources are candidates.
    * ``"count:user:<id>"`` — a per-user *count* cap (Daytona's
      sandbox-per-user limit). ``used_bytes`` / ``limit_bytes`` carry the
      count, not bytes; the policy evicts whole project sandboxes rather
      than worktrees for this scope.
    """

    backend_kind: str
    scope: str
    used_bytes: int
    limit_bytes: int
    detail: str = ""

    @property
    def is_count_scope(self) -> bool:
        return self.scope.startswith("count:")

    @property
    def user_id(self) -> str | None:
        for prefix in ("count:user:", "user:"):
            if self.scope.startswith(prefix):
                return self.scope[len(prefix) :]
        return None

    @property
    def pressure(self) -> float:
        """``used / limit``; ``0.0`` when the limit is unbounded/unknown."""
        if self.limit_bytes <= 0:
            return 0.0
        return self.used_bytes / self.limit_bytes

    def over(self, watermark: float) -> bool:
        return self.pressure >= watermark

    def reclaim_target(self, low_water: float) -> int:
        """Units (bytes or count) to free to fall back to ``low_water``.

        ``0`` when already at/under the low-water mark — the policy uses
        a positive value as both the trigger and the stop condition for
        a tiered sweep.
        """
        target = int(self.limit_bytes * low_water)
        return max(0, self.used_bytes - target)


class StorageInspector(Protocol):
    kind: str

    async def status(self, *, user_id: str | None = None) -> list[StorageStatus]:
        """Return every storage scope this backend enforces.

        ``user_id`` lets multi-tenant backends (Daytona per-user volume /
        sandbox-count caps) scope the answer; ``None`` asks for global /
        host-wide scopes only and is what the periodic sweeper passes.

        Must be cheap enough to call on the workspace-allocation path:
        local does one ``shutil.disk_usage`` syscall, Daytona one
        (cached) ``list``. Anything heavier belongs behind the sweeper,
        not here.
        """
        ...

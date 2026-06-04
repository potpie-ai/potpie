"""Local-filesystem storage inspector + a shared directory-size helper.

The local backend always reports *our* footprint (the measured size of
``.repos``) against a byte **budget**, so pressure is
``size(.repos) / budget``. Two ways the budget is set:

* **Explicit** (``SANDBOX_STORAGE_LIMIT_GB``): an absolute ceiling — the
  cache is held to that many GiB regardless of how big the disk is.
* **Default** (no explicit limit): a fraction of the filesystem
  ``.repos`` lives on (``SANDBOX_STORAGE_DISK_FRACTION``, default 0.5).
  This is the "sensible host-disk fraction" default — bounded out of the
  box, scales with the disk, and importantly reads ~0 pressure for an
  empty ``.repos`` so it never flaps in tests or fresh environments the
  way a *whole-disk-fill* signal (which counts other tenants' bytes)
  would.

We deliberately do not key pressure off total filesystem fill: we can
only reclaim what we own, and a CI disk that is independently 90% full
must not make the policy evict the workspace a test just created.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from sandbox.domain.ports.storage import StorageStatus

_GIB = 1024**3


def dir_size_bytes(path: Path) -> int:
    """Best-effort recursive on-disk size of ``path``.

    Uses ``os.scandir`` (no subprocess — portable across macOS/Linux,
    where ``du`` flags differ) and follows no symlinks. Unreadable
    entries are skipped rather than raising: a sizing call must never
    break workspace creation or a sweep. Returns ``0`` for a missing
    path.
    """
    if not path.exists():
        return 0
    total = 0
    stack: list[Path] = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        else:
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


class LocalStorageInspector:
    """Report the filesystem / budget ceiling for the local ``.repos``."""

    kind = "local"

    def __init__(
        self,
        repos_base_path: str | Path | None = None,
        *,
        limit_gb: float | None = None,
        disk_fraction: float = 0.5,
    ) -> None:
        raw = repos_base_path or os.getenv("SANDBOX_REPOS_BASE_PATH") or ".repos"
        self.repos_base_path = Path(raw).expanduser().resolve()
        self.repos_base_path.mkdir(parents=True, exist_ok=True)
        self._explicit_limit_bytes = (
            int(limit_gb * _GIB) if limit_gb and limit_gb > 0 else None
        )
        self._disk_fraction = disk_fraction

    def _budget_bytes(self) -> int | None:
        """Resolve the byte ceiling, or ``None`` if it can't be determined.

        Explicit ``limit_gb`` wins; otherwise a fraction of the
        filesystem total (recomputed each call so a disk that grows/
        shrinks under us stays correct).
        """
        if self._explicit_limit_bytes is not None:
            return self._explicit_limit_bytes
        try:
            total = shutil.disk_usage(self.repos_base_path).total
        except OSError:
            return None
        return int(total * self._disk_fraction)

    async def status(self, *, user_id: str | None = None) -> list[StorageStatus]:
        # The local backend is a single shared filesystem; per-user
        # scoping does not apply, so ``user_id`` is ignored and a single
        # "host" scope is returned.
        _ = user_id
        budget = self._budget_bytes()
        if budget is None or budget <= 0:
            return []
        used = dir_size_bytes(self.repos_base_path)
        return [
            StorageStatus(
                backend_kind=self.kind,
                scope="host",
                used_bytes=used,
                limit_bytes=budget,
                detail=(
                    f".repos={used / _GIB:.2f}GiB / "
                    f"budget={budget / _GIB:.2f}GiB"
                ),
            )
        ]

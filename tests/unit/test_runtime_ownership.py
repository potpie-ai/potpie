from __future__ import annotations

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

import stat
from pathlib import Path

from potpie.runtime import RuntimeOwnershipLock
from potpie_context_engine import Failure, Success


def test_ownership_lock_is_exclusive_owner_only_and_reacquirable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.lock"
    first = RuntimeOwnershipLock(path)
    second = RuntimeOwnershipLock(path)

    assert isinstance(first.acquire(), Success)
    conflict = second.acquire()

    assert isinstance(conflict, Failure)
    assert conflict.error.code == "daemon_ownership_conflict"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    first.release()
    assert isinstance(second.acquire(), Success)
    second.release()


def test_ownership_release_is_idempotent(tmp_path: Path) -> None:
    ownership = RuntimeOwnershipLock(tmp_path / "runtime.lock")
    assert isinstance(ownership.acquire(), Success)

    ownership.release()
    ownership.release()

    assert ownership.is_held is False

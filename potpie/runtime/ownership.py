"""Cross-platform owner-only daemon boot lock."""

from __future__ import annotations

import os
from pathlib import Path

from potpie.runtime.resource_manager import ResourceLifecycleError
from potpie_context_engine import Failure, Success

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None


class RuntimeOwnershipLock:
    """Hold one OS-backed exclusive lock for exactly one daemon boot."""

    def __init__(self, path: Path) -> None:
        if not path.is_absolute():
            raise ValueError("daemon ownership lock path must be absolute")
        self.path = path
        self._descriptor: int | None = None

    @property
    def is_held(self) -> bool:
        return self._descriptor is not None

    def acquire(self) -> Success[None] | Failure[ResourceLifecycleError]:
        if self._descriptor is not None:
            return Success(None)
        self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.path.parent.chmod(0o700)
        descriptor = os.open(  # noqa: PTH123 - owner-only cross-platform lock
            self.path,
            os.O_RDWR | os.O_CREAT,
            0o600,
        )
        os.chmod(self.path, 0o600)
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif msvcrt is not None:  # pragma: no cover - Windows
                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"\0")
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - unsupported platform
                raise OSError("daemon ownership locking is unavailable")
        except OSError:
            os.close(descriptor)
            return Failure(
                ResourceLifecycleError(
                    code="daemon_ownership_conflict",
                    message="another daemon boot owns the runtime scope",
                    recommended_next_action="stop the existing daemon instance",
                    retry_posture="safe",
                )
            )
        self._descriptor = descriptor
        return Success(None)

    def release(self) -> None:
        descriptor, self._descriptor = self._descriptor, None
        if descriptor is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        finally:
            os.close(descriptor)


__all__ = ["RuntimeOwnershipLock"]

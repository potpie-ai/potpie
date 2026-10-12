"""One way to ask ``git`` a short question without risking a hang.

``subprocess.run(capture_output=True, timeout=...)`` can wedge on Windows.
``capture_output`` gives git pipes that ``communicate()`` must drain, and after
a timeout kills git, CPython on Windows calls ``communicate()`` again without a
timeout. A grandchild git started (a credential helper, an askpass, a hook)
keeps the inherited pipe write handle open, so that second call never returns.
Writing stdout to a temporary file leaves no pipe to wait on, so the timeout
holds. ``stdin=DEVNULL`` stops git from waiting on a terminal that is not
there, and ``CREATE_NO_WINDOW`` stops a console window from flashing under a
host that has none.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final

#: ``subprocess.CREATE_NO_WINDOW`` by value; the name exists only on Windows.
_WINDOWS_CREATE_NO_WINDOW: Final[int] = 0x08000000


def run_git_probe(
    args: Sequence[str],
    *,
    cwd: Path | str | None = None,
    timeout: float = 2.0,
) -> str | None:
    """Return ``git <args>``'s stripped stdout, or ``None``.

    ``None`` covers every way of not getting an answer: git is not installed,
    it exits non-zero, ``cwd`` does not exist, or it does not finish within
    ``timeout`` seconds. Never raises.
    """

    argv = ["git"]
    if cwd is not None:
        argv += ["-C", str(cwd)]
    argv += list(args)
    try:
        with tempfile.TemporaryFile() as out:
            completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                argv,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                **_no_window_kwargs(),
            )
            if completed.returncode != 0:
                return None
            out.seek(0)
            return out.read().decode("utf-8", errors="replace").strip()
    except Exception:  # noqa: BLE001 - absent git, timeout, missing cwd: no answer
        return None


def _no_window_kwargs() -> dict[str, int]:
    if _windows():
        return {"creationflags": _WINDOWS_CREATE_NO_WINDOW}
    return {}


def _windows() -> bool:
    """Read at call time so tests can take the Windows branch on any host."""

    return os.name == "nt"


__all__ = ["run_git_probe"]

"""One way to ask ``git`` a question, shared by every place that asks.

Three sites shelled out to ``git`` with ``subprocess.run(capture_output=True,
timeout=...)``, and on Windows that pairing can hang the CLI for good.
``capture_output`` spawns reader threads that ``communicate()`` must join; if
git starts a grandchild (a credential helper, an askpass) that outlives the
timeout, the grandchild keeps the inherited pipe's write handle open after the
timeout kills git itself, and the join never returns. A temp file has no such
handle dependency, so the timeout is real. ``stdin=DEVNULL`` keeps git from
waiting on a terminal that is not there, and ``CREATE_NO_WINDOW`` keeps a
console from flashing under a host that has none.

The fix landed in one site (``potpie.cli.repo_location``) and the other two
kept the original shape. This module is the one shape; the three call it.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from potpie_context_engine.domain.repo_identity import normalize_repo_ref


def _no_window_kwargs() -> dict:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def run_git_probe(
    args: Sequence[str],
    *,
    cwd: Path | str | None = None,
    timeout: float = 2.0,
) -> str | None:
    """``git <args>``'s stdout, stripped, or ``None`` when git is absent, refuses,
    or does not answer within ``timeout``. Never raises."""
    argv = ["git"]
    if cwd is not None:
        argv += ["-C", str(cwd)]
    argv += list(args)
    try:
        with tempfile.TemporaryFile() as out:
            proc = subprocess.run(
                argv,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                **_no_window_kwargs(),
            )
            if proc.returncode != 0:
                return None
            out.seek(0)
            return out.read().decode("utf-8", errors="replace").strip()
    except Exception:  # noqa: BLE001 - absent git, a timeout, a dead cwd: all "no answer"
        return None


def current_git_remote(cwd: Path | str, *, timeout: float = 2.0) -> str | None:
    """The ``origin`` remote of the repository at ``cwd``, normalized to the
    ``host/owner/repo`` key the pot store uses, or ``None``."""
    raw = run_git_probe(["remote", "get-url", "origin"], cwd=cwd, timeout=timeout)
    if not raw:
        return None
    return normalize_repo_ref(raw)


__all__ = ["current_git_remote", "run_git_probe"]

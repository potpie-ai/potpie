"""Stage the agent-sandbox Dockerfile build context.

The bundled ``images/agent-sandbox/Dockerfile`` references two COPY
sources that are NOT checked into the repo (they're regenerated copies
of canonical sources elsewhere in the tree):

* ``parsing_src/`` ← ``app/src/parsing/`` (the parsing_rs Rust crate)
* ``sandbox_src/`` ← ``app/src/sandbox/`` (the potpie-sandbox package)

Both are gitignored under ``images/agent-sandbox/.gitignore``. The
Daytona SDK's ``Image.from_dockerfile`` ships the resolved build-
context tarball over the wire — anything missing from the Dockerfile's
directory at extract time is invisible to the build. So the stagger
must run before any snapshot build, regardless of caller:

* CLI: ``scripts/build_agent_snapshot.py`` (dev loop).
* Runtime: :meth:`DaytonaWorkspaceProvider._build_snapshot` when the
  snapshot is missing on first sandbox creation or backend startup.

This module is the single source of truth for that staging step.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable


# This file lives at app/src/sandbox/sandbox/bootstrap/snapshot_build.py.
# parents indexed from the file:
#   0=bootstrap/, 1=sandbox/ (package), 2=sandbox/ (repo dir),
#   3=src/, 4=app/, 5=<repo root>.
_PACKAGE_DIR = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[5]

DEFAULT_DOCKERFILE = _PACKAGE_DIR / "images" / "agent-sandbox" / "Dockerfile"
DEFAULT_PARSING_SRC = _REPO_ROOT / "app" / "src" / "parsing"
DEFAULT_SANDBOX_SRC = _PACKAGE_DIR

# `target/` is the cargo build cache (huge), `.venv/` and `__pycache__`
# are local artifacts. For the sandbox tree we additionally drop
# `tests/` (image doesn't run them), `images/` (recursive — includes
# our own staging dir), and `scripts/` (host-only build helpers).
_PARSING_EXCLUDE = {"target", "__pycache__", ".venv", "dist", ".pytest_cache"}
_SANDBOX_EXCLUDE = {
    "__pycache__",
    ".venv",
    "dist",
    ".pytest_cache",
    "tests",
    "images",
    "scripts",
}


def _copytree_with_excludes(src: Path, dst: Path, exclude: set[str]) -> None:
    def _ignore(_dir: str, names: list[str]) -> list[str]:
        del _dir  # required by shutil.copytree.ignore contract; unused here
        return [n for n in names if n in exclude]

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=_ignore)


def stage_build_context(
    dockerfile: Path,
    *,
    parsing_src: Path | None = None,
    sandbox_src: Path | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """Materialise ``parsing_src/`` and ``sandbox_src/`` next to the Dockerfile.

    Idempotent: removes prior staging dirs before copying so accumulated
    drift can't bleed across builds. Returns the destination mapping
    for logging. Raises ``FileNotFoundError`` if either canonical source
    is absent (caller decides whether to surface or skip).
    """
    parsing = parsing_src or DEFAULT_PARSING_SRC
    sandbox = sandbox_src or DEFAULT_SANDBOX_SRC
    if not parsing.exists():
        raise FileNotFoundError(
            f"parsing crate not found at {parsing}; can't stage build context"
        )
    if not sandbox.exists():
        raise FileNotFoundError(
            f"sandbox package not found at {sandbox}; can't stage build context"
        )

    parsing_dst = dockerfile.parent / "parsing_src"
    sandbox_dst = dockerfile.parent / "sandbox_src"
    _copytree_with_excludes(parsing, parsing_dst, _PARSING_EXCLUDE)
    _copytree_with_excludes(sandbox, sandbox_dst, _SANDBOX_EXCLUDE)

    if log is not None:
        log(
            f"staged parsing crate at {parsing_dst} "
            f"(excluded: {', '.join(sorted(_PARSING_EXCLUDE))})"
        )
        log(
            f"staged sandbox package at {sandbox_dst} "
            f"(excluded: {', '.join(sorted(_SANDBOX_EXCLUDE))})"
        )
    return {"parsing": parsing_dst, "sandbox": sandbox_dst}

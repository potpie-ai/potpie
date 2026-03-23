"""
ColGREP index helper for Potpie parsing.

How ColGREP separates multiple repos
------------------------------------
The CLI keeps **one index bundle per project path**, not a single merged index for all code.
It picks a base data directory (on Linux, ``dirs::data_dir()`` which honors ``XDG_DATA_HOME``),
then stores indices under ``<base>/colgrep/indices/``. Each checkout gets its own subdirectory
(typically derived from the project directory name plus a short hash of the path), containing
``index/``, ``state.json``, and ``project.json`` (canonical path). Different repositories,
branches, or worktrees therefore use **different** index folders because their filesystem
paths differ.

Potpie layout
-------------
We set ``XDG_DATA_HOME`` **inside** ``REPOS_BASE_PATH`` so embeddings and PLAID state live next
to clones/worktrees, e.g. ``<REPOS_BASE_PATH>/.colgrep/xdg-data/colgrep/indices/...``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

_COLGREP_CMD = "colgrep"


def default_repos_base_path() -> Path:
    """Match ``RepoManager`` resolution: ``REPOS_BASE_PATH`` env or ``<project_root>/.repos``."""
    env_path = os.getenv("REPOS_BASE_PATH")
    if env_path:
        return Path(env_path).resolve()
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    return project_root / ".repos"


def colgrep_xdg_data_home(repos_base_path: Optional[Path] = None) -> Path:
    """
    Directory to use as ``XDG_DATA_HOME`` for ColGREP subprocesses.

    Placed under the repos base (e.g. ``.repos/.colgrep/xdg-data``) so index data stays with
    repository storage. ColGREP will write ``colgrep/indices`` beneath this path.
    """
    base = repos_base_path if repos_base_path is not None else default_repos_base_path()
    xdg = (base / ".colgrep" / "xdg-data").resolve()
    xdg.mkdir(parents=True, exist_ok=True)
    return xdg


def build_colgrep_index(
    repo_root: str,
    repos_base_path: Optional[Path] = None,
) -> None:
    """
    Run ``colgrep init -y`` for ``repo_root`` (best-effort; does not raise on failure).

    Uses ``XDG_DATA_HOME`` under ``repos_base_path`` so indices are scoped under ``REPOS_BASE_PATH``.
    """
    if os.getenv("COLGREP_DISABLE_INDEX", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        logger.info("ColGREP index skipped (COLGREP_DISABLE_INDEX set)")
        return

    if not shutil.which(_COLGREP_CMD):
        logger.warning("ColGREP binary not found on PATH; skipping colgrep init")
        return

    root = Path(repo_root).resolve()
    if not root.is_dir():
        logger.warning("ColGREP init skipped: repo root is not a directory: %s", root)
        return

    xdg_home = colgrep_xdg_data_home(repos_base_path)
    env = os.environ.copy()
    env["XDG_DATA_HOME"] = str(xdg_home)
    env.setdefault("COLGREP_FORCE_CPU", "1")

    timeout_raw = os.getenv("COLGREP_INIT_TIMEOUT_SEC", "7200")
    try:
        timeout_sec = int(timeout_raw) if timeout_raw.strip() else None
    except ValueError:
        timeout_sec = 7200

    cmd = [_COLGREP_CMD, "init", "-y", str(root)]
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if result.returncode != 0:
            logger.warning(
                "colgrep init failed (exit %s) for %s: stderr=%s",
                result.returncode,
                root,
                (result.stderr or "")[:2000],
            )
        else:
            logger.info("ColGREP index initialized for %s (XDG_DATA_HOME=%s)", root, xdg_home)
    except subprocess.TimeoutExpired:
        logger.warning(
            "colgrep init timed out after %s s for %s",
            timeout_sec,
            root,
        )
    except OSError as e:
        logger.warning("colgrep init could not run for %s: %s", root, e)

"""Idempotent bare clone + worktrees for potpie-ai/potpie (PoC test repo)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/potpie-ai/potpie.git"
REPO_NAME = "potpie"


def poc_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repos_dir() -> Path:
    return poc_root() / ".repos"


def bare_path() -> Path:
    return repos_dir() / f"{REPO_NAME}.git"


def base_worktree() -> Path:
    return repos_dir() / REPO_NAME / "main"


def setup() -> None:
    rd = repos_dir()
    rd.mkdir(parents=True, exist_ok=True)
    bp = bare_path()
    if not bp.exists():
        subprocess.run(
            ["git", "clone", "--bare", REPO_URL, str(bp)],
            check=True,
        )
    bw = base_worktree()
    if not bw.exists():
        bw.parent.mkdir(parents=True, exist_ok=True)
        last_err: str | None = None
        for branch in ("main", "master"):
            r = subprocess.run(
                ["git", "-C", str(bp), "worktree", "add", str(bw), branch],
                capture_output=True,
                text=True,
            )
            if r.returncode == 0:
                logger.debug("Base worktree at %s branch=%s", bw, branch)
                break
            last_err = r.stderr or r.stdout
        else:
            raise RuntimeError(
                f"Could not create base worktree at {bw}: {last_err}"
            )


def _base_ref(bp: Path) -> str:
    for name in ("main", "master"):
        p = subprocess.run(
            ["git", "-C", str(bp), "rev-parse", "--verify", name],
            capture_output=True,
        )
        if p.returncode == 0:
            return name
    return "main"


def create_worktree(branch: str) -> str:
    """Create a new worktree on a new branch from main/master; return absolute path."""
    setup()
    bp = bare_path()
    safe = branch.replace("/", "_").replace(" ", "_")
    wt = repos_dir() / REPO_NAME / safe
    if wt.exists():
        return str(wt.resolve())
    wt.parent.mkdir(parents=True, exist_ok=True)
    base = _base_ref(bp)
    subprocess.run(
        [
            "git",
            "-C",
            str(bp),
            "worktree",
            "add",
            str(wt),
            "-b",
            branch,
            base,
        ],
        check=True,
    )
    return str(wt.resolve())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup()
    print("OK:", base_worktree())

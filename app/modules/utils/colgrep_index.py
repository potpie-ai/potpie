"""
ColGREP index helper for Potpie parsing.

How ColGREP separates multiple repos
------------------------------------
The CLI keeps **one index bundle per project path**, not a single merged index for all code.
It picks a base data directory and stores indices under ``<base>/colgrep/indices/``. Each
checkout gets its own subdirectory (typically derived from the project directory name plus a
short hash of the path), containing ``index/``, ``state.json``, and ``project.json`` (canonical
path). Different repositories, branches, or worktrees therefore use **different** index folders
because their filesystem paths differ.

Potpie layout
-------------
Potpie requests ``XDG_DATA_HOME`` under ``REPOS_BASE_PATH`` so ColGREP can keep index data near
repository storage. The modified next-plaid build used by Potpie honors ``XDG_DATA_HOME`` across
platforms, so parser-side indexing and sandboxed search can share the same index root.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

_COLGREP_CMD = "colgrep"
_COLGREP_DOCKER_PLATFORMS = {
    "amd64": "linux/amd64",
    "arm64": "linux/arm64/v8",
}


def _normalize_arch(machine: str | None = None) -> str:
    """Normalize OS/CPU arch names to the labels used for packaged Linux binaries."""
    raw = (machine or platform.machine()).strip().lower()
    if raw in {"x86_64", "amd64"}:
        return "amd64"
    if raw in {"arm64", "aarch64"}:
        return "arm64"
    return raw


def default_colgrep_binary_path() -> Path:
    """Project-local ColGREP binary built by ``scripts/ensure_colgrep.sh``."""
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    return project_root / ".tools" / "bin" / _COLGREP_CMD


def default_linux_packaged_colgrep_binary_path(arch: str = "amd64") -> Path:
    """Packaged Linux ColGREP binary built for sandbox and container use."""
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    normalized_arch = _normalize_arch(arch)
    return project_root / ".tools" / "bin" / f"colgrep-linux-{normalized_arch}"


def _packaged_linux_binary_candidates() -> list[tuple[Path, str]]:
    """
    Candidate packaged Linux binaries in priority order.

    Prefer the host's Docker-native architecture first, then allow amd64 fallback on arm64 hosts
    where Docker Desktop can emulate amd64 containers.
    """
    preferred_arch = _normalize_arch()
    candidates: list[tuple[Path, str]] = []

    if preferred_arch == "arm64":
        arch_order = ["arm64", "amd64"]
    elif preferred_arch == "amd64":
        arch_order = ["amd64"]
    else:
        arch_order = ["amd64"]

    for arch in arch_order:
        docker_platform = _COLGREP_DOCKER_PLATFORMS.get(arch)
        if docker_platform is None:
            continue
        candidates.append((default_linux_packaged_colgrep_binary_path(arch), docker_platform))

    return candidates


def _resolve_executable(path: Path) -> Optional[str]:
    if path.is_file() and os.access(path, os.X_OK):
        return str(path)
    return None


def resolve_colgrep_binary() -> Optional[str]:
    """
    Resolve the ColGREP binary with a deterministic fallback chain.

    Order:
    1. ``COLGREP_BINARY`` if it points to an executable
    2. Project-local ``.tools/bin/colgrep``
    3. ``colgrep`` found on ``PATH``
    4. Matching packaged Linux binary when running on Linux
    """
    env_binary = os.getenv("COLGREP_BINARY", "").strip()
    if env_binary:
        path = Path(env_binary).expanduser().resolve()
        resolved = _resolve_executable(path)
        if resolved:
            return resolved
        logger.warning(
            "COLGREP_BINARY was set but is not executable: {}",
            path,
        )

    local_binary = default_colgrep_binary_path()
    resolved_local_binary = _resolve_executable(local_binary)
    if resolved_local_binary:
        return resolved_local_binary

    path_binary = shutil.which(_COLGREP_CMD)
    if path_binary:
        return path_binary

    if platform.system().lower() == "linux":
        for packaged_binary, _docker_platform in _packaged_linux_binary_candidates():
            resolved_packaged_binary = _resolve_executable(packaged_binary)
            if resolved_packaged_binary:
                return resolved_packaged_binary

    return None


def resolve_sandbox_colgrep_binary() -> tuple[Optional[str], Optional[str]]:
    """
    Resolve a Linux-compatible ColGREP binary for Docker/gVisor sandboxes.

    Returns the host path to mount plus the Docker platform string that should be used for the
    sandbox image. ``COLGREP_SANDBOX_BINARY`` can override the packaged selection.
    """
    env_binary = os.getenv("COLGREP_SANDBOX_BINARY", "").strip()
    if env_binary:
        path = Path(env_binary).expanduser().resolve()
        resolved = _resolve_executable(path)
        if resolved:
            docker_platform = _COLGREP_DOCKER_PLATFORMS.get(_normalize_arch())
            return resolved, docker_platform
        logger.warning(
            "COLGREP_SANDBOX_BINARY was set but is not executable: {}",
            path,
        )

    for packaged_binary, docker_platform in _packaged_linux_binary_candidates():
        resolved_packaged_binary = _resolve_executable(packaged_binary)
        if resolved_packaged_binary:
            return resolved_packaged_binary, docker_platform

    if platform.system().lower() == "linux":
        local_binary = default_colgrep_binary_path()
        resolved_local_binary = _resolve_executable(local_binary)
        if resolved_local_binary:
            return resolved_local_binary, _COLGREP_DOCKER_PLATFORMS.get(_normalize_arch())

        path_binary = shutil.which(_COLGREP_CMD)
        if path_binary:
            return path_binary, _COLGREP_DOCKER_PLATFORMS.get(_normalize_arch())

    return None, None


def default_repos_base_path() -> Path:
    """Match ``RepoManager`` resolution: ``REPOS_BASE_PATH`` env or ``<project_root>/.repos``."""
    env_path = os.getenv("REPOS_BASE_PATH")
    if env_path:
        return Path(env_path).resolve()
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    return project_root / ".repos"


def colgrep_xdg_data_home(repos_base_path: Optional[Path] = None) -> Path:
    """
    Directory to request via ``XDG_DATA_HOME`` for ColGREP subprocesses.

    Placed under the repos base (e.g. ``.repos/.colgrep/xdg-data``) so index data can stay with
    repository storage on platforms where ColGREP honors ``XDG_DATA_HOME``.
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

    Requests ``XDG_DATA_HOME`` under ``repos_base_path`` so indices can be scoped under
    ``REPOS_BASE_PATH`` on platforms where ColGREP honors that variable.
    """
    if os.getenv("COLGREP_DISABLE_INDEX", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        logger.info("ColGREP index skipped (COLGREP_DISABLE_INDEX set)")
        return

    colgrep_binary = resolve_colgrep_binary()
    if not colgrep_binary:
        logger.warning(
            "ColGREP binary not found; checked COLGREP_BINARY, PATH, project-local .tools/bin, and Linux packaged fallback"
        )
        return

    root = Path(repo_root).resolve()
    if not root.is_dir():
        logger.warning("ColGREP init skipped: repo root is not a directory: {}", root)
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

    cmd = [colgrep_binary, "init", "-y", str(root)]
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()[:2000]
            stdout = (result.stdout or "").strip()[:2000]
            logger.warning(
                "colgrep init failed (exit {}) for {} using {}. stderr={!r} stdout={!r}",
                result.returncode,
                root,
                colgrep_binary,
                stderr,
                stdout,
            )
        else:
            logger.info(
                "ColGREP index initialized for {} using {} (requested XDG_DATA_HOME={})",
                root,
                colgrep_binary,
                xdg_home,
            )
    except subprocess.TimeoutExpired:
        logger.warning(
            "colgrep init timed out after {} s for {} using {}",
            timeout_sec,
            root,
            colgrep_binary,
        )
    except OSError as e:
        logger.warning("colgrep init could not run for {} using {}: {}", root, colgrep_binary, e)

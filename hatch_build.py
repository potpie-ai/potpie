"""Hatchling build hook: stamp the source identity into the ``potpie`` wheel.

``potpie``'s version is a constant (``2.0.0``) while the code moves with every
commit, and consumers pin it by git rev -- the Pie VSIX vendors a wheel built
from one. Once installed, nothing in the wheel says which rev that was:
``metadata.version`` answers ``2.0.0`` for every build, and the PEP 610
``direct_url.json`` an installer writes points at the wheel file, not at the
commit the wheel was built from. Two laptops can run different code under
identical version strings with no way to tell from the machine itself.

So the wheel carries ``potpie/_build.json`` -- ``{"version", "rev", "dirty",
"built_at"}`` -- written here, at build time, from the checkout being built.
:mod:`potpie.build_info` reads it back for ``potpie --version`` and the
daemon's ``/health``.

The file is written into the source tree (gitignored) rather than a temp dir,
because an *editable* install has to find it too: the package is then imported
straight from the checkout, and a file force-included into the editable wheel
would land in site-packages, where nothing looks. For a built wheel the same
file is force-included, which is what gets it past the .gitignore that
hatchling's file selection otherwise honours.

Outside a git checkout -- a wheel built from an sdist -- the rev cannot be
recomputed, so an existing ``_build.json`` (the sdist carries the one written
when it was built) is kept rather than overwritten with nulls.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface
except ImportError:  # pragma: no cover - the test venv has no build backend
    # hatchling is a build requirement, not a runtime one. The functions below
    # are tested from the ordinary venv, where only the hook class needs the
    # base; a stand-in keeps the module importable there.
    BuildHookInterface = object  # type: ignore[assignment,misc]

BUILD_FILE = Path("potpie") / "_build.json"


def _git(root: Path, *args: str) -> str | None:
    """``git -C root *args`` stdout, or None when git is absent or refuses.

    Output goes to a temp file, not a pipe -- the same shape as
    ``potpie.cli.repo_location``. ``capture_output`` spawns reader threads the
    call must join, and on Windows a git grandchild that outlives the timeout
    keeps the inherited pipe open, so the join never returns.
    """
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        with tempfile.TemporaryFile() as out:
            proc = subprocess.run(
                ["git", "-C", str(root), *args],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.DEVNULL,
                timeout=10,
                **kwargs,
            )
            if proc.returncode != 0:
                return None
            out.seek(0)
            return out.read().decode("utf-8", errors="replace").strip()
    except Exception:  # noqa: BLE001 -- any failure means "no git identity"
        return None


def collect_build_info(root: Path, version: str, *, now: datetime | None = None) -> dict[str, Any]:
    """The stamp for the checkout at ``root``. ``rev``/``dirty`` are None
    outside a repository; ``dirty`` counts untracked files, since a file that
    was never ``git add``ed is just as invisible to the rev as an edit."""
    rev = _git(root, "rev-parse", "HEAD")
    dirty: bool | None = None
    if rev:
        status = _git(root, "status", "--porcelain")
        dirty = None if status is None else bool(status)
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return {
        "version": version,
        "rev": rev,
        "dirty": dirty,
        "built_at": stamp.isoformat(timespec="seconds"),
    }


def write_build_info(root: Path, version: str) -> Path:
    """Write (or, outside git, preserve) ``<root>/potpie/_build.json``."""
    target = root / BUILD_FILE
    info = collect_build_info(root, version)
    if info["rev"] is None and target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return target


class BuildInfoHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        # `version` here is the build *target* version ("standard"/"editable"),
        # not the project's; the project version comes off the metadata.
        target = write_build_info(Path(self.root), self.metadata.version)
        if version == "editable":
            return  # read live from the source tree
        build_data.setdefault("force_include", {})[str(target)] = BUILD_FILE.as_posix()

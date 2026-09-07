"""Which build of ``potpie`` this is.

Two identities matter and they are easy to confuse:

* the **distribution** -- ``potpie``, the product: CLI, daemon, this package.
  Its version is a constant (``2.0.0``) and consumers pin it by git rev, so the
  version alone does not say which code is running. The rev lives in
  ``potpie/_build.json``, stamped into the wheel by ``hatch_build.py``.
* the **engine** -- ``potpie-context-engine``, the library underneath. Also a
  constant version. ``potpie --version`` used to report *only* this, under the
  engine's name, which told a reader nothing about the CLI they had just run.

:func:`describe` reports both, plus the build stamp, and is the one source
``--version`` and the daemon's ``/health`` read from. Every field degrades to
``None``/``"unknown"`` rather than raising: a source checkout that was never
installed has no metadata, and a wheel built outside git has no rev.

**A checkout outranks the stamp.** An editable install (``make cli-install``,
``uv tool install -e``) imports this package straight out of the source tree,
so the code that runs is whatever the checkout holds *now* -- while
``_build.json`` still says what HEAD was when the wheel was last built. On a
tree several commits past its last install, ``--version`` named a rev that no
longer existed in the code, and ``daemon status`` compared that stale stamp
against itself and reported ``stale: false`` for a daemon started before the
commits. So when the package sits inside a git checkout, ``rev`` comes from
``.git`` (a file read, no subprocess -- it is on every command's path through
:func:`cli_version`) and ``dirty`` from one ``git status`` when
:func:`describe` is asked for it.
"""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from importlib import metadata, resources
from pathlib import Path
from typing import Any, Final

DISTRIBUTION: Final = "potpie"
ENGINE_DISTRIBUTION: Final = "potpie-context-engine"
_BUILD_FILE: Final = "_build.json"
#: A stamp read off the checkout rather than the baked file carries this
#: marker so :func:`describe` knows to ask git for ``dirty``.
STAMP_SOURCE_CHECKOUT: Final = "checkout"
_GIT_TIMEOUT: Final = 2.0


def distribution_version(name: str = DISTRIBUTION) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _baked_stamp() -> dict[str, Any]:
    """The ``_build.json`` written at build time, or ``{}`` when absent.

    Read through ``importlib.resources`` so it resolves the same way for a
    site-packages install, an editable checkout, and a PyInstaller bundle
    (which collects ``potpie``'s data files along with its code).
    """
    try:
        raw = (resources.files(__package__) / _BUILD_FILE).read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError, TypeError, AttributeError):
        return {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def checkout_root() -> Path | None:
    """The git checkout this package is imported from, or ``None``.

    Only the directory that holds ``potpie/`` counts, and ``.git`` may be a
    file there (a worktree). A wheel in site-packages has no ``.git`` beside
    it, so an installed build keeps reading the stamp.
    """
    root = Path(__file__).resolve().parent.parent
    return root if (root / ".git").exists() else None


def _git(root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-C", str(root), *args],  # noqa: S607 - git from PATH, as the build hook does
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _rev_from_git_files(root: Path) -> str | None:
    """HEAD's sha from the ``.git`` files alone.

    ``HEAD`` is either the sha (detached) or ``ref: refs/heads/<branch>``; the
    ref is a loose file under ``refs/`` or a line in ``packed-refs``. A worktree's
    ``.git`` is a file pointing at its own ``gitdir``, whose ``commondir`` names
    where the refs live. Anything this does not understand returns ``None`` and
    the caller falls back to ``git rev-parse``.
    """
    git = root / ".git"
    try:
        if git.is_file():
            pointer = git.read_text(encoding="utf-8").strip()
            if not pointer.startswith("gitdir:"):
                return None
            git_dir = Path(pointer.partition(":")[2].strip())
            if not git_dir.is_absolute():
                git_dir = (root / git_dir).resolve()
        else:
            git_dir = git
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not head.startswith("ref:"):
        return head or None
    ref = head.partition(":")[2].strip()
    common = git_dir
    try:
        commondir = git_dir / "commondir"
        if commondir.is_file():
            common = (git_dir / commondir.read_text(encoding="utf-8").strip()).resolve()
    except OSError:
        pass
    for base in (git_dir, common):
        try:
            loose = base / ref
            if loose.is_file():
                sha = loose.read_text(encoding="utf-8").strip()
                if sha:
                    return sha
        except OSError:
            continue
    try:
        for line in (common / "packed-refs").read_text(encoding="utf-8").splitlines():
            if not line or line[0] in "#^":
                continue
            sha, _, name = line.partition(" ")
            if name.strip() == ref:
                return sha
    except OSError:
        pass
    return None


def _checkout_rev(root: Path) -> str | None:
    return _rev_from_git_files(root) or _git(root, "rev-parse", "HEAD")


@lru_cache(maxsize=1)
def _checkout_dirty(root: Path) -> bool | None:
    """Modified *tracked* files -- the ``git describe --dirty`` definition, the
    same one ``hatch_build.py`` stamps with, so the two readings agree. One
    subprocess, once per process; ``None`` when git does not answer."""
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    return None if status is None else bool(status)


@lru_cache(maxsize=1)
def build_stamp() -> dict[str, Any]:
    """``{rev, dirty, built_at}`` for the code that is running.

    For an installed wheel that is ``_build.json``. Inside a checkout it is
    HEAD, read from the ``.git`` files (no subprocess: :func:`cli_version` puts
    this on every command's path), with ``dirty`` left ``None`` for
    :func:`describe` to fill in -- and ``built_at`` ``None``, because the time
    the wheel was last built says nothing about the commit now checked out.
    Cached per process: a daemon that computes it at startup keeps reporting
    the rev it *loaded*, however far the checkout moves afterwards.
    """
    root = checkout_root()
    if root is not None:
        rev = _checkout_rev(root)
        if rev:
            return {
                "rev": rev,
                "dirty": None,
                "built_at": None,
                "source": STAMP_SOURCE_CHECKOUT,
            }
    return _baked_stamp()


def short_rev(rev: object) -> str | None:
    return rev[:10] if isinstance(rev, str) and rev else None


def describe() -> dict[str, Any]:
    """``{name, version, build: {rev, dirty, built_at}, engine: {name, version}}``."""
    stamp = build_stamp()
    dirty = stamp.get("dirty")
    if stamp.get("source") == STAMP_SOURCE_CHECKOUT:
        root = checkout_root()
        if root is not None:
            dirty = _checkout_dirty(root)
    return {
        "name": DISTRIBUTION,
        "version": distribution_version() or "unknown",
        "build": {
            "rev": stamp.get("rev"),
            "dirty": dirty,
            "built_at": stamp.get("built_at"),
        },
        "engine": {
            "name": ENGINE_DISTRIBUTION,
            "version": distribution_version(ENGINE_DISTRIBUTION) or "unknown",
        },
    }


def cli_version() -> str:
    """``2.0.0+81da1550e3``: the distribution version, plus the short rev as a
    PEP 440 local label when one is known.

    Telemetry used to tag every command ``cli_version: 0.1.0`` -- the *engine*
    library's version, which the engine reported because it cannot import
    ``potpie`` to know better -- so no dashboard could tell one CLI release
    from another. The distribution's version alone would not fix that either:
    it is a constant, and the rev is the part that identifies the code.
    """
    version = distribution_version() or "unknown"
    rev = short_rev(build_stamp().get("rev"))
    return f"{version}+{rev}" if rev else version


def cli_release() -> str:
    """The Sentry release name for this CLI: ``potpie-cli@<cli_version>``."""
    return f"potpie-cli@{cli_version()}"


def human_line(info: dict[str, Any]) -> str:
    """``potpie 2.0.0 (81da1550e3)``, with ``, dirty`` when the build was."""
    build = info.get("build") or {}
    rev = short_rev(build.get("rev"))
    if rev is None:
        mark = "build rev unknown"
    else:
        mark = f"{rev}, dirty" if build.get("dirty") else rev
    return f"{info.get('name', DISTRIBUTION)} {info.get('version', 'unknown')} ({mark})"


__all__ = [
    "DISTRIBUTION",
    "ENGINE_DISTRIBUTION",
    "STAMP_SOURCE_CHECKOUT",
    "build_stamp",
    "checkout_root",
    "cli_release",
    "cli_version",
    "describe",
    "distribution_version",
    "human_line",
    "short_rev",
]

"""Which build of ``potpie`` this is.

Two identities matter and they are easy to confuse:

* the **distribution** -- ``potpie``, the product: CLI, daemon, this package.
  Its version moves only at releases, and installs pinned to a git rev share
  the version of the release they follow, so the version alone does not say
  which code is running. The build hook stamps the git rev, a dirty flag and
  the build time into ``potpie/runtime/_build_info.py`` in every wheel and
  sdist.
* the **engine** -- ``potpie-context-engine``, the library underneath, with
  its own version.

An editable install has no generated module: the hook writes it into the
wheel, while an editable install imports this package straight from the
checkout. There the stamp is read from git instead, which is also the more
truthful answer, since the code running is whatever the checkout holds now.

:func:`describe` reports both identities plus the stamp, and is the one source
for ``potpie --version``. The daemon reports :func:`build_stamp` in its
status, and :func:`build_is_stale` compares it with this process's own. Every
field degrades to ``None`` / ``"unknown"`` rather than raising: a wheel built
outside git has no rev, and an uninstalled checkout has no metadata.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Mapping
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any, Final

DISTRIBUTION: Final = "potpie"
ENGINE_DISTRIBUTION: Final = "potpie-context-engine"
_GENERATED_MODULE: Final = "potpie.runtime._build_info"
_CHECKOUT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_GIT_TIMEOUT_S: Final[float] = 5.0


def distribution_version(name: str = DISTRIBUTION) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def build_stamp() -> dict[str, Any]:
    """``{version, rev, dirty, built_at}`` for this install, or ``{}``.

    ``{}`` means nothing is known: no generated build module and no git
    checkout to ask. Cached, because the answer cannot change while the
    process runs the code it describes.
    """
    try:
        # A static import, so freezers that collect modules by analysis
        # (PyInstaller) bundle the generated module with the package.
        import potpie.runtime._build_info as _build_info
    except ModuleNotFoundError as exc:
        if exc.name == _GENERATED_MODULE:
            return _checkout_stamp(_CHECKOUT_ROOT)
        return {}
    except ImportError:
        return {}
    return {
        "version": distribution_version(),
        "rev": _text(getattr(_build_info, "GIT_SHA", None)),
        "dirty": _flag(getattr(_build_info, "DIRTY", None)),
        "built_at": _text(getattr(_build_info, "BUILD_TIME", None)),
    }


def short_rev(rev: object) -> str | None:
    return rev[:10] if isinstance(rev, str) and rev else None


def describe() -> dict[str, Any]:
    """``{name, version, build: {rev, dirty, built_at}, engine: {name, version}}``."""
    stamp = build_stamp()
    return {
        "name": DISTRIBUTION,
        "version": distribution_version() or "unknown",
        "build": {
            "rev": stamp.get("rev"),
            "dirty": stamp.get("dirty"),
            "built_at": stamp.get("built_at"),
        },
        "engine": {
            "name": ENGINE_DISTRIBUTION,
            "version": distribution_version(ENGINE_DISTRIBUTION) or "unknown",
        },
    }


def human_line(info: Mapping[str, Any]) -> str:
    """``potpie 2.0.1 (aaaaaaaaaa)``, with ``, dirty`` when the build was."""
    build = info.get("build") or {}
    rev = short_rev(build.get("rev"))
    if rev is None:
        mark = "build rev unknown"
    else:
        mark = f"{rev}, dirty" if build.get("dirty") else rev
    return f"{info.get('name', DISTRIBUTION)} {info.get('version', 'unknown')} ({mark})"


def build_is_stale(served: Mapping[str, Any]) -> bool | None:
    """Whether another process's build differs from this one's.

    ``None`` when either side has no rev to compare: an unstamped build is not
    evidence either way. Revs, not trees, are compared, so two dirty builds of
    one rev compare equal.
    """
    ours = build_stamp().get("rev")
    theirs = served.get("rev")
    if not ours or not theirs:
        return None
    return ours != theirs


def _checkout_stamp(root: Path) -> dict[str, Any]:
    """The stamp of the git checkout rooted at ``root``, or ``{}``.

    Only a checkout whose top level *is* ``root`` counts, so an install that
    merely sits inside some other repository does not report that
    repository's HEAD. ``dirty`` means modified tracked files, as in
    ``git describe --dirty``.
    """
    identity = _git(root, "rev-parse", "--show-toplevel", "HEAD")
    lines = identity.splitlines() if identity else []
    if len(lines) != 2:
        return {}
    toplevel, rev = (line.strip() for line in lines)
    try:
        same_root = Path(toplevel).resolve() == root.resolve()
    except OSError:
        same_root = False
    if not same_root or not rev:
        return {}
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    return {
        "version": distribution_version(),
        "rev": rev,
        "dirty": None if status is None else bool(status),
        "built_at": None,
    }


def _git(root: Path, *args: str) -> str | None:
    """``git -C root *args`` stdout, or ``None`` when git is absent or refuses.

    Output goes to a temp file rather than a pipe: on Windows a git grandchild
    that outlives the timeout keeps an inherited pipe open, and the reader
    join would never return.
    """
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        with tempfile.TemporaryFile() as out:
            completed = subprocess.run(
                ["git", "--no-optional-locks", "-C", str(root), *args],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.DEVNULL,
                timeout=_GIT_TIMEOUT_S,
                **kwargs,
            )
            if completed.returncode != 0:
                return None
            out.seek(0)
            return out.read().decode("utf-8", errors="replace").strip()
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _text(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _flag(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    lowered = str(value or "").strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


__all__ = [
    "DISTRIBUTION",
    "ENGINE_DISTRIBUTION",
    "build_is_stale",
    "build_stamp",
    "describe",
    "distribution_version",
    "human_line",
    "short_rev",
]

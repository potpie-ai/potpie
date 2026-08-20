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
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import metadata, resources
from typing import Any, Final

DISTRIBUTION: Final = "potpie"
ENGINE_DISTRIBUTION: Final = "potpie-context-engine"
_BUILD_FILE: Final = "_build.json"


def distribution_version(name: str = DISTRIBUTION) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def build_stamp() -> dict[str, Any]:
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
    "build_stamp",
    "describe",
    "distribution_version",
    "human_line",
    "short_rev",
]

"""Temporary compatibility re-export for :mod:`potpie_context_engine.core.api`."""

from __future__ import annotations

from potpie_context_engine.core import api as _canonical
from potpie_context_engine.core.api import *  # noqa: F403

__all__ = getattr(
    _canonical,
    "__all__",
    tuple(name for name in vars(_canonical) if not name.startswith("_")),
)

"""Temporary compatibility re-export for :mod:`potpie_context_engine.core.ports.agent_context`."""

from __future__ import annotations

from potpie_context_engine.core.ports import agent_context as _canonical
from potpie_context_engine.core.ports.agent_context import *  # noqa: F403

__all__ = getattr(
    _canonical,
    "__all__",
    tuple(name for name in vars(_canonical) if not name.startswith("_")),
)

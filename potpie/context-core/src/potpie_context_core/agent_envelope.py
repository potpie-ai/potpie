"""Temporary compatibility re-export for :mod:`potpie_context_engine.core.agent_envelope`."""

from __future__ import annotations

from potpie_context_engine.core import agent_envelope as _canonical
from potpie_context_engine.core.agent_envelope import *  # noqa: F403

__all__ = getattr(
    _canonical,
    "__all__",
    tuple(name for name in vars(_canonical) if not name.startswith("_")),
)

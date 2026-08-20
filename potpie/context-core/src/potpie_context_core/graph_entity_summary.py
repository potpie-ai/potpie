"""Temporary compatibility re-export for :mod:`potpie_context_engine.core.graph_entity_summary`."""

from __future__ import annotations

from potpie_context_engine.core import graph_entity_summary as _canonical
from potpie_context_engine.core.graph_entity_summary import *  # noqa: F403

__all__ = getattr(
    _canonical,
    "__all__",
    tuple(name for name in vars(_canonical) if not name.startswith("_")),
)

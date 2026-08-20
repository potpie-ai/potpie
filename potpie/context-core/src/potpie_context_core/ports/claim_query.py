"""Temporary compatibility re-export for :mod:`potpie_context_engine.core.ports.claim_query`."""

from __future__ import annotations

from potpie_context_engine.core.ports import claim_query as _canonical
from potpie_context_engine.core.ports.claim_query import *  # noqa: F403

__all__ = getattr(
    _canonical,
    "__all__",
    tuple(name for name in vars(_canonical) if not name.startswith("_")),
)

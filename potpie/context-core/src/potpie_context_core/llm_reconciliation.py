"""Temporary compatibility re-export for :mod:`potpie_context_engine.core.llm_reconciliation`."""

from __future__ import annotations

from potpie_context_engine.core import llm_reconciliation as _canonical
from potpie_context_engine.core.llm_reconciliation import *  # noqa: F403

__all__ = getattr(
    _canonical,
    "__all__",
    tuple(name for name in vars(_canonical) if not name.startswith("_")),
)

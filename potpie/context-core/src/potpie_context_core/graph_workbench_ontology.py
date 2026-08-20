"""Temporary compatibility re-export for :mod:`potpie_context_engine.core.graph_workbench_ontology`."""

from __future__ import annotations

from potpie_context_engine.core import graph_workbench_ontology as _canonical
from potpie_context_engine.core.graph_workbench_ontology import *  # noqa: F403

__all__ = getattr(
    _canonical,
    "__all__",
    tuple(name for name in vars(_canonical) if not name.startswith("_")),
)

"""Context readers — one module per evidence family.

Each reader implements
:class:`domain.ports.context_reader.ContextReaderPort` and is registered
with the :class:`ContextReaderRegistry` at container build time. The
application layer never imports a concrete reader.
"""

from __future__ import annotations

from adapters.outbound.readers.change_history import ChangeHistoryReader
from adapters.outbound.readers.decisions import DecisionsReader
from adapters.outbound.readers.graph_overview import GraphOverviewReader
from adapters.outbound.readers.owners import OwnersReader
from adapters.outbound.readers.pr_diff import PrDiffReader
from adapters.outbound.readers.pr_review_context import PrReviewContextReader
from adapters.outbound.readers.project_graph import ProjectGraphReader
from adapters.outbound.readers.release_notes import ReleaseNotesReader
from adapters.outbound.readers.semantic_search import SemanticSearchReader
from adapters.outbound.readers.timeline import TimelineReader

__all__ = [
    "ChangeHistoryReader",
    "DecisionsReader",
    "GraphOverviewReader",
    "OwnersReader",
    "PrDiffReader",
    "PrReviewContextReader",
    "ProjectGraphReader",
    "ReleaseNotesReader",
    "SemanticSearchReader",
    "TimelineReader",
]

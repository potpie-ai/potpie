"""ContextReader port (Phase 3).

Each evidence family (decisions, change_history, owners, project_graph, ...)
implements this Protocol as a self-contained module. The registry routes
``ContextGraphQuery`` requests to the matching readers and merges the
:class:`ReaderResult` envelopes into the final response — there is no
per-family branching in the application layer.
"""

from __future__ import annotations

from typing import Protocol

from domain.context_reader import ReaderCapability, ReaderResult
from domain.graph_query import ContextGraphQuery


class ContextReaderPort(Protocol):
    def family(self) -> str:
        """Stable evidence-family key (e.g. ``"decisions"``)."""
        ...

    def capability(self) -> ReaderCapability:
        """Static capability descriptor — used by the router."""
        ...

    def read(self, request: ContextGraphQuery) -> ReaderResult:
        """Execute the read for this family. Synchronous by design."""
        ...


__all__ = ["ContextReaderPort"]

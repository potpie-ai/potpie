"""Unified context graph port.

Application code should depend on this port for graph reads. Existing
episodic/structural ports can remain adapter-internal while the migration
collapses the query surface.
"""

from __future__ import annotations

from typing import Protocol

from domain.graph_query import (
    ContextGraphQuery,
    ContextGraphResult,
)


class ContextGraphPort(Protocol):
    @property
    def enabled(self) -> bool:
        ...

    def query(self, request: ContextGraphQuery) -> ContextGraphResult:
        ...

    async def query_async(self, request: ContextGraphQuery) -> ContextGraphResult:
        ...

"""Unified context graph port.

Application code depends on this port for graph reads and writes. Graphiti
episodic writes and canonical structural mutations are adapter-internal
details of the concrete graph layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from domain.graph_mutations import ProvenanceContext
from domain.graph_query import (
    ContextGraphQuery,
    ContextGraphResult,
)
from domain.reconciliation import ReconciliationPlan, ReconciliationResult


class ContextGraphPort(Protocol):
    @property
    def enabled(self) -> bool:
        ...

    def query(self, request: ContextGraphQuery) -> ContextGraphResult:
        ...

    async def query_async(self, request: ContextGraphQuery) -> ContextGraphResult:
        ...

    def apply_plan(
        self,
        plan: ReconciliationPlan,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> ReconciliationResult:
        ...

    def write_raw_episode(
        self,
        pot_id: str,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
    ) -> dict[str, Any]:
        ...

    def reset_pot(self, pot_id: str) -> dict[str, Any]:
        ...

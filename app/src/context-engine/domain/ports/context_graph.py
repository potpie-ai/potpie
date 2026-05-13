"""Unified context graph port.

Application code depends on this port for graph reads and writes. Graphiti
episodic writes and canonical structural mutations are adapter-internal
details of the concrete graph layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from domain.actor import Actor
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

    async def apply_plan_async(
        self,
        plan: ReconciliationPlan,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> ReconciliationResult:
        """Async-native plan apply. Preferred when called from inside an
        event loop (e.g. agent tools, FastAPI handlers) — avoids the
        sync→async→sync bridge that can cross-bind Neo4j connections to a
        dead loop."""
        ...

    def write_raw_episode(
        self,
        pot_id: str,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
        *,
        actor: Actor | None = None,
    ) -> dict[str, Any]:
        ...

    def reset_pot(self, pot_id: str) -> dict[str, Any]:
        ...

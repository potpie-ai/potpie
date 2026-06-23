"""Legacy managed context graph compatibility port.

New application code should depend on ``domain.ports.services.graph_service``
or ``domain.ports.agent_context``. This port remains only for managed callers
that still speak ``ContextGraphQuery`` while they migrate to the canonical DTOs.
Legacy ``MutationBatch`` apply is intentionally allowed to be not implemented.
"""

from __future__ import annotations

from typing import Any, Protocol

from potpie.context_engine.domain.graph_mutations import ProvenanceContext
from potpie.context_engine.domain.graph_query import (
    ContextGraphQuery,
    ContextGraphResult,
)
from potpie.context_engine.domain.reconciliation import MutationBatch, MutationResult


class ContextGraphPort(Protocol):
    @property
    def enabled(self) -> bool: ...

    def query(self, request: ContextGraphQuery) -> ContextGraphResult: ...

    async def query_async(self, request: ContextGraphQuery) -> ContextGraphResult: ...

    def apply_plan(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> MutationResult: ...

    async def apply_plan_async(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> MutationResult:
        """Async-native plan apply. Preferred when called from inside an
        event loop (e.g. agent tools, FastAPI handlers) — avoids the
        sync→async→sync bridge that can cross-bind Neo4j connections to a
        dead loop."""
        ...

    def reset_pot(self, pot_id: str) -> dict[str, Any]: ...

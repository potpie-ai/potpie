"""Provider-neutral port for backing intelligence systems (graph, vector, hybrid)."""

from __future__ import annotations

from typing import Any, Protocol

from domain.intelligence_models import (
    ArtifactContext,
    ArtifactRef,
    CapabilitySet,
    CausalChainItem,
    ChangeRecord,
    ContextScope,
    DebuggingMemoryRecord,
    DecisionRecord,
    DiscussionRecord,
    OwnershipRecord,
    ProjectContextRecord,
)


class IntelligenceProvider(Protocol):
    """Async intelligence provider. Sync backends wrap I/O in asyncio.to_thread."""

    async def search_context(
        self,
        pot_id: str,
        query: str,
        *,
        limit: int = 8,
        node_labels: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        ...

    async def get_artifact_context(
        self,
        pot_id: str,
        artifact: ArtifactRef,
    ) -> ArtifactContext | None:
        ...

    async def get_change_history(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        limit: int = 10,
        as_of: str | None = None,
        query: str | None = None,
    ) -> list[ChangeRecord]:
        ...

    async def get_decision_context(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        limit: int = 20,
        query: str | None = None,
    ) -> list[DecisionRecord]:
        ...

    async def get_related_discussions(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        limit: int = 10,
    ) -> list[DiscussionRecord]:
        ...

    async def get_ownership(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        limit: int = 5,
    ) -> list[OwnershipRecord]:
        ...

    async def get_project_map(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        include: list[str],
        limit: int = 12,
    ) -> list[ProjectContextRecord]:
        ...

    async def get_debugging_memory(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        include: list[str],
        query: str,
        limit: int = 12,
    ) -> list[DebuggingMemoryRecord]:
        ...

    async def get_causal_chain(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        query: str,
        max_depth: int = 6,
        as_of_iso: str | None = None,
        window_days: int = 180,
    ) -> list[CausalChainItem]:
        ...

    async def list_open_conflicts(self, pot_id: str) -> list[dict[str, Any]]:
        """Open predicate-family ``QualityIssue`` rows for this pot (episodic graph)."""

    def get_capabilities(self) -> CapabilitySet:
        ...

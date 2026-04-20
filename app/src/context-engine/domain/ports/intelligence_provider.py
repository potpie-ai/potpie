"""Provider-neutral port for backing intelligence systems (graph, vector, hybrid)."""

from __future__ import annotations

from typing import Any, Protocol

from domain.intelligence_models import (
    ArtifactContext,
    ArtifactRef,
    CapabilitySet,
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
    ) -> list[ChangeRecord]:
        ...

    async def get_decision_context(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        limit: int = 20,
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

    def get_capabilities(self) -> CapabilitySet:
        ...

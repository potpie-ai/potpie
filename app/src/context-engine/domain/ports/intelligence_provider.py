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
        """Semantic / hybrid search over pot-scoped knowledge."""

    async def get_artifact_context(
        self,
        pot_id: str,
        artifact: ArtifactRef,
    ) -> ArtifactContext | None:
        """Load structured context for a known artifact (e.g. PR)."""

    async def get_change_history(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        limit: int = 10,
    ) -> list[ChangeRecord]:
        """Deterministic change history for optional file/function scope."""

    async def get_decision_context(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        limit: int = 20,
    ) -> list[DecisionRecord]:
        """Design decisions linked to code or PR review."""

    async def get_related_discussions(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        limit: int = 10,
    ) -> list[DiscussionRecord]:
        """Review threads / discussions when scope includes PR or file context."""

    async def get_ownership(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        limit: int = 5,
    ) -> list[OwnershipRecord]:
        """Likely owners for a file path."""

    async def get_project_map(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        include: list[str],
        limit: int = 12,
    ) -> list[ProjectContextRecord]:
        """Canonical project-map records for features, services, docs, operations, and instructions."""

    async def get_debugging_memory(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        include: list[str],
        query: str,
        limit: int = 12,
    ) -> list[DebuggingMemoryRecord]:
        """Reusable prior fixes, incidents, alerts, investigations, and diagnostic signals."""

    def get_capabilities(self) -> CapabilitySet:
        """Static capability flags for this provider."""

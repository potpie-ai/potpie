"""Deterministic mock provider for tests and local development."""

from __future__ import annotations

from domain.intelligence_models import (
    ArtifactContext,
    ArtifactRef,
    CapabilitySet,
    ChangeRecord,
    ContextScope,
    DecisionRecord,
    DiscussionRecord,
    OwnershipRecord,
)
from domain.ports.intelligence_provider import IntelligenceProvider


class MockIntelligenceProvider(IntelligenceProvider):
    """Returns canned data; all capabilities enabled."""

    async def search_context(
        self,
        pot_id: str,
        query: str,
        *,
        limit: int = 8,
        node_labels: list[str] | None = None,
    ) -> list[dict]:
        _ = node_labels
        return [
            {
                "uuid": "mock-uuid-1",
                "name": f"Mock hit for {query[:40]}",
                "summary": "Mock semantic summary",
                "fact": None,
            }
        ][:limit]

    async def get_artifact_context(
        self,
        pot_id: str,
        artifact: ArtifactRef,
    ) -> ArtifactContext | None:
        return ArtifactContext(
            kind=artifact.kind,
            identifier=artifact.identifier,
            title="Mock artifact",
            summary="Mock artifact summary",
        )

    async def get_change_history(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        limit: int = 10,
    ) -> list[ChangeRecord]:
        _ = limit
        return [
            ChangeRecord(
                pr_number=1,
                title="Mock PR",
                summary="Mock change",
                artifact_ref="PR #1",
            )
        ]

    async def get_decision_context(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        limit: int = 20,
    ) -> list[DecisionRecord]:
        _ = scope
        _ = limit
        return [
            DecisionRecord(
                decision="Mock decision",
                rationale="Because tests",
                pr_number=1,
            )
        ]

    async def get_related_discussions(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        limit: int = 10,
    ) -> list[DiscussionRecord]:
        _ = limit
        if scope.pr_number is None:
            return []
        return [
            DiscussionRecord(
                source_ref=f"PR #{scope.pr_number}",
                summary="Mock discussion",
                full_text="Mock thread body",
            )
        ]

    async def get_ownership(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        limit: int = 5,
    ) -> list[OwnershipRecord]:
        _ = limit
        if not scope.file_path:
            return []
        return [
            OwnershipRecord(
                file_path=scope.file_path,
                owner="mock-owner",
                confidence_signal="mock",
            )
        ]

    def get_capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            semantic_search=True,
            artifact_context=True,
            change_history=True,
            decision_context=True,
            discussion_context=True,
            ownership_context=True,
            workflow_context=False,
        )

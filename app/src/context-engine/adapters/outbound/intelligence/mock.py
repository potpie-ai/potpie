"""Deterministic mock provider for tests and local development."""

from __future__ import annotations

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

    async def get_project_map(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        include: list[str],
        limit: int = 12,
    ) -> list[ProjectContextRecord]:
        _ = pot_id
        requested = set(include)
        records = [
            ProjectContextRecord(
                family="service_map",
                kind="Service",
                entity_key="service:context-engine",
                name="context-engine",
                summary="Mock service project-map record",
                relationships=[
                    {
                        "type": "BACKED_BY",
                        "direction": "out",
                        "target_kind": "Repository",
                        "target_name": scope.repo_name or "potpie",
                    }
                ],
            ),
            ProjectContextRecord(
                family="feature_map",
                kind="Feature",
                entity_key="feature:context-graph",
                name="context graph",
                summary="Mock feature project-map record",
            ),
            ProjectContextRecord(
                family="docs",
                kind="Document",
                entity_key="doc:context-graph",
                name="Context Graph Architecture",
                summary="Mock docs project-map record",
                source_uri="docs/context-graph/graph.md",
            ),
            ProjectContextRecord(
                family="local_workflows",
                kind="LocalWorkflow",
                entity_key="workflow:context-engine-tests",
                name="Context engine unit tests",
                summary="Run uv run pytest app/src/context-engine/tests/unit/ from the repository root.",
                source_uri="app/src/context-engine/README.md",
            ),
            ProjectContextRecord(
                family="runbooks",
                kind="Runbook",
                entity_key="runbook:context-engine-doctor",
                name="Context engine doctor",
                summary="Run uv run context-engine --json doctor from app/src/context-engine to check Potpie API connectivity.",
                source_uri="app/src/context-engine/adapters/inbound/cli/README.md",
            ),
            ProjectContextRecord(
                family="scripts",
                kind="Script",
                entity_key="script:context-engine-lab",
                name="context_engine_lab.py",
                summary="Run bundled mock and live smoke flows for context-engine.",
                source_uri="app/src/context-engine/scripts/context_engine_lab.py",
            ),
            ProjectContextRecord(
                family="config",
                kind="ConfigVariable",
                entity_key="config:potpie-api-url",
                name="POTPIE_API_URL",
                summary="Potpie base URL used by context-engine CLI and MCP clients.",
                source_uri="app/src/context-engine/adapters/inbound/cli/README.md",
            ),
        ]
        filtered = [
            item
            for item in records
            if not requested or item.family in requested or "operations" in requested
        ]
        return filtered[:limit]

    async def get_debugging_memory(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        include: list[str],
        query: str,
        limit: int = 12,
    ) -> list[DebuggingMemoryRecord]:
        _ = pot_id
        _ = query
        requested = set(include)
        records = [
            DebuggingMemoryRecord(
                family="prior_fixes",
                kind="Fix",
                entity_key="fix:mock-timeout",
                title="Mock timeout fix",
                summary="Increased ingestion timeout after repeated repository fetch failures.",
                status="active",
                fix_type="configuration",
                root_cause="Provider requests exceeded the previous timeout.",
                source_ref="github:pr:1",
                affected_scope=[
                    {
                        "kind": "Service",
                        "name": scope.services[0]
                        if scope.services
                        else "context-engine",
                    }
                ],
                diagnostic_signals=[
                    {
                        "kind": "DiagnosticSignal",
                        "summary": "repository ingestion timeout",
                    }
                ],
                related_changes=[{"kind": "PullRequest", "identifier": "1"}],
            ),
            DebuggingMemoryRecord(
                family="diagnostic_signals",
                kind="DiagnosticSignal",
                entity_key="signal:mock-timeout",
                title="Repository ingestion timeout",
                summary="Timeout while fetching repository metadata.",
                status="active",
                affected_scope=[
                    {
                        "kind": "Environment",
                        "name": scope.environment or "staging",
                    }
                ],
            ),
        ]
        filtered = [
            item for item in records if not requested or item.family in requested
        ]
        return filtered[:limit]

    def get_capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            semantic_search=True,
            artifact_context=True,
            change_history=True,
            decision_context=True,
            discussion_context=True,
            ownership_context=True,
            project_map_context=True,
            debugging_memory_context=True,
            workflow_context=False,
        )

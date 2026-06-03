"""Deterministic mock provider for tests and local development."""

from __future__ import annotations

from typing import Any

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
        as_of: str | None = None,
        query: str | None = None,
    ) -> list[ChangeRecord]:
        _ = limit
        _ = query
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
        query: str | None = None,
    ) -> list[DecisionRecord]:
        _ = pot_id
        _ = scope
        _ = query
        records = [
            DecisionRecord(
                decision="Use Neo4j for graph storage and Postgres for the event ledger",
                rationale="Graphiti requires Neo4j for temporal edge support. Postgres provides ACID guarantees for reconciliation events.",
                pr_number=1,
            ),
            DecisionRecord(
                decision="Minimal agent context port with four tools",
                rationale="Reduces agent tool surface and keeps all workflows behind context_resolve recipes.",
                pr_number=2,
            ),
            DecisionRecord(
                decision="API v2 uses API keys instead of Firebase auth",
                rationale="CLI and MCP tools need stable authentication independent of Firebase.",
                pr_number=3,
            ),
        ]
        return records[:limit]

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
                family="tickets",
                kind="Issue",
                entity_key="issue:context-graph-42",
                name="#42: Add ticket context",
                summary="Mock ticket project-map record",
                source_uri="https://github.com/acme/app/issues/42",
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
                summary="Run uv run potpie --json doctor from app/src/context-engine to check Potpie API connectivity.",
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
                summary="Potpie base URL used by Potpie CLI and MCP clients.",
                source_uri="app/src/context-engine/adapters/inbound/cli/README.md",
            ),
            ProjectContextRecord(
                family="deployments",
                kind="Deployment",
                entity_key="deployment:context-engine-prod",
                name="Context engine production deployment",
                summary="Deploy rollback: if health checks fail for more than five minutes, roll back to previous image and alert on-call.",
                source_uri="app/src/context-engine/docs/deploy.md",
            ),
            ProjectContextRecord(
                family="alerts",
                kind="Alert",
                entity_key="alert:celery-queue-depth",
                name="Celery queue depth alert",
                summary="Monitor context-graph-etl queue depth every sixty seconds. Alert if depth exceeds one hundred messages.",
                source_uri="app/src/context-engine/docs/alerts.md",
            ),
            ProjectContextRecord(
                family="incidents",
                kind="Incident",
                entity_key="incident:repo-ingestion-timeout",
                name="Repository ingestion timeout",
                summary="Ingestion worker timed out after 120 seconds. Root cause: missing index on event_id. Fix: added composite index and moved to context-graph-etl queue.",
                source_uri="app/src/context-engine/docs/incidents.md",
            ),
            ProjectContextRecord(
                family="preferences",
                kind="Preference",
                entity_key="preference:source-reference-first",
                name="Source-reference-first graph memory",
                summary="Store compact facts and source refs in graph. Fetch full diffs only when requested with summary, snippets, or verify policies.",
                source_uri="app/src/context-engine/docs/preferences.md",
            ),
            ProjectContextRecord(
                family="owners",
                kind="Owner",
                entity_key="owner:context-engine",
                name="Context engine team",
                summary="Owned by the platform team. Primary on-call rotates weekly.",
                source_uri="app/src/context-engine/docs/owners.md",
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
        requested = set(include)
        q = query.lower()
        records: list[DebuggingMemoryRecord] = []

        if "memory" in q or "leak" in q:
            records.append(
                DebuggingMemoryRecord(
                    family="prior_fixes",
                    kind="Fix",
                    entity_key="fix:mock-memory-leak",
                    title="Mock memory leak fix",
                    summary="Clamped batch size to 64 and added garbage collection between episodes to fix memory leak in bulk ingestion.",
                    status="active",
                    fix_type="configuration",
                    root_cause="Embedding tensors accumulated without release during bulk repository ingestion.",
                    source_ref="github:pr:2",
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
                            "summary": "memory leak in context graph worker",
                        }
                    ],
                    related_changes=[{"kind": "PullRequest", "identifier": "2"}],
                )
            )

        if "connection" in q or "pool" in q or "neo4j" in q or "503" in q:
            records.append(
                DebuggingMemoryRecord(
                    family="prior_fixes",
                    kind="Fix",
                    entity_key="fix:mock-neo4j-pool",
                    title="Mock Neo4j connection pool fix",
                    summary="Increased Neo4j connection pool max size to 50 and added exponential backoff retry for transient connection failures.",
                    status="active",
                    fix_type="configuration",
                    root_cause="Default connection pool size of 10 was insufficient under concurrent search load.",
                    source_ref="github:pr:3",
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
                            "summary": "Neo4j connection pool exhaustion",
                        }
                    ],
                    related_changes=[{"kind": "PullRequest", "identifier": "3"}],
                )
            )

        records.append(
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
            )
        )
        records.append(
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
            )
        )
        filtered = [
            item for item in records if not requested or item.family in requested
        ]
        return filtered[:limit]

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
        _ = pot_id
        _ = scope
        _ = query
        _ = max_depth
        _ = as_of_iso
        _ = window_days
        return []

    async def list_open_conflicts(self, pot_id: str) -> list[dict[str, Any]]:
        _ = pot_id
        return []

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
            causal_chain_context=True,
        )

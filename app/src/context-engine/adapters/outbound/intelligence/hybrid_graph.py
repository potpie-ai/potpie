"""IntelligenceProvider backed by Graphiti (episodic) + Neo4j (structural)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

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
from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.intelligence_provider import IntelligenceProvider
from domain.ports.structural_graph import StructuralGraphPort

logger = logging.getLogger(__name__)

_DEFAULT_NODE_LABELS = ["PullRequest", "Decision", "Issue", "Feature"]

_PROJECT_MAP_FAMILIES_BY_LABEL = {
    "Pot": "purpose",
    "Repository": "repo_map",
    "System": "purpose",
    "Service": "service_map",
    "Component": "service_map",
    "Capability": "feature_map",
    "Feature": "feature_map",
    "Functionality": "feature_map",
    "Requirement": "feature_map",
    "RoadmapItem": "feature_map",
    "Interface": "service_map",
    "DataStore": "service_map",
    "Integration": "service_map",
    "Dependency": "service_map",
    "Document": "docs",
    "Deployment": "deployments",
    "DeploymentTarget": "deployments",
    "DeploymentStrategy": "deployments",
    "Environment": "deployments",
    "Runbook": "runbooks",
    "Script": "scripts",
    "ConfigVariable": "config",
    "Preference": "preferences",
    "AgentInstruction": "agent_instructions",
    "LocalWorkflow": "local_workflows",
    "Person": "owners",
    "Team": "owners",
}

_OPERATIONS_FAMILIES = {
    "deployments",
    "runbooks",
    "scripts",
    "config",
    "local_workflows",
}

_DEBUGGING_MEMORY_FAMILIES_BY_LABEL = {
    "Fix": "prior_fixes",
    "BugPattern": "prior_fixes",
    "Investigation": "prior_fixes",
    "DiagnosticSignal": "diagnostic_signals",
    "Incident": "incidents",
    "Alert": "alerts",
}


def _semantic_hits_to_dicts(items: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        rows.append(
            {
                "uuid": str(getattr(item, "uuid", "")),
                "name": getattr(item, "name", None),
                "summary": getattr(item, "summary", None),
                "fact": getattr(item, "fact", None),
            }
        )
    return rows


def _row_to_change_record(r: dict[str, Any]) -> ChangeRecord:
    prn = r.get("pr_number")
    try:
        pr_int = int(prn) if prn is not None else None
    except (TypeError, ValueError):
        pr_int = None
    decs = r.get("decisions")
    if not isinstance(decs, list):
        decs = None
    return ChangeRecord(
        pr_number=pr_int,
        title=r.get("title"),
        summary=r.get("why_summary"),
        artifact_ref=f"PR #{pr_int}" if pr_int is not None else None,
        decisions=decs,
        change_type=r.get("change_type"),
        feature_area=r.get("feature_area"),
    )


def _row_to_decision_record(r: dict[str, Any]) -> DecisionRecord:
    prn = r.get("pr_number")
    try:
        pr_int = int(prn) if prn is not None else None
    except (TypeError, ValueError):
        pr_int = None
    return DecisionRecord(
        decision=str(r.get("decision_made") or ""),
        rationale=r.get("rationale") or None,
        alternatives_rejected=r.get("alternatives_rejected") or None,
        pr_number=pr_int,
    )


def _pr_review_to_artifact(row: dict[str, Any], identifier: str) -> ArtifactContext:
    return ArtifactContext(
        kind="pr",
        identifier=identifier,
        title=row.get("pr_title"),
        summary=row.get("pr_summary") or row.get("pr_description"),
        author=row.get("pr_author") or None,
        created_at=row.get("pr_merged_at") or None,
        extra={
            "commits": row.get("commits") or [],
            "found": row.get("found"),
        },
    )


def _threads_to_discussions(
    pr_number: int,
    threads: list[dict[str, Any]],
) -> list[DiscussionRecord]:
    out: list[DiscussionRecord] = []
    for t in threads:
        if not t:
            continue
        tid = t.get("thread_id")
        src = f"PR #{pr_number} thread {tid}" if tid else f"PR #{pr_number} review"
        line = t.get("line")
        try:
            line_int = int(line) if line is not None else None
        except (TypeError, ValueError):
            line_int = None
        out.append(
            DiscussionRecord(
                source_ref=src,
                file_path=t.get("file_path"),
                line=line_int,
                headline=t.get("headline"),
                full_text=t.get("full_discussion") or "",
                summary=t.get("headline"),
            )
        )
    return out


def _row_to_project_context_record(row: dict[str, Any]) -> ProjectContextRecord | None:
    labels = row.get("labels") or []
    if not isinstance(labels, list):
        labels = []
    canonical = [label for label in labels if label in _PROJECT_MAP_FAMILIES_BY_LABEL]
    if not canonical:
        return None
    kind = canonical[0]
    props = row.get("properties") or {}
    if not isinstance(props, dict):
        props = {}
    family = _PROJECT_MAP_FAMILIES_BY_LABEL[kind]
    relationships = row.get("relationships") or []
    if not isinstance(relationships, list):
        relationships = []
    source_uri = (
        props.get("source_uri")
        or props.get("url")
        or props.get("uri")
        or props.get("retrieval_uri")
    )
    return ProjectContextRecord(
        family=family,
        kind=kind,
        entity_key=row.get("entity_key") or props.get("entity_key"),
        name=props.get("name") or props.get("title") or props.get("statement"),
        summary=(
            props.get("summary")
            or props.get("description")
            or props.get("statement")
            or props.get("command")
        ),
        status=props.get("status") or props.get("lifecycle_state"),
        source_ref=props.get("source_ref"),
        source_uri=str(source_uri) if source_uri else None,
        relationships=relationships,
        properties={
            key: value
            for key, value in props.items()
            if key
            in {
                "criticality",
                "environment_type",
                "component_type",
                "integration_type",
                "dependency_type",
                "workflow_type",
                "preference_type",
                "instruction_type",
                "scope_kind",
                "command",
                "path_hint",
            }
        },
    )


def _row_to_debugging_memory_record(
    row: dict[str, Any],
) -> DebuggingMemoryRecord | None:
    labels = row.get("labels") or []
    if not isinstance(labels, list):
        labels = []
    canonical = [
        label for label in labels if label in _DEBUGGING_MEMORY_FAMILIES_BY_LABEL
    ]
    if not canonical:
        return None
    kind = canonical[0]
    props = row.get("properties") or {}
    if not isinstance(props, dict):
        props = {}
    relationships = row.get("relationships") or []
    if not isinstance(relationships, list):
        relationships = []
    source_uri = (
        props.get("source_uri")
        or props.get("url")
        or props.get("uri")
        or props.get("retrieval_uri")
    )
    return DebuggingMemoryRecord(
        family=_DEBUGGING_MEMORY_FAMILIES_BY_LABEL[kind],
        kind=kind,
        entity_key=row.get("entity_key") or props.get("entity_key"),
        title=props.get("title") or props.get("name"),
        summary=(
            props.get("summary")
            or props.get("description")
            or props.get("root_cause")
            or props.get("message")
        ),
        status=props.get("status") or props.get("lifecycle_state"),
        severity=props.get("severity"),
        root_cause=props.get("root_cause"),
        fix_type=props.get("fix_type"),
        source_ref=props.get("source_ref"),
        source_uri=str(source_uri) if source_uri else None,
        affected_scope=_filter_relationships(
            relationships,
            {"IMPACTS", "SEEN_IN", "INVOLVES_CODE", "FIRED_IN"},
        ),
        diagnostic_signals=_filter_relationships(relationships, {"OBSERVED_IN"}),
        related_changes=_filter_relationships(
            relationships, {"CHANGED_BY", "RESOLVED"}
        ),
        relationships=relationships,
        properties={
            key: value
            for key, value in props.items()
            if key
            in {
                "signal_type",
                "fingerprint",
                "environment",
                "service",
                "component",
                "confidence",
                "last_observed_at",
                "started_at",
                "resolved_at",
            }
        },
    )


def _filter_relationships(
    relationships: list[dict[str, Any]],
    edge_types: set[str],
) -> list[dict[str, Any]]:
    return [rel for rel in relationships if rel.get("type") in edge_types]


class HybridGraphIntelligenceProvider(IntelligenceProvider):
    """Maps episodic + structural ports into normalized intelligence records."""

    def __init__(
        self,
        episodic: EpisodicGraphPort,
        structural: StructuralGraphPort,
    ) -> None:
        self._episodic = episodic
        self._structural = structural

    async def search_context(
        self,
        pot_id: str,
        query: str,
        *,
        limit: int = 8,
        node_labels: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self._episodic.enabled:
            return []
        labels = node_labels or _DEFAULT_NODE_LABELS
        try:
            items = await self._episodic.search_async(
                pot_id=pot_id,
                query=query,
                limit=max(1, min(limit, 50)),
                node_labels=labels,
            )
        except Exception as exc:
            logger.exception("search_async failed: %s", exc)
            return []
        return _semantic_hits_to_dicts(items)

    async def get_artifact_context(
        self,
        pot_id: str,
        artifact: ArtifactRef,
    ) -> ArtifactContext | None:
        if artifact.kind != "pr":
            return None
        try:
            num = int(artifact.identifier)
        except ValueError:
            return None

        def _load() -> dict[str, Any]:
            return self._structural.get_pr_review_context(pot_id, num, repo_name=None)

        try:
            row = await asyncio.to_thread(_load)
        except Exception as exc:
            logger.exception("get_pr_review_context failed: %s", exc)
            return None
        if not row.get("found"):
            return None
        return _pr_review_to_artifact(row, artifact.identifier)

    async def get_change_history(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        limit: int = 10,
    ) -> list[ChangeRecord]:
        def _load() -> list[dict[str, Any]]:
            return self._structural.get_change_history(
                pot_id,
                scope.function_name,
                scope.file_path,
                max(1, min(limit, 100)),
                repo_name=scope.repo_name,
                pr_number=scope.pr_number,
            )

        try:
            rows = await asyncio.to_thread(_load)
        except Exception as exc:
            logger.exception("get_change_history failed: %s", exc)
            return []
        return [_row_to_change_record(r) for r in rows]

    async def get_decision_context(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        limit: int = 20,
    ) -> list[DecisionRecord]:
        def _load() -> list[dict[str, Any]]:
            return self._structural.get_decisions(
                pot_id,
                scope.file_path,
                scope.function_name,
                max(1, min(limit, 100)),
                repo_name=scope.repo_name,
                pr_number=scope.pr_number,
            )

        try:
            rows = await asyncio.to_thread(_load)
        except Exception as exc:
            logger.exception("get_decisions failed: %s", exc)
            return []
        return [_row_to_decision_record(r) for r in rows]

    async def get_related_discussions(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        limit: int = 10,
    ) -> list[DiscussionRecord]:
        _ = limit  # structural API returns full PR context; limit reserved for future
        if scope.pr_number is None:
            return []

        pr_num = scope.pr_number

        def _load() -> dict[str, Any]:
            return self._structural.get_pr_review_context(
                pot_id, pr_num, repo_name=scope.repo_name
            )

        try:
            row = await asyncio.to_thread(_load)
        except Exception as exc:
            logger.exception("get_pr_review_context (discussions) failed: %s", exc)
            return []
        if not row.get("found"):
            return []
        threads = row.get("review_threads") or []
        return _threads_to_discussions(pr_num, threads)

    async def get_ownership(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        limit: int = 5,
    ) -> list[OwnershipRecord]:
        if not scope.file_path:
            return []

        def _load() -> list[dict[str, Any]]:
            return self._structural.get_file_owners(
                pot_id,
                scope.file_path,
                max(1, min(limit, 50)),
                repo_name=scope.repo_name,
            )

        try:
            rows = await asyncio.to_thread(_load)
        except Exception as exc:
            logger.exception("get_file_owners failed: %s", exc)
            return []
        out: list[OwnershipRecord] = []
        for r in rows:
            login = r.get("github_login") or "unknown"
            cnt = r.get("pr_count")
            out.append(
                OwnershipRecord(
                    file_path=scope.file_path,
                    owner=str(login),
                    confidence_signal=f"{cnt} PR(s)" if cnt is not None else None,
                )
            )
        return out

    async def get_project_map(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        include: list[str],
        limit: int = 12,
    ) -> list[ProjectContextRecord]:
        query_include = set(include)
        if "operations" in query_include:
            query_include.update(_OPERATIONS_FAMILIES)
        structural_scope = {
            "repo_name": scope.repo_name,
            "services": list(scope.services),
            "features": list(scope.features),
            "environment": scope.environment,
            "ticket_ids": list(scope.ticket_ids),
            "user": scope.user,
            "file_path": scope.file_path,
        }

        def _load() -> dict[str, Any]:
            return self._structural.get_project_graph(
                pot_id,
                pr_number=scope.pr_number,
                limit=max(1, min(limit, 100)),
                scope=structural_scope,
                include=sorted(query_include),
            )

        try:
            payload = await asyncio.to_thread(_load)
        except Exception as exc:
            logger.exception("get_project_graph failed: %s", exc)
            return []
        rows = payload.get("nodes") or []
        out: list[ProjectContextRecord] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            item = _row_to_project_context_record(row)
            if item is None:
                continue
            if query_include and item.family not in query_include:
                continue
            out.append(item)
        return out[:limit]

    async def get_debugging_memory(
        self,
        pot_id: str,
        scope: ContextScope,
        *,
        include: list[str],
        query: str,
        limit: int = 12,
    ) -> list[DebuggingMemoryRecord]:
        structural_scope = {
            "repo_name": scope.repo_name,
            "services": list(scope.services),
            "features": list(scope.features),
            "environment": scope.environment,
            "ticket_ids": list(scope.ticket_ids),
            "user": scope.user,
            "file_path": scope.file_path,
        }

        def _load() -> dict[str, Any]:
            return self._structural.get_debugging_memory(
                pot_id,
                limit=max(1, min(limit, 100)),
                scope=structural_scope,
                include=include,
                query=query,
            )

        try:
            payload = await asyncio.to_thread(_load)
        except Exception as exc:
            logger.exception("get_debugging_memory failed: %s", exc)
            return []
        rows = payload.get("nodes") or []
        requested = set(include)
        out: list[DebuggingMemoryRecord] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            item = _row_to_debugging_memory_record(row)
            if item is None:
                continue
            if requested and item.family not in requested:
                continue
            out.append(item)
        return out[:limit]

    def get_capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            semantic_search=self._episodic.enabled,
            artifact_context=True,
            change_history=True,
            decision_context=True,
            discussion_context=True,
            ownership_context=True,
            project_map_context=True,
            debugging_memory_context=True,
            workflow_context=False,
        )

"""IntelligenceProvider backed by Graphiti (episodic) + Neo4j (structural)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from application.use_cases.query_context import search_pot_context_async

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
    "Issue": "tickets",
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
        labels = node_labels or _DEFAULT_NODE_LABELS
        try:
            return await search_pot_context_async(
                self._episodic,
                pot_id,
                query,
                limit=max(1, min(limit, 50)),
                node_labels=labels,
                structural=self._structural,
            )
        except Exception as exc:
            logger.exception("search_context failed: %s", exc)
            return []

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
        as_of: str | None = None,
    ) -> list[ChangeRecord]:
        def _load() -> list[dict[str, Any]]:
            return self._structural.get_change_history(
                pot_id,
                scope.function_name,
                scope.file_path,
                max(1, min(limit, 100)),
                repo_name=scope.repo_name,
                pr_number=scope.pr_number,
                as_of=as_of,
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
        focal: str | None = None
        if scope.services:
            try:
                focal = self._structural.resolve_entity_uuid_for_service_hint(
                    pot_id, scope.services[0]
                )
            except Exception as exc:
                logger.debug("resolve_entity_uuid_for_service_hint: %s", exc)
        if not focal and self._episodic.enabled:
            try:
                edges = await self._episodic.search_async(
                    pot_id,
                    query,
                    limit=1,
                    node_labels=_DEFAULT_NODE_LABELS,
                )
            except Exception as exc:
                logger.exception("get_causal_chain semantic seed failed: %s", exc)
                edges = []
            if edges:
                e0 = edges[0]
                focal = str(
                    getattr(e0, "target_node_uuid", None)
                    or getattr(e0, "source_node_uuid", None)
                    or ""
                ).strip() or None
        if not focal:
            return []

        def _load() -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
            chain = self._structural.walk_causal_chain_backward(
                pot_id,
                focal,
                max_depth=max_depth,
                as_of_iso=as_of_iso,
                window_days=window_days,
            )
            focal_row = self._structural.get_episodic_entity_node(pot_id, focal)
            return chain, focal_row

        try:
            chain, focal_row = await asyncio.to_thread(_load)
        except Exception as exc:
            logger.exception("get_causal_chain structural failed: %s", exc)
            return []

        def _ref_time(row: dict[str, Any]) -> str | None:
            raw = row.get("valid_at") or row.get("pred_valid_at")
            if raw is None:
                return None
            return str(raw)

        def _src_refs(row: dict[str, Any]) -> list[str]:
            ref = row.get("source_ref")
            return [str(ref)] if ref else []

        items: list[CausalChainItem] = []
        for i, row in enumerate(chain):
            rel = chain[i - 1].get("edge_name") if i > 0 else None
            uid = str(row.get("uuid") or "")
            if not uid:
                continue
            items.append(
                CausalChainItem(
                    node_uuid=uid,
                    name=row.get("name"),
                    summary=(row.get("summary") or "").strip() or None,
                    reference_time=_ref_time(row),
                    source_refs=_src_refs(row),
                    confidence=0.72,
                    relation_from_previous=str(rel) if rel else None,
                )
            )
        fr = focal_row or {}
        rel_f = chain[-1].get("edge_name") if chain else None
        items.append(
            CausalChainItem(
                node_uuid=str(fr.get("uuid") or focal),
                name=fr.get("name"),
                summary=(fr.get("summary") or "").strip() or None,
                reference_time=_ref_time(fr) if fr else None,
                source_refs=_src_refs(fr),
                confidence=0.75,
                relation_from_previous=str(rel_f) if rel_f else None,
            )
        )
        return items

    async def list_open_conflicts(self, pot_id: str) -> list[dict[str, Any]]:
        def _load() -> list[dict[str, Any]]:
            try:
                return self._episodic.list_open_conflicts(pot_id)
            except Exception as exc:
                logger.exception("list_open_conflicts failed: %s", exc)
                return []

        try:
            return await asyncio.to_thread(_load)
        except Exception as exc:
            logger.exception("list_open_conflicts thread failed: %s", exc)
            return []

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
            causal_chain_context=True,
        )

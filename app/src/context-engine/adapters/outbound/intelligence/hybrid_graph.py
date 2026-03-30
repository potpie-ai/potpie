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
    DecisionRecord,
    DiscussionRecord,
    OwnershipRecord,
)
from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.intelligence_provider import IntelligenceProvider
from domain.ports.structural_graph import StructuralGraphPort

logger = logging.getLogger(__name__)

_DEFAULT_NODE_LABELS = ["PullRequest", "Decision", "Issue", "Feature"]


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
        project_id: str,
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
                pot_id=project_id,
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
        project_id: str,
        artifact: ArtifactRef,
    ) -> ArtifactContext | None:
        if artifact.kind != "pr":
            return None
        try:
            num = int(artifact.identifier)
        except ValueError:
            return None

        def _load() -> dict[str, Any]:
            return self._structural.get_pr_review_context(project_id, num, repo_name=None)

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
        project_id: str,
        scope: ContextScope,
        *,
        limit: int = 10,
    ) -> list[ChangeRecord]:
        def _load() -> list[dict[str, Any]]:
            return self._structural.get_change_history(
                project_id,
                scope.function_name,
                scope.file_path,
                max(1, min(limit, 100)),
                repo_name=None,
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
        project_id: str,
        scope: ContextScope,
        *,
        limit: int = 20,
    ) -> list[DecisionRecord]:
        def _load() -> list[dict[str, Any]]:
            return self._structural.get_decisions(
                project_id,
                scope.file_path,
                scope.function_name,
                max(1, min(limit, 100)),
                repo_name=None,
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
        project_id: str,
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
                project_id, pr_num, repo_name=None
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
        project_id: str,
        scope: ContextScope,
        *,
        limit: int = 5,
    ) -> list[OwnershipRecord]:
        if not scope.file_path:
            return []

        def _load() -> list[dict[str, Any]]:
            return self._structural.get_file_owners(
                project_id,
                scope.file_path,
                max(1, min(limit, 50)),
                repo_name=None,
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

    def get_capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            semantic_search=self._episodic.enabled,
            artifact_context=True,
            change_history=True,
            decision_context=True,
            discussion_context=True,
            ownership_context=True,
            workflow_context=False,
        )

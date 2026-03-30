"""Orchestrates intelligence provider calls into a single IntelligenceBundle."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from domain.intelligence_models import (
    ArtifactContext,
    ChangeRecord,
    ContextResolutionRequest,
    CoverageReport,
    DecisionRecord,
    DiscussionRecord,
    IntelligenceBundle,
    OwnershipRecord,
    ResolutionError,
    ResolutionMeta,
)
from domain.intelligence_policy import EvidencePlan, build_evidence_plan
from domain.intelligence_signals import extract_signals
from domain.ports.intelligence_provider import IntelligenceProvider

logger = logging.getLogger(__name__)


def _compute_coverage(
    plan: EvidencePlan,
    bundle: IntelligenceBundle,
    errors: list[ResolutionError],
) -> CoverageReport:
    """Derive coverage from planned families vs non-empty results and errors."""
    available: list[str] = []
    missing: list[str] = []
    missing_reasons: dict[str, str] = {}

    def _has_semantic() -> bool:
        return bool(bundle.semantic_hits)

    def _has_artifact() -> bool:
        return bool(bundle.artifacts)

    def _has_changes() -> bool:
        return bool(bundle.changes)

    def _has_decisions() -> bool:
        return bool(bundle.decisions)

    def _has_discussions() -> bool:
        return bool(bundle.discussions)

    def _has_ownership() -> bool:
        return bool(bundle.ownership)

    checks: list[tuple[str, bool, bool]] = [
        ("semantic_search", plan.run_semantic_search, _has_semantic()),
        ("artifact_context", plan.run_artifact, _has_artifact()),
        ("change_history", plan.run_change_history, _has_changes()),
        ("decision_context", plan.run_decisions, _has_decisions()),
        ("discussion_context", plan.run_discussions, _has_discussions()),
        ("ownership_context", plan.run_ownership, _has_ownership()),
    ]

    err_by_source = {e.source: e for e in errors}

    for family, planned, has_data in checks:
        if not planned:
            continue
        if has_data:
            available.append(family)
            continue
        missing.append(family)
        if family in err_by_source:
            missing_reasons[family] = "error"
        else:
            missing_reasons[family] = "empty_result"

    status = "complete"
    if not available and (missing or errors):
        status = "empty"
    elif missing:
        status = "partial"

    for m in plan.mandatory:
        if m in missing:
            missing_reasons.setdefault(m, missing_reasons.get(m, "empty_result"))

    return CoverageReport(
        status=status,
        available=available,
        missing=missing,
        missing_reasons=missing_reasons,
    )


class ContextResolutionService:
    def __init__(self, provider: IntelligenceProvider) -> None:
        self._provider = provider

    async def resolve(self, request: ContextResolutionRequest) -> IntelligenceBundle:
        sig = extract_signals(request.query)
        caps = self._provider.get_capabilities()
        plan = build_evidence_plan(request, signals=sig, caps=caps)

        semantic_hits: list[dict[str, Any]] = []
        artifacts: list[ArtifactContext] = []
        changes: list[ChangeRecord] = []
        decisions: list[DecisionRecord] = []
        discussions: list[DiscussionRecord] = []
        ownership: list[OwnershipRecord] = []
        errors: list[ResolutionError] = []
        per_lat: dict[str, int] = {}
        capabilities_used: list[str] = []

        deadline = plan.timeout_budget_ms / 1000.0
        provider = self._provider

        async def _timed(name: str, coro: Any) -> tuple[str, Any]:
            t0 = time.perf_counter()
            try:
                res = await coro
                per_lat[name] = int((time.perf_counter() - t0) * 1000)
                capabilities_used.append(name)
                return name, res
            except Exception as exc:
                per_lat[name] = int((time.perf_counter() - t0) * 1000)
                logger.exception("%s failed: %s", name, exc)
                errors.append(
                    ResolutionError(
                        source=name,
                        error=str(exc),
                        recoverable=True,
                    )
                )
                return name, None

        coros: list[Any] = []

        if plan.run_semantic_search:
            coros.append(
                _timed(
                    "semantic_search",
                    provider.search_context(
                        request.project_id,
                        request.query,
                        limit=8,
                        node_labels=None,
                    ),
                )
            )

        if plan.run_artifact and plan.artifact_ref is not None:
            coros.append(
                _timed(
                    "artifact_context",
                    provider.get_artifact_context(
                        request.project_id,
                        plan.artifact_ref,
                    ),
                )
            )

        if plan.run_change_history:
            coros.append(
                _timed(
                    "change_history",
                    provider.get_change_history(
                        request.project_id,
                        plan.scope,
                        limit=10,
                    ),
                )
            )

        if plan.run_decisions:
            coros.append(
                _timed(
                    "decision_context",
                    provider.get_decision_context(
                        request.project_id,
                        plan.scope,
                        limit=20,
                    ),
                )
            )

        if plan.run_discussions:
            coros.append(
                _timed(
                    "discussion_context",
                    provider.get_related_discussions(
                        request.project_id,
                        plan.scope,
                        limit=10,
                    ),
                )
            )

        if plan.run_ownership:
            coros.append(
                _timed(
                    "ownership_context",
                    provider.get_ownership(
                        request.project_id,
                        plan.scope,
                        limit=5,
                    ),
                )
            )

        if coros:
            tasks = [asyncio.create_task(c) for c in coros]
            done: set[asyncio.Task[Any]]
            pending: set[asyncio.Task[Any]]
            done, pending = await asyncio.wait(tasks, timeout=deadline)

            # Cancel anything that didn't finish in the deadline, but keep completed results.
            if pending:
                for t in pending:
                    t.cancel()
                errors.append(
                    ResolutionError(
                        source="resolve",
                        error=f"Resolution timed out after {plan.timeout_budget_ms}ms",
                        recoverable=True,
                    )
                )

            # Collect completed outcomes (including exceptions).
            outcomes: list[Any] = []
            for t in done:
                try:
                    outcomes.append(await t)
                except asyncio.CancelledError:
                    continue
                except Exception as exc:
                    outcomes.append(exc)

            for item in outcomes:
                if isinstance(item, BaseException):
                    errors.append(
                        ResolutionError(
                            source="gather",
                            error=str(item),
                            recoverable=True,
                        )
                    )
                    continue
                if not isinstance(item, tuple) or len(item) != 2:
                    continue
                name, payload = item[0], item[1]
                if payload is None:
                    continue
                if name == "semantic_search" and isinstance(payload, list):
                    semantic_hits = payload
                elif name == "artifact_context" and isinstance(payload, ArtifactContext):
                    artifacts = [payload]
                elif name == "change_history" and isinstance(payload, list):
                    changes = list(payload)
                elif name == "decision_context" and isinstance(payload, list):
                    decisions = list(payload)
                elif name == "discussion_context" and isinstance(payload, list):
                    discussions = list(payload)
                elif name == "ownership_context" and isinstance(payload, list):
                    ownership = list(payload)

        total_ms = int(sum(per_lat.values()))
        meta = ResolutionMeta(
            provider=type(self._provider).__name__,
            total_latency_ms=total_ms,
            per_call_latency_ms=per_lat,
            capabilities_used=capabilities_used,
            schema_version="1",
        )

        bundle = IntelligenceBundle(
            request=request,
            semantic_hits=semantic_hits,
            artifacts=artifacts,
            changes=changes,
            decisions=decisions,
            discussions=discussions,
            ownership=ownership,
            errors=errors,
            meta=meta,
        )
        bundle.coverage = _compute_coverage(plan, bundle, errors)
        return bundle

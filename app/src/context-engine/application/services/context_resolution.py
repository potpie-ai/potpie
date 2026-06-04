"""Orchestrates intelligence provider calls into a single IntelligenceBundle."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from domain.intelligence_models import (
    ArtifactContext,
    CausalChainItem,
    ChangeRecord,
    ContextResolutionRequest,
    CoverageReport,
    DebuggingMemoryRecord,
    DecisionRecord,
    DiscussionRecord,
    IntelligenceBundle,
    OwnershipRecord,
    ProjectContextRecord,
    ResolutionError,
    ResolutionMeta,
)
from domain.agent_context_port import (
    FALLBACK_ONLY_INCLUDES,
    includes_for_request,
    unsupported_include_values,
)
from domain.graph_quality import (
    assess_graph_quality,
    freshness_ttl_hours_for_source_type,
    source_of_truth_for_source_type,
)
from domain.intelligence_policy import EvidencePlan, build_evidence_plan
from domain.intelligence_signals import extract_signals
from domain.ports.intelligence_provider import IntelligenceProvider
from domain.ports.source_resolver import SourceResolverPort
from domain.source_references import (
    assess_freshness,
    dedupe_source_references,
    normalize_source_policy,
    source_policy_fallbacks,
    source_ref_key,
    source_reference_from_mapping,
    SourceFallback,
    SourceReferenceRecord,
)
from domain.source_resolution import (
    ResolverAuthContext,
    ResolverBudget,
    ResolverFallback,
    SourceResolutionResult,
)

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

    def _has_project_map() -> bool:
        return bool(bundle.project_map)

    def _has_debugging_memory() -> bool:
        return bool(bundle.debugging_memory)

    def _has_causal_chain() -> bool:
        return bool(bundle.causal_chain)

    checks: list[tuple[str, bool, bool]] = [
        ("semantic_search", plan.run_semantic_search, _has_semantic()),
        ("artifact_context", plan.run_artifact, _has_artifact()),
        ("change_history", plan.run_change_history, _has_changes()),
        ("decision_context", plan.run_decisions, _has_decisions()),
        ("discussion_context", plan.run_discussions, _has_discussions()),
        ("ownership_context", plan.run_ownership, _has_ownership()),
        ("project_map_context", plan.run_project_map, _has_project_map()),
        (
            "debugging_memory_context",
            plan.run_debugging_memory,
            _has_debugging_memory(),
        ),
        (
            "causal_chain_context",
            plan.run_causal_chain,
            _has_causal_chain(),
        ),
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


def _infer_source_system(kind: str | None, url: str | None) -> str | None:
    """Pick a ``source_system`` so the composite resolver can dispatch.

    Phase 5 routes diff/snippet detail through ``artifact={kind:'pr'}`` plus
    ``source_policy=summary``/``snippets``. For that to work the artifact ref
    must carry a ``source_system`` the resolver chain matches on. PR-shaped
    artifacts default to GitHub today; other kinds inherit only if their URL
    obviously belongs to a known host.
    """
    k = (kind or "").lower()
    if k in {"pr", "pull_request", "pullrequest"}:
        return "github"
    if url:
        host = url.lower()
        if "github.com" in host:
            return "github"
        if "gitlab" in host:
            return "gitlab"
    return None


def _collect_source_references(
    request: ContextResolutionRequest,
    semantic_hits: list[dict[str, Any]],
    artifacts: list[ArtifactContext],
    changes: list[ChangeRecord],
    decisions: list[DecisionRecord],
    discussions: list[DiscussionRecord],
    project_map: list[ProjectContextRecord],
    debugging_memory: list[DebuggingMemoryRecord],
) -> list[SourceReferenceRecord]:
    refs: list[SourceReferenceRecord] = []
    for raw in request.scope.source_refs if request.scope else []:
        refs.append(
            SourceReferenceRecord(
                ref=raw,
                source_type=raw.split(":", 1)[0] if ":" in raw else "unknown",
                external_id=raw.split(":", 1)[1] if ":" in raw else raw,
            )
        )

    for hit in semantic_hits:
        ref = source_reference_from_mapping(hit)
        if ref is not None:
            refs.append(ref)

    for artifact in artifacts:
        artifact_system = _infer_source_system(artifact.kind, artifact.url)
        refs.append(
            SourceReferenceRecord(
                ref=source_ref_key(
                    artifact.kind,
                    artifact.identifier,
                    source_system=artifact_system,
                ),
                source_type=artifact.kind,
                source_system=artifact_system,
                external_id=artifact.identifier,
                uri=artifact.url,
                retrieval_uri=artifact.url,
                title=artifact.title,
                summary=artifact.summary,
                fetchable=bool(artifact.url),
                access="allowed" if artifact.url else "unknown",
                last_seen_at=artifact.created_at,
                verified_against=source_of_truth_for_source_type(artifact.kind),
                freshness_ttl_hours=freshness_ttl_hours_for_source_type(artifact.kind),
                freshness="needs_verification",
                sync_status="needs_resync",
                verification_state="unverified",
            )
        )

    for change in changes:
        ref_value = change.artifact_ref
        if not ref_value and change.pr_number is not None:
            ref_value = f"PR #{change.pr_number}"
        if not ref_value:
            continue
        change_system = "github" if change.pr_number is not None else None
        refs.append(
            SourceReferenceRecord(
                ref=source_ref_key("change", ref_value),
                source_type="change",
                source_system=change_system,
                external_id=ref_value,
                title=change.title,
                summary=change.summary,
                last_seen_at=change.date,
                verified_against=source_of_truth_for_source_type("change"),
                freshness_ttl_hours=freshness_ttl_hours_for_source_type("change"),
                freshness="needs_verification",
                sync_status="needs_resync",
                verification_state="unverified",
            )
        )

    for decision in decisions:
        ref_value = decision.source_ref
        if not ref_value and decision.pr_number is not None:
            ref_value = f"PR #{decision.pr_number}"
        if not ref_value:
            continue
        refs.append(
            SourceReferenceRecord(
                ref=source_ref_key("decision", ref_value),
                source_type="decision",
                external_id=ref_value,
                summary=decision.decision,
                verified_against=source_of_truth_for_source_type("decision"),
                freshness_ttl_hours=freshness_ttl_hours_for_source_type("decision"),
                freshness="needs_verification",
                sync_status="needs_resync",
                verification_state="unverified",
            )
        )

    for discussion in discussions:
        if not discussion.source_ref:
            continue
        refs.append(
            SourceReferenceRecord(
                ref=source_ref_key("discussion", discussion.source_ref),
                source_type="discussion",
                external_id=discussion.source_ref,
                title=discussion.headline,
                summary=discussion.summary,
                verified_against=source_of_truth_for_source_type("discussion"),
                freshness_ttl_hours=freshness_ttl_hours_for_source_type("discussion"),
                freshness="needs_verification",
                sync_status="needs_resync",
                verification_state="unverified",
            )
        )

    for item in project_map:
        ref_value = item.source_ref or item.source_uri
        if not ref_value:
            continue
        refs.append(
            SourceReferenceRecord(
                ref=source_ref_key(item.kind or item.family, ref_value),
                source_type=item.kind or item.family,
                external_id=ref_value,
                uri=item.source_uri,
                retrieval_uri=item.source_uri,
                title=item.name,
                summary=item.summary,
                fetchable=bool(item.source_uri),
                access="allowed" if item.source_uri else "unknown",
                verified_against=source_of_truth_for_source_type(item.kind),
                freshness_ttl_hours=freshness_ttl_hours_for_source_type(item.kind),
                freshness="needs_verification",
                sync_status="needs_resync",
                verification_state="unverified",
            )
        )

    for item in debugging_memory:
        ref_value = item.source_ref or item.source_uri
        if not ref_value:
            continue
        refs.append(
            SourceReferenceRecord(
                ref=source_ref_key(item.kind or item.family, ref_value),
                source_type=item.kind or item.family,
                external_id=ref_value,
                uri=item.source_uri,
                retrieval_uri=item.source_uri,
                title=item.title,
                summary=item.summary,
                fetchable=bool(item.source_uri),
                access="allowed" if item.source_uri else "unknown",
                verified_against=source_of_truth_for_source_type(item.kind),
                freshness_ttl_hours=freshness_ttl_hours_for_source_type(item.kind),
                freshness="needs_verification",
                sync_status="needs_resync",
                verification_state="unverified",
            )
        )

    return dedupe_source_references(refs)


def _recommended_next_actions(
    *,
    source_policy: str,
    fallbacks: list[SourceFallback],
) -> list[dict[str, Any]]:
    if source_policy == "verify" or not fallbacks:
        return []
    if any(
        fallback.code in {"source_unverified", "source_resolver_unavailable"}
        for fallback in fallbacks
    ):
        return [
            {
                "action": "resolve",
                "mode": "verify",
                "source_policy": "verify",
                "reason": "Verify source-backed facts before high-impact action.",
            }
        ]
    return []


def _resolver_fallbacks_as_source_fallbacks(
    result: SourceResolutionResult,
) -> list[SourceFallback]:
    """Surface resolver-level fallbacks under the same public shape.

    Resolver fallbacks live on ``bundle.source_resolution.fallbacks`` with a
    richer shape (codes, per-ref detail); they are also flattened into
    ``bundle.fallbacks`` so agent clients that only read ``fallbacks`` still
    see "source-backed read failed" signals.
    """
    return [
        SourceFallback(
            code=rf.code,
            message=rf.message,
            impact=rf.impact,
            ref=rf.ref,
        )
        for rf in result.fallbacks
    ]


def _apply_verifications_to_refs(
    refs: list[SourceReferenceRecord],
    result: SourceResolutionResult,
) -> None:
    """Update each ref's ``verification_state`` from resolver output."""
    by_ref = {v.ref: v for v in result.verifications}
    for ref in refs:
        v = by_ref.get(ref.ref)
        if v is None:
            continue
        ref.verification_state = v.verification_state
        if v.checked_at:
            ref.last_verified_at = v.checked_at


class ContextResolutionService:
    def __init__(
        self,
        provider: IntelligenceProvider,
        *,
        source_resolver: SourceResolverPort | None = None,
    ) -> None:
        self._provider = provider
        self._source_resolver = source_resolver

    @property
    def source_resolver(self) -> SourceResolverPort | None:
        return self._source_resolver

    def set_source_resolver(self, resolver: SourceResolverPort | None) -> None:
        """Late-bind a resolver (the host wires this after container build)."""
        self._source_resolver = resolver

    async def _run_source_resolver(
        self,
        *,
        pot_id: str,
        refs: list[SourceReferenceRecord],
        source_policy: str,
        request: ContextResolutionRequest,
    ) -> SourceResolutionResult:
        """Dispatch source-backed reads when policy != references_only.

        When no resolver is wired the method returns an empty result and
        relies on :func:`source_policy_fallbacks` to add the user-facing
        ``source_resolver_unavailable`` signal. When a resolver is wired it
        owns the fallback taxonomy and the service only times/guards it.
        """
        if source_policy == "references_only" or not refs:
            return SourceResolutionResult()
        resolver = self._source_resolver
        if resolver is None:
            return SourceResolutionResult()

        budget = ResolverBudget(
            timeout_ms=max(500, request.budget.timeout_ms // 2),
        )
        auth = ResolverAuthContext()
        try:
            task = asyncio.create_task(
                resolver.resolve(
                    pot_id=pot_id,
                    refs=refs,
                    source_policy=source_policy,
                    budget=budget,
                    auth=auth,
                )
            )
            return await asyncio.wait_for(task, timeout=budget.timeout_ms / 1000.0)
        except asyncio.TimeoutError:
            return SourceResolutionResult(
                fallbacks=[
                    ResolverFallback(
                        code="source_resolver_timeout",
                        message=f"Source resolver exceeded {budget.timeout_ms}ms.",
                    )
                ]
            )
        except Exception as exc:
            logger.exception("source_resolver failed: %s", exc)
            return SourceResolutionResult(
                fallbacks=[
                    ResolverFallback(
                        code="source_resolver_error",
                        message=f"Source resolver raised: {exc}",
                    )
                ]
            )

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
        project_map: list[ProjectContextRecord] = []
        debugging_memory: list[DebuggingMemoryRecord] = []
        causal_chain: list[CausalChainItem] = []
        errors: list[ResolutionError] = []
        per_lat: dict[str, int] = {}
        capabilities_used: list[str] = []

        deadline = plan.timeout_budget_ms / 1000.0
        provider = self._provider
        effective_max_items = (
            plan.max_items if plan.max_items is not None else request.effective_max_items
        )

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
                        request.pot_id,
                        request.query,
                        limit=effective_max_items,
                        node_labels=None,
                    ),
                )
            )

        if plan.run_artifact and plan.artifact_ref is not None:
            coros.append(
                _timed(
                    "artifact_context",
                    provider.get_artifact_context(
                        request.pot_id,
                        plan.artifact_ref,
                    ),
                )
            )

        if plan.run_change_history:
            coros.append(
                _timed(
                    "change_history",
                    provider.get_change_history(
                        request.pot_id,
                        plan.scope,
                        limit=effective_max_items,
                        as_of=request.as_of.isoformat() if request.as_of else None,
                        query=request.query or None,
                    ),
                )
            )

        if plan.run_decisions:
            coros.append(
                _timed(
                    "decision_context",
                    provider.get_decision_context(
                        request.pot_id,
                        plan.scope,
                        limit=max(effective_max_items, 20),
                        query=request.query or None,
                    ),
                )
            )

        if plan.run_discussions:
            coros.append(
                _timed(
                    "discussion_context",
                    provider.get_related_discussions(
                        request.pot_id,
                        plan.scope,
                        limit=effective_max_items,
                    ),
                )
            )

        if plan.run_ownership:
            coros.append(
                _timed(
                    "ownership_context",
                    provider.get_ownership(
                        request.pot_id,
                        plan.scope,
                        limit=min(effective_max_items, 10),
                    ),
                )
            )

        if plan.run_project_map:
            coros.append(
                _timed(
                    "project_map_context",
                    provider.get_project_map(
                        request.pot_id,
                        plan.scope,
                        include=plan.project_map_includes,
                        limit=effective_max_items,
                    ),
                )
            )

        if plan.run_debugging_memory:
            coros.append(
                _timed(
                    "debugging_memory_context",
                    provider.get_debugging_memory(
                        request.pot_id,
                        plan.scope,
                        include=plan.debugging_memory_includes,
                        query=request.query,
                        limit=effective_max_items,
                    ),
                )
            )

        if plan.run_causal_chain:
            depth = max(2, min(effective_max_items, 12))
            coros.append(
                _timed(
                    "causal_chain_context",
                    provider.get_causal_chain(
                        request.pot_id,
                        plan.scope,
                        query=request.query,
                        max_depth=depth,
                        as_of_iso=request.as_of.isoformat() if request.as_of else None,
                        window_days=180,
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
                elif name == "artifact_context" and isinstance(
                    payload, ArtifactContext
                ):
                    artifacts = [payload]
                elif name == "change_history" and isinstance(payload, list):
                    changes = list(payload)
                elif name == "decision_context" and isinstance(payload, list):
                    decisions = list(payload)
                elif name == "discussion_context" and isinstance(payload, list):
                    discussions = list(payload)
                elif name == "ownership_context" and isinstance(payload, list):
                    ownership = list(payload)
                elif name == "project_map_context" and isinstance(payload, list):
                    project_map = list(payload)
                elif name == "debugging_memory_context" and isinstance(payload, list):
                    debugging_memory = list(payload)
                elif name == "causal_chain_context" and isinstance(payload, list):
                    causal_chain = list(payload)

        total_ms = int(sum(per_lat.values()))
        meta = ResolutionMeta(
            provider=type(self._provider).__name__,
            total_latency_ms=total_ms,
            per_call_latency_ms=per_lat,
            capabilities_used=capabilities_used,
            schema_version="4",
        )
        source_policy = normalize_source_policy(
            plan.source_policy or request.source_policy
        )
        source_refs = _collect_source_references(
            request,
            semantic_hits,
            artifacts,
            changes,
            decisions,
            discussions,
            project_map,
            debugging_memory,
        )
        resolution_result = await self._run_source_resolver(
            pot_id=request.pot_id,
            refs=source_refs,
            source_policy=source_policy,
            request=request,
        )
        if resolution_result.verifications:
            _apply_verifications_to_refs(source_refs, resolution_result)
        freshness = assess_freshness(source_refs, evidence_rows=semantic_hits)
        fallbacks = source_policy_fallbacks(
            source_policy=source_policy,
            refs=source_refs,
        )
        fallbacks.extend(_resolver_fallbacks_as_source_fallbacks(resolution_result))
        requested_includes = set(
            includes_for_request(request.intent, request.include, request.exclude)
        )
        unsupported_includes = unsupported_include_values(list(requested_includes))
        for include in sorted(set(unsupported_includes)):
            fallbacks.append(
                SourceFallback(
                    code="unsupported_include",
                    message=f"The requested context family '{include}' is not recognized.",
                    impact="The resolver ignored this include value.",
                )
            )
        for include in sorted(requested_includes & FALLBACK_ONLY_INCLUDES):
            fallbacks.append(
                SourceFallback(
                    code="context_family_not_implemented",
                    message=(
                        f"The '{include}' context family is part of the public contract "
                        "but is not backed by a dedicated resolver yet."
                    ),
                    impact="Use semantic evidence and source references until this family is implemented.",
                )
            )

        bundle = IntelligenceBundle(
            request=request,
            semantic_hits=semantic_hits,
            artifacts=artifacts,
            changes=changes,
            decisions=decisions,
            discussions=discussions,
            ownership=ownership,
            project_map=project_map,
            debugging_memory=debugging_memory,
            causal_chain=causal_chain,
            source_refs=source_refs,
            freshness=freshness,
            fallbacks=fallbacks,
            source_resolution=resolution_result,
            recommended_next_actions=_recommended_next_actions(
                source_policy=source_policy,
                fallbacks=fallbacks,
            ),
            errors=errors,
            meta=meta,
        )
        bundle.coverage = _compute_coverage(plan, bundle, errors)
        bundle.quality = assess_graph_quality(
            refs=bundle.source_refs,
            coverage=bundle.coverage,
            fallbacks=bundle.fallbacks,
        )
        open_cf: list[dict[str, Any]] = []
        try:
            open_cf = await self._provider.list_open_conflicts(request.pot_id)
        except Exception as exc:
            logger.warning("list_open_conflicts failed: %s", exc)
        bundle.open_conflicts = open_cf
        bundle.quality.conflicts = open_cf
        if open_cf and any(
            isinstance(x, dict) and x.get("auto_resolvable") is False for x in open_cf
        ):
            bundle.recommended_next_actions.append(
                {
                    "action": "review_conflicts",
                    "reason": (
                        "Open predicate-family conflicts need human review or "
                        "`supersede_older` resolution before relying on those facts."
                    ),
                }
            )
        return bundle

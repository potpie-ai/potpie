"""Evidence planning from signals and provider capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.intelligence_models import (
    ArtifactRef,
    CapabilitySet,
    ContextResolutionRequest,
    ContextScope,
)
from domain.intelligence_signals import SignalSet, extract_signals
from domain.agent_context_port import (
    DEBUGGING_MEMORY_INCLUDES,
    PROJECT_MAP_INCLUDES,
    includes_for_request,
    normalize_context_values,
)


@dataclass
class EvidencePlan:
    """Which provider methods to invoke for this turn."""

    run_semantic_search: bool = False
    run_artifact: bool = False
    artifact_ref: ArtifactRef | None = None
    run_change_history: bool = False
    run_decisions: bool = False
    run_discussions: bool = False
    run_ownership: bool = False
    run_project_map: bool = False
    project_map_includes: list[str] = field(default_factory=list)
    run_debugging_memory: bool = False
    debugging_memory_includes: list[str] = field(default_factory=list)
    scope: ContextScope = field(default_factory=ContextScope)
    mandatory: list[str] = field(default_factory=list)
    timeout_budget_ms: int = 4000


def _merge_scope(
    request: ContextResolutionRequest,
    signals: SignalSet,
) -> ContextScope:
    scope = request.scope or ContextScope()
    fp = scope.file_path
    if not fp and signals.mentioned_file_paths:
        fp = signals.mentioned_file_paths[0]
    fn = scope.function_name
    if not fn and signals.mentioned_symbols:
        fn = signals.mentioned_symbols[0]
    pr = scope.pr_number
    if pr is None and signals.mentioned_pr is not None:
        pr = signals.mentioned_pr
    if request.artifact_ref and request.artifact_ref.kind == "pr":
        try:
            pr = int(request.artifact_ref.identifier)
        except ValueError:
            pass
    return ContextScope(
        repo_name=scope.repo_name,
        branch=scope.branch,
        file_path=fp,
        function_name=fn,
        symbol=scope.symbol,
        pr_number=pr,
        services=list(scope.services),
        features=list(scope.features),
        environment=scope.environment,
        ticket_ids=list(scope.ticket_ids),
        user=scope.user,
        source_refs=list(scope.source_refs),
    )


def build_evidence_plan(
    request: ContextResolutionRequest,
    signals: SignalSet | None = None,
    caps: CapabilitySet | None = None,
) -> EvidencePlan:
    """Build an evidence plan from the request and optional pre-extracted signals."""
    sig = signals or extract_signals(request.query)
    caps = caps or CapabilitySet()
    scope = _merge_scope(request, sig)
    timeout = max(500, min(request.effective_timeout_ms, 30_000))
    includes = set(
        includes_for_request(request.intent, request.include, request.exclude)
    )
    excludes = set(normalize_context_values(request.exclude))

    plan = EvidencePlan(scope=scope, timeout_budget_ms=timeout)

    # Semantic search: align with QnA prefetch — use when episodic layer is available
    plan.run_semantic_search = (
        bool(caps.semantic_search) and "semantic_search" not in excludes
    )

    if request.artifact_ref and caps.artifact_context and "artifact" not in excludes:
        plan.run_artifact = True
        plan.artifact_ref = request.artifact_ref

    wants_artifact = (
        scope.pr_number is not None
        or "artifact" in includes
        or "discussions" in includes
    )
    if scope.pr_number is not None and caps.artifact_context and wants_artifact:
        plan.run_artifact = True
        if plan.artifact_ref is None:
            plan.artifact_ref = ArtifactRef(kind="pr", identifier=str(scope.pr_number))

    if (
        scope.pr_number is not None
        and caps.discussion_context
        and "discussions" not in excludes
    ):
        plan.run_discussions = True

    project_map_includes = sorted(includes & PROJECT_MAP_INCLUDES)
    debugging_memory_includes = sorted(includes & DEBUGGING_MEMORY_INCLUDES)
    wants_changes = bool({"recent_changes", "prior_fixes"} & includes)
    wants_decisions = "decisions" in includes
    wants_ownership = "owners" in includes
    want_structural = (
        sig.needs_history
        or wants_changes
        or wants_decisions
        or wants_ownership
        or scope.file_path is not None
        or scope.function_name is not None
        or scope.pr_number is not None
    )
    if want_structural and caps.change_history and "recent_changes" not in excludes:
        plan.run_change_history = True
    if want_structural and caps.decision_context and "decisions" not in excludes:
        plan.run_decisions = True

    if (
        (sig.needs_ownership or wants_ownership)
        and scope.file_path
        and caps.ownership_context
    ):
        plan.run_ownership = True

    if project_map_includes and caps.project_map_context:
        plan.run_project_map = True
        plan.project_map_includes = project_map_includes

    if debugging_memory_includes and caps.debugging_memory_context:
        plan.run_debugging_memory = True
        plan.debugging_memory_includes = debugging_memory_includes

    if scope.pr_number is not None:
        plan.mandatory.extend(["artifact_context", "discussion_context"])
    elif sig.needs_history and (scope.file_path or scope.function_name):
        plan.mandatory.append("change_history")

    return plan

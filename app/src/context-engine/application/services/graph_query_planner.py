"""Compile ``ContextGraphQuery`` requests into ``ExecutionPlan`` objects.

The planner replaces the old named-handler dispatch in
``GraphitiContextGraphAdapter``. Each read request becomes a deterministic
sequence of :class:`QueryLeg` entries (semantic, temporal, exact,
traversal, aggregate, answer), which the adapter-side executor runs and
merges into one envelope with consistent provenance and fallbacks.
"""

from __future__ import annotations

from typing import Any

from domain.graph_query import (
    ContextGraphGoal,
    ContextGraphQuery,
    ContextGraphStrategy,
)
from domain.graph_query_plan import (
    ExecutionPlan,
    LegStrategy,
    MergePolicy,
    QueryLeg,
)


# Ordered family catalog: planning order is preserved in the execution
# plan so multi-family responses come back in a stable shape.
_FAMILY_ORDER: tuple[str, ...] = (
    "semantic_search",
    "change_history",
    "owners",
    "decisions",
    "pr_review_context",
    "pr_diff",
    "project_graph",
    "graph_overview",
)


class GraphQueryPlanner:
    """Turn a declarative ``ContextGraphQuery`` into an ``ExecutionPlan``."""

    def plan(self, request: ContextGraphQuery) -> ExecutionPlan:
        if request.goal == ContextGraphGoal.ANSWER:
            return ExecutionPlan(
                pot_id=request.pot_id,
                legs=(
                    QueryLeg(
                        name="answer",
                        family="resolve_context",
                        strategy=LegStrategy.ANSWER,
                        limit=request.limit,
                        as_of=request.as_of,
                    ),
                ),
                merge_policy=MergePolicy.SINGLE,
                budget=request.budget,
            )

        include = {str(x).strip() for x in request.include if str(x).strip()}
        legs: list[QueryLeg] = []
        fallbacks: list[dict[str, Any]] = []

        requested = self._requested_families(request, include)

        for family in _FAMILY_ORDER:
            if family not in requested:
                continue
            leg, fallback = self._build_leg(family, request)
            if leg is not None:
                legs.append(leg)
            if fallback is not None:
                fallbacks.append(fallback)

        # Unknown include tokens → explicit fallback, no silent drop.
        for token in sorted(include - set(_FAMILY_ORDER)):
            fallbacks.append(
                {
                    "family": token,
                    "reason": "unsupported_include",
                    "detail": (
                        f"'{token}' is not a known context graph evidence family"
                    ),
                }
            )

        if not legs:
            fallbacks.append(
                {
                    "family": "*",
                    "reason": "no_matching_family",
                    "detail": "request did not map to any known evidence family",
                }
            )

        merge_policy = MergePolicy.SINGLE if len(legs) <= 1 else MergePolicy.MULTI

        return ExecutionPlan(
            pot_id=request.pot_id,
            legs=tuple(legs),
            merge_policy=merge_policy,
            budget=request.budget,
            planner_fallbacks=tuple(fallbacks),
        )

    # ------------------------------------------------------------------
    # Family selection
    # ------------------------------------------------------------------
    def _requested_families(
        self,
        request: ContextGraphQuery,
        include: set[str],
    ) -> set[str]:
        """Collapse goal/strategy/include into the set of families to run."""
        families: set[str] = set(include)

        # Strategy/goal-driven auto-selection for backward compatibility
        # with requests that don't spell out ``include``.
        has_query = bool(request.query and request.query.strip())
        if (
            request.strategy
            in {ContextGraphStrategy.SEMANTIC, ContextGraphStrategy.HYBRID}
            and has_query
        ):
            families.add("semantic_search")

        if request.goal == ContextGraphGoal.TIMELINE:
            families.add("change_history")
        elif request.goal == ContextGraphGoal.NEIGHBORHOOD:
            families.add("project_graph")
        elif request.goal == ContextGraphGoal.AGGREGATE and not families:
            # AGGREGATE without an explicit include defaults to the graph
            # overview; aggregate + ``owners`` stays as owners-only.
            families.add("graph_overview")

        return families

    # ------------------------------------------------------------------
    # Leg builders
    # ------------------------------------------------------------------
    def _build_leg(
        self,
        family: str,
        request: ContextGraphQuery,
    ) -> tuple[QueryLeg | None, dict[str, Any] | None]:
        scope = request.scope
        if family == "semantic_search":
            if not (request.query and request.query.strip()):
                return None, {
                    "family": family,
                    "reason": "missing_query",
                    "detail": "semantic_search requires a non-empty query",
                }
            return (
                QueryLeg(
                    name="semantic_search",
                    family="semantic_search",
                    strategy=(
                        LegStrategy.HYBRID
                        if request.strategy == ContextGraphStrategy.HYBRID
                        else LegStrategy.SEMANTIC
                    ),
                    limit=request.limit,
                    as_of=request.as_of,
                ),
                None,
            )

        if family == "change_history":
            return (
                QueryLeg(
                    name="change_history",
                    family="change_history",
                    strategy=LegStrategy.TEMPORAL,
                    limit=request.limit,
                    as_of=request.as_of,
                    requires_scope=frozenset(),
                ),
                None,
            )

        if family == "owners":
            if not scope.file_path:
                return None, {
                    "family": family,
                    "reason": "missing_scope",
                    "detail": "owners requires scope.file_path",
                }
            return (
                QueryLeg(
                    name="owners",
                    family="owners",
                    strategy=LegStrategy.EXACT,
                    limit=min(request.limit, 50),
                    requires_scope=frozenset({"file_path"}),
                ),
                None,
            )

        if family == "decisions":
            return (
                QueryLeg(
                    name="decisions",
                    family="decisions",
                    strategy=LegStrategy.EXACT,
                    limit=request.limit,
                ),
                None,
            )

        if family == "pr_review_context":
            if not scope.pr_number:
                return None, {
                    "family": family,
                    "reason": "missing_scope",
                    "detail": "pr_review_context requires scope.pr_number",
                }
            return (
                QueryLeg(
                    name="pr_review_context",
                    family="pr_review_context",
                    strategy=LegStrategy.EXACT,
                    limit=request.limit,
                    requires_scope=frozenset({"pr_number"}),
                ),
                None,
            )

        if family == "pr_diff":
            # pr_diff is compat-only: full diffs belong behind source
            # resolvers (see planning-next-steps.md phase 5). The planner
            # still schedules it for legacy callers, but marks the leg
            # so the response can surface a deprecation hint.
            if not scope.pr_number:
                return None, {
                    "family": family,
                    "reason": "missing_scope",
                    "detail": "pr_diff requires scope.pr_number",
                }
            return (
                QueryLeg(
                    name="pr_diff",
                    family="pr_diff",
                    strategy=LegStrategy.EXACT,
                    limit=request.limit,
                    requires_scope=frozenset({"pr_number"}),
                    compat=True,
                ),
                None,
            )

        if family == "project_graph":
            return (
                QueryLeg(
                    name="project_graph",
                    family="project_graph",
                    strategy=LegStrategy.TRAVERSAL,
                    limit=min(request.limit, 50),
                ),
                None,
            )

        if family == "graph_overview":
            return (
                QueryLeg(
                    name="graph_overview",
                    family="graph_overview",
                    strategy=LegStrategy.EXACT,
                    limit=min(request.limit, 100),
                ),
                None,
            )

        return None, {
            "family": family,
            "reason": "unsupported_include",
            "detail": f"no executor for family '{family}'",
        }


__all__ = ["GraphQueryPlanner"]

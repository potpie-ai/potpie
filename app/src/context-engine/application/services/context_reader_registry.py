"""Context reader registry — Phase 3 router.

Replaces ``GraphQueryPlanner`` and the per-family executor dispatch.
Resolve a family set from ``include`` + goal/strategy auto-selection,
ask each registered :class:`ContextReader` to run, merge the results.
The application layer never branches on family names.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from domain.context_reader import (
    ReaderCapability,
    ReaderManifestEntry,
    ReaderResult,
    RouterFallback,
)
from domain.graph_query import (
    ContextGraphGoal,
    ContextGraphQuery,
    ContextGraphResult,
    ContextGraphStrategy,
)
from bootstrap.observability_runtime import get_observability
from domain.ports.context_reader import ContextReaderPort
from domain.ports.observability import SPAN_KIND_INTERNAL

logger = logging.getLogger(__name__)


_GOAL_AUTO_FAMILIES: dict[ContextGraphGoal, tuple[str, ...]] = {
    ContextGraphGoal.NEIGHBORHOOD: ("project_graph",),
    ContextGraphGoal.AGGREGATE: ("graph_overview",),
}


class ContextReaderRegistry:
    def __init__(self) -> None:
        self._by_family: dict[str, ContextReaderPort] = {}
        self._order: list[str] = []

    # -- registration ----------------------------------------------------
    def register(self, reader: ContextReaderPort) -> None:
        family = reader.family().strip().lower()
        if not family:
            raise ValueError("ContextReader.family() must be non-empty")
        if family in self._by_family:
            raise ValueError(f"Reader for family {family!r} already registered")
        self._by_family[family] = reader
        self._order.append(family)

    def get(self, family: str) -> ContextReaderPort | None:
        return self._by_family.get(family.strip().lower())

    def families(self) -> tuple[str, ...]:
        return tuple(self._order)

    def all(self) -> Sequence[ContextReaderPort]:
        return tuple(self._by_family[f] for f in self._order)

    def capabilities(self) -> tuple[ReaderCapability, ...]:
        return tuple(self._by_family[f].capability() for f in self._order)

    def manifest(self) -> tuple[ReaderManifestEntry, ...]:
        out: list[ReaderManifestEntry] = []
        for family in self._order:
            cap = self._by_family[family].capability()
            out.append(
                ReaderManifestEntry(
                    family=cap.family,
                    description=cap.description,
                    intents=tuple(sorted(cap.intents)),
                    requires_scope=tuple(sorted(cap.requires_scope)),
                    cost=cap.cost.label,
                    backend=cap.backend,
                )
            )
        return tuple(out)

    # -- routing ---------------------------------------------------------
    def execute(self, request: ContextGraphQuery) -> ContextGraphResult:
        families, fallbacks = self._resolve_families(request)
        obs = get_observability()

        results: list[ReaderResult] = []
        for family in families:
            reader = self._by_family[family]
            cap = reader.capability()
            missing = self._missing_scope(cap, request)
            if missing:
                obs.counter(
                    "ce.resolve.reader_fallback_total",
                    1,
                    attributes={"family": family, "reason": "missing_scope"},
                )
                fallbacks.append(
                    RouterFallback(
                        family=family,
                        reason="missing_scope",
                        detail=f"{family} requires scope.{'/'.join(missing)}",
                    )
                )
                continue
            try:
                with obs.span(
                    f"reader.{family}",
                    kind=SPAN_KIND_INTERNAL,
                    attributes={"reader.family": family},
                ) as _rspan:
                    outcome = reader.read(request)
                    _rspan.set_attribute("reader.count", outcome.count)
                    if outcome.fallback_reason:
                        _rspan.set_attribute(
                            "reader.fallback", outcome.fallback_reason
                        )
            except Exception as exc:  # noqa: BLE001
                logger.exception("reader %s failed", family)
                obs.counter(
                    "ce.resolve.reader_fallback_total",
                    1,
                    attributes={"family": family, "reason": "executor_error"},
                )
                results.append(
                    ReaderResult(
                        family=family,
                        error=str(exc) or exc.__class__.__name__,
                        fallback_reason="executor_error",
                        compat=cap.compat,
                    )
                )
                continue
            if outcome.fallback_reason:
                obs.counter(
                    "ce.resolve.reader_fallback_total",
                    1,
                    attributes={
                        "family": family,
                        "reason": outcome.fallback_reason,
                    },
                )
            results.append(
                ReaderResult(
                    family=outcome.family or family,
                    result=outcome.result,
                    count=outcome.count,
                    error=outcome.error,
                    fallback_reason=outcome.fallback_reason,
                    compat=outcome.compat or cap.compat,
                )
            )

        return self._merge(request, results, fallbacks)

    # -- family selection ------------------------------------------------
    def _resolve_families(
        self, request: ContextGraphQuery
    ) -> tuple[list[str], list[RouterFallback]]:
        explicit: list[str] = []
        seen: set[str] = set()
        unsupported: list[str] = []
        for raw in request.include or []:
            family = str(raw).strip().lower()
            if not family or family in seen:
                continue
            seen.add(family)
            (explicit if family in self._by_family else unsupported).append(family)

        auto: set[str] = set()
        has_query = bool(request.query and request.query.strip())
        if (
            request.strategy in {ContextGraphStrategy.SEMANTIC, ContextGraphStrategy.HYBRID}
            and has_query
            and "semantic_search" in self._by_family
        ):
            auto.add("semantic_search")
        for fam in _GOAL_AUTO_FAMILIES.get(request.goal, ()):
            if not explicit and fam in self._by_family:
                auto.add(fam)
        if request.goal == ContextGraphGoal.TIMELINE:
            scope = request.scope
            code_scoped = bool(
                (scope.file_path and scope.file_path.strip())
                or (scope.function_name and scope.function_name.strip())
                or scope.pr_number is not None
            )
            target = "change_history" if code_scoped and "timeline" not in seen else "timeline"
            if target in self._by_family:
                auto.add(target)

        wanted = set(explicit) | auto
        ordered = [family for family in self._order if family in wanted]

        fallbacks = [
            RouterFallback(
                family=family,
                reason="unsupported_include",
                detail=f"'{family}' is not a registered evidence family",
            )
            for family in unsupported
        ]
        return ordered, fallbacks

    @staticmethod
    def _missing_scope(
        cap: ReaderCapability, request: ContextGraphQuery
    ) -> list[str]:
        if not cap.requires_scope:
            return []
        scope = request.scope
        present = {
            "file_path": bool(scope.file_path and scope.file_path.strip()),
            "function_name": bool(scope.function_name and scope.function_name.strip()),
            "pr_number": scope.pr_number is not None,
            "repo_name": bool(scope.repo_name and scope.repo_name.strip()),
            "branch": bool(scope.branch and scope.branch.strip()),
            "user": bool(scope.user and scope.user.strip()),
            "query": bool(request.query and request.query.strip()),
        }
        return [field for field in cap.requires_scope if not present.get(field)]

    # -- merge -----------------------------------------------------------
    @staticmethod
    def _merge(
        request: ContextGraphQuery,
        outcomes: list[ReaderResult],
        fallbacks: list[RouterFallback],
    ) -> ContextGraphResult:
        legs_meta: list[dict[str, Any]] = []
        for r in outcomes:
            entry: dict[str, Any] = {"family": r.family, "compat": r.compat}
            if r.count is not None:
                entry["count"] = r.count
            if r.error is not None:
                entry["error"] = r.error
            legs_meta.append(entry)
            if r.error or r.fallback_reason:
                fallbacks.append(
                    RouterFallback(
                        family=r.family,
                        reason=r.fallback_reason or "executor_error",
                        detail=r.error or "",
                    )
                )

        fb_payload = [
            {"family": f.family, "reason": f.reason, "detail": f.detail}
            for f in fallbacks
        ]

        if not outcomes:
            empty_meta: dict[str, Any] = {
                "include": list(request.include),
                "fallbacks": fb_payload,
            }
            return ContextGraphResult(
                kind=request.goal.value,
                goal=request.goal.value,
                strategy=request.strategy.value,
                error="unsupported_context_graph_query",
                meta=empty_meta,
            )

        if len(outcomes) == 1:
            r = outcomes[0]
            single_meta: dict[str, Any] = {"legs": legs_meta}
            if fb_payload:
                single_meta["fallbacks"] = fb_payload
            if r.compat:
                single_meta["compat"] = True
            return ContextGraphResult(
                kind=r.family,
                goal=request.goal.value,
                strategy=request.strategy.value,
                result=r.result,
                error=r.error,
                meta=single_meta,
            )

        results_by_family = {r.family: r.result for r in outcomes if r.error is None}
        multi_meta: dict[str, Any] = {"legs": legs_meta, "merge": "multi"}
        if fb_payload:
            multi_meta["fallbacks"] = fb_payload
        return ContextGraphResult(
            kind="multi",
            goal=request.goal.value,
            strategy=request.strategy.value,
            result=results_by_family,
            meta=multi_meta,
        )


__all__ = ["ContextReaderRegistry"]

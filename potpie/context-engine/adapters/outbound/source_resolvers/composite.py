"""Composite source resolver that dispatches refs to registered children.

The composite picks a child resolver per ref by matching ``source_system``
(falling back to ``source_type``) against each child's declared capabilities.
Refs with no matching child yield an ``unsupported_source_type`` fallback.
This lets the host wire a list of focused resolvers (one for GitHub, one for
docs, etc.) without the resolution service needing to know about any of them.

Budget enforcement is global: once the aggregate character cap is reached, the
composite stops dispatching and emits ``budget_exceeded`` for the remainder.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

from domain.ports.source_resolver import SourceResolverPort
from domain.source_references import SourceReferenceRecord, normalize_source_policy
from domain.source_resolution import (
    BUDGET_EXCEEDED,
    RESOLVER_ERROR,
    UNSUPPORTED_SOURCE_POLICY,
    UNSUPPORTED_SOURCE_TYPE,
    ResolverAuthContext,
    ResolverBudget,
    ResolverCapabilityEntry,
    ResolverFallback,
    SourceResolutionResult,
)

logger = logging.getLogger(__name__)


class CompositeSourceResolver:
    """Fan-out across provider-specific resolvers."""

    def __init__(self, children: Iterable[SourceResolverPort]) -> None:
        self._children: list[SourceResolverPort] = list(children)

    def _match_for_ref(
        self,
        ref: SourceReferenceRecord,
        policy: str,
    ) -> SourceResolverPort | None:
        """Return the first child whose capabilities cover this ref + policy."""
        provider_key = (ref.source_system or ref.source_type or "").lower()
        type_key = (ref.source_type or "").lower()
        for child in self._children:
            for cap in child.capabilities():
                if policy not in cap.policies:
                    continue
                cap_provider = cap.provider.lower()
                cap_kind = cap.source_kind.lower()
                if cap_provider in {provider_key, type_key} or cap_kind in {
                    provider_key,
                    type_key,
                }:
                    return child
        return None

    async def resolve(
        self,
        *,
        pot_id: str,
        refs: Sequence[SourceReferenceRecord],
        source_policy: str,
        budget: ResolverBudget,
        auth: ResolverAuthContext,
    ) -> SourceResolutionResult:
        policy = normalize_source_policy(source_policy)
        out = SourceResolutionResult()
        if policy == "references_only":
            return out

        # Group refs by chosen child so each child sees one call with its slice.
        unmatched: list[SourceReferenceRecord] = []
        by_child: dict[int, list[SourceReferenceRecord]] = {}
        child_index: dict[int, SourceResolverPort] = {}
        for ref in refs:
            child = self._match_for_ref(ref, policy)
            if child is None:
                unmatched.append(ref)
                continue
            key = id(child)
            by_child.setdefault(key, []).append(ref)
            child_index[key] = child

        for ref in unmatched:
            out.fallbacks.append(
                ResolverFallback(
                    code=UNSUPPORTED_SOURCE_TYPE,
                    message=(
                        f"No resolver registered for source_type={ref.source_type!r} "
                        f"under policy={policy!r}."
                    ),
                    ref=ref.ref,
                    source_type=ref.source_type,
                )
            )

        remaining_chars = budget.max_total_chars
        refs_served = 0

        for key, child_refs in by_child.items():
            child = child_index[key]
            # Clamp the per-child ref slice to the global max_refs budget.
            allowed = max(0, budget.max_refs - refs_served)
            if allowed <= 0:
                for ref in child_refs:
                    out.fallbacks.append(
                        ResolverFallback(
                            code=BUDGET_EXCEEDED,
                            message="Request hit the max_refs resolver budget.",
                            ref=ref.ref,
                            source_type=ref.source_type,
                        )
                    )
                continue
            served = child_refs[:allowed]
            overflow = child_refs[allowed:]
            for ref in overflow:
                out.fallbacks.append(
                    ResolverFallback(
                        code=BUDGET_EXCEEDED,
                        message="Request hit the max_refs resolver budget.",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
            # Give the child a tightened budget so max_total_chars stays global.
            child_budget = ResolverBudget(
                max_refs=len(served),
                max_chars_per_item=budget.max_chars_per_item,
                max_total_chars=max(0, remaining_chars),
                max_snippets_per_ref=budget.max_snippets_per_ref,
                timeout_ms=budget.timeout_ms,
            )
            try:
                child_result = await child.resolve(
                    pot_id=pot_id,
                    refs=served,
                    source_policy=policy,
                    budget=child_budget,
                    auth=auth,
                )
            except NotImplementedError as exc:
                out.fallbacks.append(
                    ResolverFallback(
                        code=UNSUPPORTED_SOURCE_POLICY,
                        message=str(exc) or f"policy={policy!r} not implemented",
                    )
                )
                continue
            except Exception as exc:
                logger.exception("source resolver child failed: %s", exc)
                out.fallbacks.append(
                    ResolverFallback(
                        code=RESOLVER_ERROR,
                        message=f"Resolver raised: {exc}",
                    )
                )
                continue

            out.extend(child_result)
            refs_served += len(served)
            remaining_chars = max(0, remaining_chars - child_result.total_chars())
            if remaining_chars <= 0:
                # Stop dispatching further children; emit budget_exceeded for the rest.
                break

        return out

    def capabilities(self) -> Sequence[ResolverCapabilityEntry]:
        merged: dict[tuple[str, str], set[str]] = {}
        for child in self._children:
            for cap in child.capabilities():
                key = (cap.provider.lower(), cap.source_kind.lower())
                merged.setdefault(key, set()).update(cap.policies)
        return [
            ResolverCapabilityEntry(
                provider=provider,
                source_kind=source_kind,
                policies=frozenset(policies),
            )
            for (provider, source_kind), policies in sorted(merged.items())
        ]

"""Null source resolver used when no real resolver is wired.

For any non-``references_only`` policy the null resolver emits a single
``source_resolver_unavailable`` fallback so agents see a clear signal instead
of silent success. It intentionally reports no capabilities so
``context_status`` tells consumers which policies are not backed on this server.
"""

from __future__ import annotations

from typing import Sequence

from domain.source_references import SourceReferenceRecord, normalize_source_policy
from domain.source_resolution import (
    RESOLVER_UNAVAILABLE,
    ResolverAuthContext,
    ResolverBudget,
    ResolverCapabilityEntry,
    ResolverFallback,
    SourceResolutionResult,
)


class NullSourceResolver:
    """Default resolver — never fulfills a policy, always emits a fallback."""

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
        if policy == "references_only":
            return SourceResolutionResult()
        return SourceResolutionResult(
            fallbacks=[
                ResolverFallback(
                    code=RESOLVER_UNAVAILABLE,
                    message=(
                        f"No source resolver is configured on this server for "
                        f"policy={policy!r}."
                    ),
                    impact="Use the returned source references to inspect sources manually.",
                )
            ]
        )

    def capabilities(self) -> Sequence[ResolverCapabilityEntry]:
        return ()

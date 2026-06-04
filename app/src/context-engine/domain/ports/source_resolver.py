"""Port for source-backed resolvers sitting behind ``context_resolve``.

``source_policy`` modes ``summary``, ``verify``, and ``snippets`` are fulfilled
by implementations of :class:`SourceResolverPort`. The resolver receives the
source refs collected from the graph plus the policy, budget, and caller auth
context; it returns a structured :class:`SourceResolutionResult` with any
per-ref fallbacks. The resolution service merges the output into the agent
envelope.

Resolvers also advertise which ``(provider, source_kind, policy)`` triples
they can fulfill via :meth:`capabilities`, which lets
``POST /api/v2/context/status`` and the baseline ``context_status`` capability
matrix report real availability rather than the default placeholder.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from domain.source_references import SourceReferenceRecord
from domain.source_resolution import (
    ResolverAuthContext,
    ResolverBudget,
    ResolverCapabilityEntry,
    SourceResolutionResult,
)


class SourceResolverPort(Protocol):
    """Resolve source refs under a given ``source_policy``.

    Implementations must:

    * Respect ``budget`` — stop once caps are hit and emit a
      ``budget_exceeded`` fallback for unresolved refs.
    * Never raise on partial failure — return a structured fallback instead.
    * Ignore refs they cannot handle (callers use :meth:`capabilities` for
      discovery; unresolved refs do not error, they become fallbacks the
      composite layer collapses).
    """

    async def resolve(
        self,
        *,
        pot_id: str,
        refs: Sequence[SourceReferenceRecord],
        source_policy: str,
        budget: ResolverBudget,
        auth: ResolverAuthContext,
    ) -> SourceResolutionResult:
        ...

    def capabilities(self) -> Sequence[ResolverCapabilityEntry]:
        """Advertise supported ``(provider, source_kind, policy)`` triples.

        Empty sequence means the resolver does not support anything (the
        default ``NullSourceResolver`` behavior). Returned entries are
        consumed by ``context_status`` to surface real availability.
        """
        ...

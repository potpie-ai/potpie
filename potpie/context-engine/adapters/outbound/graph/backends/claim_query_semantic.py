"""SemanticSearchPort backed by a ClaimQueryPort."""

from __future__ import annotations

from dataclasses import dataclass

from domain.ports.claim_query import ClaimQueryFilter, ClaimQueryPort, ClaimRow


@dataclass(slots=True)
class ClaimQuerySemanticSearch:
    """Route explicit semantic search through ``ClaimQueryFilter.fact_query``."""

    claim_query: ClaimQueryPort

    def search(
        self,
        *,
        pot_id: str,
        query: str,
        k: int = 10,
        filter_: ClaimQueryFilter | None = None,
    ) -> list[ClaimRow]:
        base = filter_ or ClaimQueryFilter(pot_id=pot_id)
        return self.claim_query.find_claims(
            ClaimQueryFilter(
                pot_id=pot_id,
                predicate_in=base.predicate_in,
                subject_key_in=base.subject_key_in,
                object_key_in=base.object_key_in,
                subject_label=base.subject_label,
                object_label=base.object_label,
                valid_at_after=base.valid_at_after,
                valid_at_before=base.valid_at_before,
                include_invalidated=base.include_invalidated,
                as_of=base.as_of,
                source_system_in=base.source_system_in,
                fact_query=query,
                limit=k,
            )
        )


__all__ = ["ClaimQuerySemanticSearch"]

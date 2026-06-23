"""SemanticSearchPort backed by a ClaimQueryPort."""

from __future__ import annotations

from dataclasses import dataclass, replace

from potpie.context_engine.domain.ports.claim_query import ClaimQueryFilter, ClaimQueryPort, ClaimRow


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
        # Preserve every incoming filter axis (claim_key_in / subgraph_in /
        # mutation_id_in included); only override pot_id + the semantic anchor.
        return self.claim_query.find_claims(
            replace(base, pot_id=pot_id, fact_query=query, limit=k)
        )


__all__ = ["ClaimQuerySemanticSearch"]

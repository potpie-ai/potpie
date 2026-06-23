"""``SemanticSearchPort`` — vector retrieval projection of a ``GraphBackend``.

A rebuildable projection: claims carry ``fact_embedding`` in the canonical
store, and this port answers nearest-neighbour queries over them. The embedded
``fact_query`` field on :class:`ClaimQueryFilter` is the inline path; this port
is the explicit semantic surface used by ``context_search`` and readers that
want similarity ordering independent of structural filters.
"""

from __future__ import annotations

from typing import Protocol

from potpie.context_engine.domain.ports.claim_query import ClaimQueryFilter, ClaimRow


class SemanticSearchPort(Protocol):
    """Nearest-neighbour retrieval over claim embeddings."""

    def search(
        self,
        *,
        pot_id: str,
        query: str,
        k: int = 10,
        filter_: ClaimQueryFilter | None = None,
    ) -> list[ClaimRow]:
        """Return up to ``k`` claims most similar to ``query``, optionally
        narrowed by ``filter_``. Similarity is stamped onto
        ``ClaimRow.properties['semantic_similarity']``."""
        ...


__all__ = ["SemanticSearchPort"]

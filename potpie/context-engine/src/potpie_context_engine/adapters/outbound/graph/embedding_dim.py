"""Single source of truth for vector-index width.

The vector index a writer creates must be the same width as the vectors the
configured embedder emits: a wider or narrower index silently degrades every
semantic read to the lexical fallback. Both graph writers (Neo4j and
FalkorDB) resolve the width through :func:`resolve_embedding_dim`, so index
creation and embedding writes can never disagree — and no width is ever
guessed when there is no embedder to ask.
"""

from __future__ import annotations

from typing import Any


def resolve_embedding_dim(embedder: Any | None) -> int | None:
    """Return the vector width declared by the wired embedder.

    - ``None`` when no embedder is wired: the caller must skip vector-index
      creation rather than guess a width. A wrong-width index is worse than
      no index — without one, semantic reads take the *labeled* lexical
      fallback instead of silently returning junk.
    - ``ValueError`` when an embedder is wired but does not declare a
      positive integer ``dimensions``: an embedder that cannot state its
      width is misconfigured, and guessing one would hide that.
    """
    if embedder is None:
        return None
    dim = getattr(embedder, "dimensions", None)
    if dim is None:
        raise ValueError(
            "wired embedder does not declare `dimensions`; refusing to guess "
            "a vector index width"
        )
    dim = int(dim)
    if dim <= 0:
        raise ValueError(
            f"embedder declares non-positive dimensions ({dim}); refusing to "
            "create a vector index at that width"
        )
    return dim


__all__ = ["resolve_embedding_dim"]

"""Repair vocabulary + selection for the claim fact-embedding index.

The re-embed pass is the recovery (and migration) path for claims whose
embeddings are missing (a failed attach), stale (written by a different
embedder model), or mis-sized (written at different dimensions — e.g. after
switching from the 256-dim hashing embedder to a sentence-transformers model).
Backend-specific mechanics (vecf32, index DDL) live with each backend; this
module owns the shared target vocabulary and the needs-repair predicate.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

SEMANTIC_INDEX_TARGET = "semantic_index"
SEMANTIC_INDEX_TARGETS = frozenset(
    {
        SEMANTIC_INDEX_TARGET,
        "semantic-index",
        "semantic",
        "embeddings",
        "vectors",
        "vector_index",
    }
)

SEMANTIC_INDEX_REPAIR_LIMIT = 100_000


def wants_semantic_index_repair(targets: Sequence[str] = ()) -> bool:
    """Return true when a repair invocation should re-embed stale claims."""
    if not targets:
        return True
    return any(t.strip().lower() in SEMANTIC_INDEX_TARGETS for t in targets)


def claim_needs_reembed(
    props: Mapping[str, Any], *, embedder_name: str, embedder_dim: int
) -> bool:
    """Does a stored claim's embedding disagree with the active embedder?"""
    if props.get("fact_embedding") is None:
        return True
    if str(props.get("embedding_model") or "") != embedder_name:
        return True
    try:
        stored_dim = int(props.get("embedding_dim") or 0)
    except (TypeError, ValueError):
        return True
    return stored_dim != int(embedder_dim)


def stored_dim_mismatch(
    props: Mapping[str, Any], *, embedder_dim: int
) -> bool:
    """True when a stored embedding was written at a different dimension."""
    if props.get("fact_embedding") is None:
        return False
    try:
        stored_dim = int(props.get("embedding_dim") or 0)
    except (TypeError, ValueError):
        return False
    return stored_dim not in (0, int(embedder_dim))


__all__ = [
    "SEMANTIC_INDEX_REPAIR_LIMIT",
    "SEMANTIC_INDEX_TARGET",
    "SEMANTIC_INDEX_TARGETS",
    "claim_needs_reembed",
    "stored_dim_mismatch",
    "wants_semantic_index_repair",
]

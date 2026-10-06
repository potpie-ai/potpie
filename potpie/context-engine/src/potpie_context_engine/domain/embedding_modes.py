"""Shared embedding mode vocabulary for setup and local embedder wiring."""

from __future__ import annotations

DISABLED_EMBEDDER_ALIASES = frozenset(
    {
        "none",
        "off",
        "lexical",
        "disabled",
        "0",
        "false",
    }
)

HASHING_EMBEDDER_ALIASES = frozenset(
    {
        "",
        "local",
        "hashing",
        "default",
        "on",
        "1",
        "true",
    }
)

AUTO_SENTENCE_TRANSFORMER_ALIASES = frozenset({"auto", "best", "semantic"})

EXPLICIT_SENTENCE_TRANSFORMER_ALIASES = frozenset(
    {
        "legacy",
        "sentence-transformers",
        "sentence_transformers",
        "sbert",
        "minilm",
        "all-minilm-l6-v2",
    }
)

SEMANTIC_EMBEDDER_ALIASES = (
    AUTO_SENTENCE_TRANSFORMER_ALIASES | EXPLICIT_SENTENCE_TRANSFORMER_ALIASES
)

EMBEDDING_MODEL_PREP_SKIPPED_ALIASES = DISABLED_EMBEDDER_ALIASES | frozenset(
    {
        "local",
        "hashing",
        "default",
    }
)

#: The repair when a semantic mode is selected but sentence-transformers is not
#: installed. The `potpie` base install leaves it out (it pulls in torch), so
#: this is the expected state of a bare install, not a fault.
SEMANTIC_EMBEDDINGS_INSTALL_HINT = (
    "install the embeddings extra for semantic search: "
    "`potpie[embeddings]` (or `potpie-context-engine[embeddings]`)"
)


def normalize_embedding_mode(value: str | None) -> str:
    return (value or "").strip().lower().replace("_", "-")


__all__ = [
    "AUTO_SENTENCE_TRANSFORMER_ALIASES",
    "DISABLED_EMBEDDER_ALIASES",
    "EMBEDDING_MODEL_PREP_SKIPPED_ALIASES",
    "EXPLICIT_SENTENCE_TRANSFORMER_ALIASES",
    "HASHING_EMBEDDER_ALIASES",
    "SEMANTIC_EMBEDDER_ALIASES",
    "SEMANTIC_EMBEDDINGS_INSTALL_HINT",
    "normalize_embedding_mode",
]

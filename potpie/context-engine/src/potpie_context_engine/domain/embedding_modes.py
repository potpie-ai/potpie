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


def normalize_embedding_mode(value: str | None) -> str:
    return (value or "").strip().lower().replace("_", "-")


__all__ = [
    "AUTO_SENTENCE_TRANSFORMER_ALIASES",
    "DISABLED_EMBEDDER_ALIASES",
    "EMBEDDING_MODEL_PREP_SKIPPED_ALIASES",
    "EXPLICIT_SENTENCE_TRANSFORMER_ALIASES",
    "HASHING_EMBEDDER_ALIASES",
    "SEMANTIC_EMBEDDER_ALIASES",
    "normalize_embedding_mode",
]

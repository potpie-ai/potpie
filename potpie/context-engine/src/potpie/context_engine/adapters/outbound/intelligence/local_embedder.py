"""Context-graph embedders.

The Trigger Model non-negotiable: retrieval must work with **no API key and no
external embeddings service**, on the user's machine. This is the default OSS
embedder. It is a real local embedding model in the feature-hashing family: text
is tokenized into words *and* character n-grams, each feature is hashed (via a
stable digest, so embeddings are identical across processes — required for the
JSON-persisted embedded backend) into a fixed-dimension accumulator with a signed
contribution, then L2-normalized.

Why this beats the old Jaccard stub for V1.5 recall:

- It scores in continuous vector space, so near-misses rank above noise instead
  of collapsing to 0.
- Character n-grams catch morphological variants (``retry`` / ``retries`` /
  ``retrying`` share trigrams) and typos — the paraphrase failure mode Jaccard
  misses entirely.

It is intentionally swappable: ``build_embedder`` reads ``CONTEXT_ENGINE_EMBEDDER``.
The default remains the dependency-free hashing embedder; setting
``CONTEXT_ENGINE_EMBEDDER=legacy`` (or ``sentence-transformers``) uses the same
SentenceTransformer family legacy Potpie uses for code-graph semantic search.
``CONTEXT_ENGINE_EMBEDDER=none`` disables embeddings and the store falls back to
a *labeled* lexical match.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from typing import Sequence

from potpie.context_engine.domain.ports.embedder import EmbedderPort

_DEFAULT_DIMENSIONS = 256
_WORD_RE = re.compile(r"[a-z0-9]+")
_CHAR_NGRAM_MIN = 3
_CHAR_NGRAM_MAX = 4


@dataclass(slots=True)
class HashingEmbedder:
    """Deterministic feature-hashing embedder (the bundled local default)."""

    dimensions: int = _DEFAULT_DIMENSIONS
    name: str = "local-hashing-v1"

    def __post_init__(self) -> None:
        # A non-positive dimension would crash at runtime (modulo-by-zero in
        # ``_bucket`` for 0, negative indexing for <0); reject it up front.
        self.dimensions = int(self.dimensions)
        if self.dimensions <= 0:
            raise ValueError("HashingEmbedder.dimensions must be a positive integer")

    def embed(self, text: str) -> tuple[float, ...]:
        vec = [0.0] * self.dimensions
        for token, weight in _features(text):
            idx, sign = _bucket(token, self.dimensions)
            vec[idx] += sign * weight
        return _l2_normalize(vec)

    def embed_many(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [self.embed(t) for t in texts]


@dataclass(slots=True)
class SentenceTransformerEmbedder:
    """Legacy-compatible local embedder backed by ``sentence-transformers``."""

    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    _model: object | None = None

    @property
    def name(self) -> str:
        return f"sentence-transformers/{self.model_name}"

    @property
    def dimensions(self) -> int:
        model = self._get_model()
        dim = getattr(model, "get_sentence_embedding_dimension", lambda: None)()
        return int(dim or 384)

    def embed(self, text: str) -> tuple[float, ...]:
        embedding = self._get_model().encode(text or "")
        return tuple(float(x) for x in embedding.tolist())

    def embed_many(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        embeddings = self._get_model().encode(list(texts))
        return [tuple(float(x) for x in emb.tolist()) for emb in embeddings]

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "CONTEXT_ENGINE_EMBEDDER=legacy requires sentence-transformers. "
                    "Install the legacy Potpie dependencies or add an embeddings extra."
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model


def _features(text: str):
    """Yield (token, weight) features: whole words plus char n-grams.

    Words carry full weight; subword n-grams carry a smaller weight so exact
    word matches dominate but morphological variants still contribute.
    """
    if not text:
        return
    words = _WORD_RE.findall(text.lower())
    for word in words:
        yield (f"w:{word}", 1.0)
        # Bound the n-gram cost on very long tokens.
        padded = f"^{word}$"
        for n in range(_CHAR_NGRAM_MIN, _CHAR_NGRAM_MAX + 1):
            if len(padded) < n:
                continue
            for i in range(len(padded) - n + 1):
                yield (f"c{n}:{padded[i : i + n]}", 0.45)


def _bucket(token: str, dimensions: int) -> tuple[int, float]:
    """Stable bucket index + sign for a feature (process-independent hash)."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    idx = value % dimensions
    sign = 1.0 if (value >> 63) & 1 else -1.0
    return idx, sign


def _l2_normalize(vec: list[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0.0:
        return tuple(vec)
    return tuple(x / norm for x in vec)


def build_embedder() -> EmbedderPort | None:
    """Build the configured embedder (bundled local default, or None to disable).

    ``CONTEXT_ENGINE_EMBEDDER``:
      - unset / ``local`` / ``hashing`` → :class:`HashingEmbedder` (default)
      - ``legacy`` / ``sentence-transformers`` → all-MiniLM-L6-v2 by default
      - ``none`` / ``off`` / ``lexical`` → ``None`` (labeled lexical fallback)

    ``CONTEXT_ENGINE_EMBEDDING_MODEL`` overrides the SentenceTransformer model
    name when the legacy embedder is selected.
    """
    choice = (os.getenv("CONTEXT_ENGINE_EMBEDDER") or "local").strip().lower()
    if choice in ("none", "off", "lexical", "disabled", "0", "false"):
        return None
    if choice in ("", "local", "hashing", "default", "on", "1", "true"):
        return HashingEmbedder()
    if choice in (
        "legacy",
        "sentence-transformers",
        "sentence_transformers",
        "sbert",
        "minilm",
        "all-minilm-l6-v2",
    ):
        return SentenceTransformerEmbedder(
            model_name=(
                os.getenv("CONTEXT_ENGINE_EMBEDDING_MODEL") or "all-MiniLM-L6-v2"
            ).strip()
            or "all-MiniLM-L6-v2"
        )
    # Unknown value: default to the bundled local embedder rather than crashing.
    return HashingEmbedder()


__all__ = ["HashingEmbedder", "SentenceTransformerEmbedder", "build_embedder"]

"""``EmbedderPort`` — the text → vector seam for semantic retrieval (R1).

Retrieval recall needs real vectors, not Jaccard token overlap. This port is the
seam: composition wires a concrete embedder (the bundled, dependency-free local
model by default — no API key) and the claim store embeds the agent-authored
retrieval card on write and the query on read, scoring by cosine similarity.

The port is deliberately tiny so a future API-backed or ONNX model is a wiring
change, not a service rewrite. When no embedder is wired, the store falls back to
a *labeled* lexical match (``match_mode == "lexical"``), never silently.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class EmbedderPort(Protocol):
    """Deterministic text → fixed-dimension unit vector."""

    @property
    def name(self) -> str:
        """Stable embedder identifier (e.g. ``local-hashing-v1``)."""
        ...

    @property
    def dimensions(self) -> int:
        """Vector dimensionality (constant for a given embedder)."""
        ...

    def embed(self, text: str) -> tuple[float, ...]:
        """Embed one text into a unit-norm vector of length :attr:`dimensions`."""
        ...

    def embed_many(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Embed a batch of texts (default impls may just map :meth:`embed`)."""
        ...


__all__ = ["EmbedderPort"]

"""ResourceIndexPort — the retrieval index over stored chunks.

Separate from ``ResourceStorePort`` on purpose: the byte store must stay
swappable to backends (blob storage) that cannot host an index, and indexes
must stay swappable to richer profiles (semantic/hybrid, hosted) without
touching storage. An index is always derivable from the store: feed it
``store.iter_chunks(...)`` and it rebuilds.
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from pydantic import BaseModel

from potpie_context_core.ports.resource_store import Chunk


class IndexCapabilities(BaseModel):
    lexical: bool = False
    semantic: bool = False


class ChunkHit(BaseModel):
    doc_slug: str
    section_slug: str
    seq: int
    score: float | None = None


class ResourceIndexStatus(BaseModel):
    ready: bool
    profile: str
    location: str | None = None
    detail: str | None = None


@runtime_checkable
class ResourceIndexPort(Protocol):
    """Contract every chunk index implements; see the conformance suite."""

    def capabilities(self) -> IndexCapabilities: ...

    def index_document(
        self, *, pot_id: str, doc_slug: str, chunks: Iterable[Chunk]
    ) -> int: ...

    def remove_document(self, *, pot_id: str, doc_slug: str) -> None: ...

    def search(
        self, *, pot_id: str, query: str, limit: int = 20
    ) -> list[ChunkHit]: ...

    def status(self, pot_id: str | None = None) -> ResourceIndexStatus: ...


__all__ = [
    "ChunkHit",
    "IndexCapabilities",
    "ResourceIndexPort",
    "ResourceIndexStatus",
]

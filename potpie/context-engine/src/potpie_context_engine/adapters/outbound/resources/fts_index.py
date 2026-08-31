"""SQLite FTS5 implementation of ResourceIndexPort.

Shares the per-pot ``registry.db`` with the local store's catalog, but owns
the index plane (``chunk_text`` + ``chunks_fts``) exclusively: the store
never writes those tables. A blob-backed deployment swaps this adapter for a
hosted index and feeds it the same ``store.iter_chunks(...)`` stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pathlib import Path

from potpie_context_engine.adapters.outbound.resources.local_chunk_store import (
    pot_resources_root,
)
from potpie_context_engine.adapters.outbound.resources.sqlite_registry import (
    SqliteResourceRegistry,
)
from potpie_context_core.ports.resource_index import (
    ChunkHit,
    IndexCapabilities,
    ResourceIndexStatus,
)
from potpie_context_core.ports.resource_store import Chunk

_PROFILE = "sqlite_fts"


@dataclass(slots=True)
class SqliteFtsResourceIndex:
    """Depends only on a home directory, never on a store implementation —
    the bytes may live anywhere (local disk, blob storage) while this index
    stays a local SQLite file fed through ``index_document``."""

    home: Path

    def registry(self, pot_id: str) -> SqliteResourceRegistry:
        return SqliteResourceRegistry(
            pot_resources_root(self.home, pot_id) / "registry.db"
        )

    def capabilities(self) -> IndexCapabilities:
        return IndexCapabilities(lexical=True, semantic=False)

    def index_document(
        self, *, pot_id: str, doc_slug: str, chunks: Iterable[Chunk]
    ) -> int:
        registry = self.registry(pot_id)
        registry.clear_chunk_text(pot_id, doc_slug)
        count = 0
        for chunk in chunks:
            registry.upsert_chunk_text(
                pot_id=pot_id,
                doc_slug=doc_slug,
                section_slug=chunk.section_slug,
                seq=chunk.seq,
                content=chunk.content,
                ocr_text=chunk.ocr_text,
            )
            count += 1
        registry.rebuild_fts_for_document(pot_id, doc_slug)
        return count

    def remove_document(self, *, pot_id: str, doc_slug: str) -> None:
        self.registry(pot_id).clear_chunk_text(pot_id, doc_slug)

    def search(self, *, pot_id: str, query: str, limit: int = 20) -> list[ChunkHit]:
        rows = self.registry(pot_id).search_chunks(pot_id, query, limit=limit)
        return [
            ChunkHit(
                doc_slug=row["doc_slug"],
                section_slug=row["section_slug"],
                seq=int(row["seq"]),
                score=row.get("score"),
            )
            for row in rows
        ]

    def status(self, pot_id: str | None = None) -> ResourceIndexStatus:
        location = (
            str(pot_resources_root(self.home, pot_id) / "registry.db")
            if pot_id
            else str(self.home / "resources")
        )
        return ResourceIndexStatus(ready=True, profile=_PROFILE, location=location)


__all__ = ["SqliteFtsResourceIndex"]

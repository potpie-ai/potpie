"""SQLite FTS5 implementation of ResourceIndexPort.

Shares the per-pot ``registry.db`` with the local store's catalog, but owns
the index plane (``chunk_text`` + ``chunks_fts``) exclusively: the store
never writes those tables. A blob-backed deployment swaps this adapter for a
hosted index and feeds it the same ``store.iter_chunks(...)`` stream.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable

from pathlib import Path

from potpie_context_engine.adapters.outbound.resources.local_chunk_store import (
    pot_resources_root,
)
from potpie_context_engine.adapters.outbound.resources.sqlite_registry import (
    SqliteResourceRegistry,
)
from potpie_context_engine.core.ports.resource_index import (
    ChunkHit,
    IndexCapabilities,
    ResourceIndexStatus,
    ResourceSearchFilters,
)
from potpie_context_engine.core.ports.resource_store import Chunk
from potpie_context_engine.domain.ports.embedder import EmbedderPort

logger = logging.getLogger(__name__)
_SEMANTIC_MIN_COSINE = 0.25

_PROFILE = "sqlite_fts"


@dataclass(slots=True)
class SqliteFtsResourceIndex:
    """Depends only on a home directory, never on a store implementation —
    the bytes may live anywhere (local disk, blob storage) while this index
    stays a local SQLite file fed through ``index_document``."""

    home: Path
    embedder: EmbedderPort | None = None

    def registry(self, pot_id: str) -> SqliteResourceRegistry:
        return SqliteResourceRegistry(
            pot_resources_root(self.home, pot_id) / "registry.db"
        )

    def capabilities(self) -> IndexCapabilities:
        provider, model = _semantic_metadata(self.embedder)
        return IndexCapabilities(
            profile=_PROFILE,
            semantic_provider=provider,
            semantic_model=model,
            lexical=True,
            semantic=self.embedder is not None,
        )

    def index_document(
        self, *, pot_id: str, doc_slug: str, chunks: Iterable[Chunk]
    ) -> int:
        registry = self.registry(pot_id)
        rows: list[dict[str, object]] = []
        for chunk in chunks:
            rows.append(
                {
                    "section_slug": chunk.section_slug,
                    "seq": chunk.seq,
                    "content": chunk.content,
                    "ocr_text": chunk.ocr_text,
                    "section_title": chunk.section_title or "",
                    "source_ref": chunk.source_ref or "",
                    "source_kind": chunk.source_kind or "",
                    "parser_tier": chunk.parser_tier or "",
                    "page_numbers": chunk.page_numbers,
                }
            )
        embeddings: dict[tuple[str, int], tuple[str, list[float]]] = {}
        if self.embedder is not None and rows:
            texts = [
                "\n".join(
                    value
                    for value in (
                        str(row.get("section_title") or ""),
                        str(row.get("content") or ""),
                        str(row.get("ocr_text") or ""),
                    )
                    if value
                )
                for row in rows
            ]
            vectors = self._embed_many(texts)
            if len(vectors) == len(rows):
                for row, vector in zip(rows, vectors, strict=True):
                    embeddings[(str(row["section_slug"]), int(row["seq"]))] = (
                        self.embedder.name,
                        [float(value) for value in vector],
                    )
        return registry.replace_chunk_text_document(
            pot_id=pot_id,
            doc_slug=doc_slug,
            rows=rows,
            embeddings=embeddings,
        )

    def remove_document(self, *, pot_id: str, doc_slug: str) -> None:
        self.registry(pot_id).clear_chunk_text(pot_id, doc_slug)

    def search(
        self,
        *,
        pot_id: str,
        query: str,
        limit: int = 20,
        filters: ResourceSearchFilters | None = None,
    ) -> list[ChunkHit]:
        registry = self.registry(pot_id)
        rows = registry.search_chunks(
            pot_id, query, limit=limit, filters=filters
        )
        semantic_rows: list[dict[str, object]] = []
        if self.embedder is not None:
            semantic_rows = registry.search_semantic_chunks(
                pot_id,
                self._embed(query),
                limit=limit,
                filters=filters,
            )
        fts_scores = [
            float(row["score"])
            for row in rows
            if isinstance(row.get("score"), (int, float))
        ]
        best_fts = min(fts_scores) if fts_scores else None
        worst_fts = max(fts_scores) if fts_scores else None
        combined: dict[tuple[str, str, int], dict[str, object]] = {}
        for row in rows:
            raw_score = row.get("score")
            lexical = _normalize_fts_score(raw_score, best_fts, worst_fts)
            key = (row["doc_slug"], row["section_slug"], int(row["seq"]))
            combined[key] = {
                "fts_score": raw_score,
                "fts_similarity": lexical,
                "semantic_score": None,
                "semantic_similarity": None,
                "retrieval_source": "fts",
            }
        for row in semantic_rows:
            semantic_score = row.get("semantic_score")
            if (
                isinstance(semantic_score, (int, float))
                and float(semantic_score) < _SEMANTIC_MIN_COSINE
            ):
                continue
            key = (row["doc_slug"], row["section_slug"], int(row["seq"]))
            current = combined.setdefault(
                key,
                {
                    "fts_score": None,
                    "fts_similarity": None,
                    "semantic_score": None,
                    "semantic_similarity": None,
                    "retrieval_source": "semantic",
                },
            )
            current["semantic_score"] = semantic_score
            current["semantic_similarity"] = _normalize_cosine(
                semantic_score
            )
            if current.get("fts_score") is not None:
                current["retrieval_source"] = "hybrid"

        ranked = sorted(
            combined.items(),
            key=lambda pair: (
                -_combined_score(pair[1]),
                -float(pair[1].get("fts_similarity") or 0.0),
                pair[0][0],
                pair[0][1],
                pair[0][2],
            ),
        )
        return [
            ChunkHit(
                doc_slug=key[0],
                section_slug=key[1],
                seq=key[2],
                score=_combined_score(values),
                fts_score=values.get("fts_score"),
                semantic_score=values.get("semantic_score"),
                retrieval_source=str(values.get("retrieval_source") or "fts"),
            )
            for key, values in ranked[:limit]
        ]

    def _embed_many(self, texts: list[str]) -> list[tuple[float, ...]]:
        assert self.embedder is not None
        try:
            return [tuple(vector) for vector in self.embedder.embed_many(texts)]
        except Exception as exc:  # noqa: BLE001 - retrieval must retain lexical service
            logger.warning(
                "semantic chunk indexing unavailable for %s; using local hashing fallback: %s",
                self.embedder.name,
                exc,
            )
            from potpie_context_engine.adapters.outbound.intelligence.local_embedder import (
                HashingEmbedder,
            )

            self.embedder = HashingEmbedder()
            return [tuple(vector) for vector in self.embedder.embed_many(texts)]

    def _embed(self, text: str) -> tuple[float, ...]:
        assert self.embedder is not None
        try:
            return tuple(self.embedder.embed(text))
        except Exception as exc:  # noqa: BLE001 - retrieval must retain lexical service
            logger.warning(
                "semantic chunk query unavailable for %s; using local hashing fallback: %s",
                self.embedder.name,
                exc,
            )
            from potpie_context_engine.adapters.outbound.intelligence.local_embedder import (
                HashingEmbedder,
            )

            self.embedder = HashingEmbedder()
            return tuple(self.embedder.embed(text))

    def status(self, pot_id: str | None = None) -> ResourceIndexStatus:
        location = (
            str(pot_resources_root(self.home, pot_id) / "registry.db")
            if pot_id
            else str(self.home / "resources")
        )
        provider, model = _semantic_metadata(self.embedder)
        if provider is None:
            detail = "semantic=disabled"
        elif model:
            detail = f"semantic={provider} model={model}"
        else:
            detail = f"semantic={provider}"
        return ResourceIndexStatus(
            ready=True, profile=_PROFILE, location=location, detail=detail
        )


__all__ = ["SqliteFtsResourceIndex"]


def _semantic_metadata(
    embedder: EmbedderPort | None,
) -> tuple[str | None, str | None]:
    """Return a stable provider label without exposing implementation classes."""
    if embedder is None:
        return None, None
    name = str(getattr(embedder, "name", "") or "").strip()
    if name.startswith("sentence-transformers/"):
        return "sentence-transformers", name.split("/", 1)[1] or None
    if name.startswith("local-hashing"):
        return "hashing", None
    return name or "custom", None


def _normalize_fts_score(
    score: object, best: float | None, worst: float | None
) -> float | None:
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return None
    if best is None or worst is None:
        return None
    if best == worst:
        return 1.0
    return max(0.0, min(1.0, (worst - float(score)) / (worst - best)))


def _normalize_cosine(score: object) -> float | None:
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return None
    return max(0.0, min(1.0, (float(score) + 1.0) / 2.0))


def _combined_score(values: dict[str, object]) -> float:
    lexical = values.get("fts_similarity")
    semantic = values.get("semantic_similarity")
    if isinstance(lexical, (int, float)) and isinstance(semantic, (int, float)):
        # Exact lexical evidence is especially important for identifiers,
        # table rows, percentages, and model-size values. Semantic similarity
        # remains useful for paraphrases, but must not displace a direct match.
        return 0.70 * float(lexical) + 0.30 * float(semantic)
    if isinstance(semantic, (int, float)):
        return float(semantic)
    if isinstance(lexical, (int, float)):
        return float(lexical)
    return 0.0

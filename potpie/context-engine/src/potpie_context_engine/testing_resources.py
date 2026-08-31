"""In-memory ResourceStorePort / ResourceIndexPort implementations.

They exist to keep the conformance suite honest: a contract only two
filesystem-backed classes satisfy is a coincidence, not a port. They also
sketch what a blob-backed store must provide — no local paths survive past
``import_dir``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from potpie_context_core.ports.resource_index import (
    ChunkHit,
    IndexCapabilities,
    ResourceIndexStatus,
)
from potpie_context_core.ports.resource_store import (
    RESOURCE_CHUNK_TOO_LARGE,
    RESOURCE_MANIFEST_INVALID,
    RESOURCE_NOT_FOUND,
    Chunk,
    ResourceStoreError,
    ResourceStoreStatus,
)
from potpie_context_core.resource_models import (
    CHUNK_HARD_CAP,
    DocumentElementRecord,
    ResourceImportReport,
    ResourceManifest,
    chunk_uri,
)


@dataclass(slots=True)
class InMemoryResourceStore:
    """Dict-backed ResourceStorePort implementation."""

    _manifests: dict[tuple[str, str], ResourceManifest] = field(default_factory=dict)
    _chunks: dict[tuple[str, str, str, int], tuple[str, str]] = field(default_factory=dict)
    _file_hashes: dict[str, set[str]] = field(default_factory=dict)

    def read_manifest(self, source_dir: Path) -> ResourceManifest:
        meta_path = Path(source_dir) / "meta.json"
        if not meta_path.is_file():
            raise ResourceStoreError(
                RESOURCE_MANIFEST_INVALID, "staging directory missing meta.json"
            )
        return ResourceManifest.model_validate(
            json.loads(meta_path.read_text(encoding="utf-8"))
        )

    def read_elements(self, *, pot_id: str, doc_slug: str) -> list[DocumentElementRecord]:
        return []

    def import_dir(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        source_dir: Path,
        manifest: ResourceManifest | None = None,
        force: bool = False,
    ) -> ResourceImportReport:
        stage = Path(source_dir)
        meta = manifest or self.read_manifest(stage)
        errors: list[dict[str, Any]] = []
        staged: dict[tuple[str, int], tuple[str, str]] = {}
        for section in meta.sections:
            section_dir = stage / section.slug
            if not section_dir.is_dir():
                errors.append(
                    {
                        "code": RESOURCE_MANIFEST_INVALID,
                        "message": f"missing section directory: {section.slug}",
                    }
                )
                continue
            for chunk_ref in section.chunks:
                chunk_path = section_dir / f"{chunk_ref.seq:04d}.txt"
                if not chunk_path.is_file():
                    errors.append(
                        {
                            "code": RESOURCE_MANIFEST_INVALID,
                            "message": f"missing chunk file: {chunk_path.name}",
                        }
                    )
                    continue
                text = chunk_path.read_text(encoding="utf-8")
                if len(text) > CHUNK_HARD_CAP:
                    errors.append(
                        {
                            "code": RESOURCE_CHUNK_TOO_LARGE,
                            "message": (
                                f"chunk exceeds {CHUNK_HARD_CAP} chars: "
                                f"{section.slug}/{chunk_ref.seq}"
                            ),
                        }
                    )
                    continue
                ocr_path = section_dir / f"{chunk_ref.seq:04d}.ocr.txt"
                ocr = ocr_path.read_text(encoding="utf-8").strip() if ocr_path.is_file() else ""
                staged[(section.slug, chunk_ref.seq)] = (text, ocr)
        if errors:
            return ResourceImportReport(
                pot_id=pot_id,
                doc_slug=doc_slug,
                graph_written=False,
                errors=errors,
                recommended_next_action="fix staging directory and re-import",
            )

        prior = self._manifests.get((pot_id, doc_slug))
        prior_slugs = {s.slug for s in prior.sections} if prior else set()
        new_slugs = {s.slug for s in meta.sections}
        prior_hashes = (
            {s.slug: s.content_hash for s in prior.sections} if prior else {}
        )

        for key in [k for k in self._chunks if k[0] == pot_id and k[1] == doc_slug]:
            del self._chunks[key]
        for (section_slug, seq), (text, ocr) in staged.items():
            self._chunks[(pot_id, doc_slug, section_slug, seq)] = (text, ocr)
        self._manifests[(pot_id, doc_slug)] = meta

        added = sorted(new_slugs - prior_slugs)
        removed = sorted(prior_slugs - new_slugs)
        common = new_slugs & prior_slugs
        changed = sorted(
            s.slug
            for s in meta.sections
            if s.slug in common and s.content_hash != prior_hashes.get(s.slug)
        )
        kept = sorted(common - set(changed))
        return ResourceImportReport(
            pot_id=pot_id,
            doc_slug=doc_slug,
            sections_added=added,
            sections_kept=kept,
            sections_changed=changed,
            sections_removed=removed,
            summary_pending=[s.slug for s in meta.sections if not s.summary.strip()],
            graph_written=False,
        )

    def get(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        section_slug: str,
        seq: int,
        with_neighbors: bool = False,
    ) -> Chunk:
        entry = self._chunks.get((pot_id, doc_slug, section_slug, seq))
        if entry is None:
            raise ResourceStoreError(
                RESOURCE_NOT_FOUND,
                f"chunk not found: {chunk_uri(doc_slug, section_slug, seq)}",
            )
        text, ocr = entry
        neighbors: list[dict[str, Any]] | None = None
        if with_neighbors:
            neighbors = []
            for neighbor_seq in (seq - 1, seq + 1):
                sibling = self._chunks.get((pot_id, doc_slug, section_slug, neighbor_seq))
                if sibling is not None:
                    neighbors.append(
                        {
                            "uri": chunk_uri(doc_slug, section_slug, neighbor_seq),
                            "seq": neighbor_seq,
                            "content": sibling[0],
                        }
                    )
        return Chunk(
            uri=chunk_uri(doc_slug, section_slug, seq),
            pot_id=pot_id,
            doc_slug=doc_slug,
            section_slug=section_slug,
            seq=seq,
            content=text,
            ocr_text=ocr,
            provenance=None,
            neighbors=neighbors,
        )

    def iter_chunks(self, *, pot_id: str, doc_slug: str) -> list[Chunk]:
        return [
            self.get(pot_id=pot_id, doc_slug=doc_slug, section_slug=section, seq=seq)
            for (pid, doc, section, seq) in sorted(self._chunks)
            if pid == pot_id and doc == doc_slug
        ]

    def list_documents(self, *, pot_id: str) -> list[dict[str, Any]]:
        return [
            {"doc_slug": doc, "section_count": len(manifest.sections)}
            for (pid, doc), manifest in sorted(self._manifests.items())
            if pid == pot_id
        ]

    def delete(self, *, pot_id: str, doc_slug: str) -> dict[str, Any]:
        self._manifests.pop((pot_id, doc_slug), None)
        for key in [k for k in self._chunks if k[0] == pot_id and k[1] == doc_slug]:
            del self._chunks[key]
        return {"pot_id": pot_id, "doc_slug": doc_slug, "removed": True}

    def remove_document(self, *, pot_id: str, doc_slug: str) -> dict[str, Any]:
        return self.delete(pot_id=pot_id, doc_slug=doc_slug)

    def purge_pot(self, pot_id: str) -> bool:
        for key in [k for k in self._manifests if k[0] == pot_id]:
            del self._manifests[key]
        for key in [k for k in self._chunks if k[0] == pot_id]:
            del self._chunks[key]
        self._file_hashes.pop(pot_id, None)
        return True

    def status(self, pot_id: str | None = None) -> ResourceStoreStatus:
        return ResourceStoreStatus(ready=True, backend="memory")

    def is_file_imported(self, *, pot_id: str, content_hash: str) -> bool:
        return content_hash in self._file_hashes.get(pot_id, set())

    def record_file_hash(self, *, pot_id: str, content_hash: str, doc_slug: str) -> None:
        self._file_hashes.setdefault(pot_id, set()).add(content_hash)


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(slots=True)
class InMemoryResourceIndex:
    """Token-overlap ResourceIndexPort implementation."""

    _docs: dict[tuple[str, str], list[tuple[str, int, set[str]]]] = field(
        default_factory=dict
    )

    def capabilities(self) -> IndexCapabilities:
        return IndexCapabilities(lexical=True, semantic=False)

    def index_document(
        self, *, pot_id: str, doc_slug: str, chunks: Iterable[Chunk]
    ) -> int:
        rows: list[tuple[str, int, set[str]]] = []
        for chunk in chunks:
            tokens = {t.lower() for t in _TOKEN_RE.findall(f"{chunk.content} {chunk.ocr_text}")}
            rows.append((chunk.section_slug, chunk.seq, tokens))
        self._docs[(pot_id, doc_slug)] = rows
        return len(rows)

    def remove_document(self, *, pot_id: str, doc_slug: str) -> None:
        self._docs.pop((pot_id, doc_slug), None)

    def search(self, *, pot_id: str, query: str, limit: int = 20) -> list[ChunkHit]:
        terms = {t.lower() for t in _TOKEN_RE.findall(query)}
        if not terms:
            return []
        scored: list[tuple[float, ChunkHit]] = []
        for (pid, doc_slug), rows in self._docs.items():
            if pid != pot_id:
                continue
            for section_slug, seq, tokens in rows:
                overlap = len(terms & tokens)
                if overlap:
                    scored.append(
                        (
                            -overlap,
                            ChunkHit(
                                doc_slug=doc_slug,
                                section_slug=section_slug,
                                seq=seq,
                                score=float(overlap),
                            ),
                        )
                    )
        scored.sort(key=lambda pair: pair[0])
        return [hit for _, hit in scored[:limit]]

    def status(self, pot_id: str | None = None) -> ResourceIndexStatus:
        return ResourceIndexStatus(ready=True, profile="memory")


__all__ = ["InMemoryResourceIndex", "InMemoryResourceStore"]

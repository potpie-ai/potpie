"""ResourceStorePort — document payload storage behind one swappable port.

The store owns bytes and the per-pot document catalog: chunk text, OCR
sidecars, provenance, manifests, dedup hashes. Local disk implements it
today; blob storage (S3, GCS) implements the same contract later. Searching
is deliberately NOT here — that is ``ResourceIndexPort``, so a byte store
never has to host an index.

Errors carry stable codes (``resource_not_found``,
``resource_manifest_invalid``, ``resource_chunk_too_large``) because callers
retry a bad slug and an oversized chunk differently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

from pydantic import BaseModel

from potpie_context_core.resource_models import (
    DocumentElementRecord,
    ResourceImportReport,
    ResourceManifest,
)

RESOURCE_NOT_FOUND = "resource_not_found"
RESOURCE_MANIFEST_INVALID = "resource_manifest_invalid"
RESOURCE_CHUNK_TOO_LARGE = "resource_chunk_too_large"


class ResourceStoreError(ValueError):
    """Store failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.code, self.message, self.detail))


class Chunk(BaseModel):
    """One stored chunk, addressable by ``potpie://res/<doc>/<section>/<seq>``."""

    uri: str
    pot_id: str
    doc_slug: str
    section_slug: str
    seq: int
    content: str
    ocr_text: str = ""
    provenance: list[dict[str, Any]] | None = None
    neighbors: list[dict[str, Any]] | None = None
    score: float | None = None

    def to_payload(self) -> dict[str, Any]:
        """The CLI/RPC wire shape (kept byte-compatible with the pre-port dict)."""
        payload: dict[str, Any] = {
            "uri": self.uri,
            "pot_id": self.pot_id,
            "doc_slug": self.doc_slug,
            "section_slug": self.section_slug,
            "seq": self.seq,
            "content": self.content,
        }
        if self.ocr_text:
            payload["ocr_text"] = self.ocr_text
        payload["provenance"] = self.provenance
        if self.neighbors is not None:
            payload["neighbors"] = self.neighbors
        if self.score is not None:
            payload["score"] = self.score
        return payload


class ResourceStoreStatus(BaseModel):
    ready: bool
    backend: str
    location: str | None = None
    detail: str | None = None


@runtime_checkable
class ResourceStorePort(Protocol):
    """Contract every chunk store implements; see the conformance suite."""

    def import_dir(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        source_dir: Path,
        manifest: ResourceManifest | None = None,
        force: bool = False,
    ) -> ResourceImportReport: ...

    def read_manifest(self, source_dir: Path) -> ResourceManifest: ...

    def read_elements(
        self, *, pot_id: str, doc_slug: str
    ) -> list[DocumentElementRecord]: ...

    def get(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        section_slug: str,
        seq: int,
        with_neighbors: bool = False,
    ) -> Chunk: ...

    def iter_chunks(self, *, pot_id: str, doc_slug: str) -> Iterable[Chunk]: ...

    def list_documents(self, *, pot_id: str) -> list[dict[str, Any]]: ...

    def delete(self, *, pot_id: str, doc_slug: str) -> dict[str, Any]: ...

    def purge_pot(self, pot_id: str) -> bool: ...

    def status(self, pot_id: str | None = None) -> ResourceStoreStatus: ...

    def is_file_imported(self, *, pot_id: str, content_hash: str) -> bool: ...

    def record_file_hash(
        self, *, pot_id: str, content_hash: str, doc_slug: str
    ) -> None: ...


__all__ = [
    "RESOURCE_CHUNK_TOO_LARGE",
    "RESOURCE_MANIFEST_INVALID",
    "RESOURCE_NOT_FOUND",
    "Chunk",
    "ResourceStoreError",
    "ResourceStorePort",
    "ResourceStoreStatus",
]

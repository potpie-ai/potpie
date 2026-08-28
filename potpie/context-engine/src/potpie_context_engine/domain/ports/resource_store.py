"""Port for the local resource (document chunk) payload plane."""

from __future__ import annotations

from typing import Any, Protocol

from potpie_context_engine.domain.resource_models import ResourceImportReport, ResourceManifest


class ResourceStorePort(Protocol):
    def import_manifest(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        staging_dir: str,
        manifest: ResourceManifest,
        force: bool = False,
    ) -> ResourceImportReport: ...

    def get_chunk_text(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        section_slug: str,
        seq: int,
        with_neighbors: bool = False,
    ) -> dict[str, Any]: ...

    def list_documents(self, *, pot_id: str) -> list[dict[str, Any]]: ...

    def remove_document(self, *, pot_id: str, doc_slug: str) -> dict[str, Any]: ...

    def search_chunks(
        self,
        *,
        pot_id: str,
        query: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]: ...

    def is_file_imported(self, *, pot_id: str, content_hash: str) -> bool: ...

    def record_file_hash(
        self,
        *,
        pot_id: str,
        content_hash: str,
        doc_slug: str,
    ) -> None: ...


__all__ = ["ResourceStorePort"]

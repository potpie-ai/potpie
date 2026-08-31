"""Resource contracts moved to potpie-context-core; this shim keeps imports stable.

The models are the wire contract between stores, indexes, the daemon RPC
surface, and the CLI, so they live in ``potpie_context_core.resource_models``
next to the ports that speak them. Import from there in new code.
"""

from __future__ import annotations

from potpie_context_core.resource_models import (
    CHUNK_HARD_CAP,
    CHUNK_TARGET_DEFAULT,
    LABEL_MAX,
    SECTION_CHUNK_MAX,
    SECTION_CHUNK_MIN,
    SLUG_RE,
    SUMMARY_MAX,
    TITLE_MAX,
    CanonicalChunk,
    ChunkArtifacts,
    ChunkMetadata,
    ChunkPointers,
    ChunkProvenanceRecord,
    ChunkProvenanceSidecar,
    ChunkRef,
    DocumentElementRecord,
    ResourceImportReport,
    ResourceManifest,
    SectionManifest,
    chunk_uri,
    parse_chunk_uri,
    text_sha256,
    validate_doc_slug,
)

__all__ = [
    "CHUNK_HARD_CAP",
    "CHUNK_TARGET_DEFAULT",
    "LABEL_MAX",
    "SECTION_CHUNK_MAX",
    "SECTION_CHUNK_MIN",
    "SLUG_RE",
    "SUMMARY_MAX",
    "TITLE_MAX",
    "CanonicalChunk",
    "ChunkArtifacts",
    "ChunkMetadata",
    "ChunkPointers",
    "ChunkProvenanceRecord",
    "ChunkProvenanceSidecar",
    "ChunkRef",
    "DocumentElementRecord",
    "ResourceImportReport",
    "ResourceManifest",
    "SectionManifest",
    "chunk_uri",
    "parse_chunk_uri",
    "text_sha256",
    "validate_doc_slug",
]

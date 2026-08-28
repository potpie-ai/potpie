"""Pydantic models for the resource (document chunk) payload plane."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

CHUNK_HARD_CAP = 8000
CHUNK_TARGET_DEFAULT = 4000
SECTION_CHUNK_MIN = 1
SECTION_CHUNK_MAX = 5
SUMMARY_MAX = 2000
TITLE_MAX = 200
LABEL_MAX = 200
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class ChunkRef(BaseModel):
    seq: int = Field(ge=0)
    label: str = Field(min_length=1, max_length=LABEL_MAX)
    offset: int | None = None
    page: int | None = None


class SectionManifest(BaseModel):
    slug: str
    title: str = Field(max_length=TITLE_MAX)
    summary: str = Field(default="", max_length=SUMMARY_MAX)
    ordinal: int = Field(default=0, ge=0)
    content_hash: str = ""
    chunks: list[ChunkRef] = Field(min_length=1, max_length=SECTION_CHUNK_MAX)

    @field_validator("slug")
    @classmethod
    def _slug(cls, value: str) -> str:
        slug = value.strip()
        if not SLUG_RE.match(slug):
            raise ValueError(f"invalid section slug: {slug!r}")
        return slug

    @field_validator("summary")
    @classmethod
    def _summary(cls, value: str) -> str:
        return value.strip()


class DocumentElementRecord(BaseModel):
    element_id: str = Field(min_length=1, max_length=120)
    element_type: str = Field(min_length=1, max_length=80)
    text: str = ""
    artifact_ref: str | None = None
    page_number: int | None = None
    bbox: list[float] | None = None
    char_start: int | None = None
    char_end: int | None = None
    text_hash: str = ""

    @field_validator("bbox")
    @classmethod
    def _bbox(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError("bbox must have four coordinates [l, t, r, b]")
        return value


class ChunkProvenanceRecord(BaseModel):
    element_id: str = Field(min_length=1, max_length=120)
    page_number: int | None = None
    bbox: list[float] | None = None
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)

    @field_validator("bbox")
    @classmethod
    def _bbox(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError("bbox must have four coordinates [l, t, r, b]")
        return value


class ChunkProvenanceSidecar(BaseModel):
    provenance: list[ChunkProvenanceRecord] = Field(default_factory=list)


class ResourceManifest(BaseModel):
    source_ref: str = ""
    source_kind: Literal[
        "markdown", "text", "pdf", "html", "image", "docx", "pptx", "xlsx"
    ] = "text"
    source_content_hash: str = ""
    mime_type: str = ""
    provenance_version: int = 0
    parser_tier: str = ""
    sections: list[SectionManifest] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_section_slugs(self) -> ResourceManifest:
        slugs = [s.slug for s in self.sections]
        if len(slugs) != len(set(slugs)):
            raise ValueError("duplicate section slugs in manifest")
        return self


class ChunkArtifacts(BaseModel):
    image_path: str | None = None
    ocr_text: str | None = None
    table_csv_path: str | None = None


class ChunkPointers(BaseModel):
    next_chunk_id: str | None = None
    parent_section_id: str | None = None
    references_artifacts: list[str] = Field(default_factory=list)


class ChunkMetadata(BaseModel):
    page_number: int | None = None
    bbox: list[float] | None = None
    source_path: str | None = None


class CanonicalChunk(BaseModel):
    chunk_id: str
    doc_id: str
    content_hash: str
    chunk_type: Literal["text", "table", "image_block"] = "text"
    content: str = Field(max_length=CHUNK_HARD_CAP)
    artifacts: ChunkArtifacts = Field(default_factory=ChunkArtifacts)
    pointers: ChunkPointers = Field(default_factory=ChunkPointers)
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)


class ResourceImportReport(BaseModel):
    pot_id: str
    doc_slug: str
    sections_added: list[str] = Field(default_factory=list)
    sections_kept: list[str] = Field(default_factory=list)
    sections_changed: list[str] = Field(default_factory=list)
    sections_removed: list[str] = Field(default_factory=list)
    elements_added: list[str] = Field(default_factory=list)
    elements_removed: list[str] = Field(default_factory=list)
    provenance_version: int = 0
    parser_tier: str | None = None
    summary_pending: list[str] = Field(default_factory=list)
    graph_written: bool = False
    missing_claim_keys: list[str] = Field(default_factory=list)
    recommended_next_action: str | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)


def validate_doc_slug(slug: str) -> str:
    value = slug.strip()
    if not SLUG_RE.match(value):
        raise ValueError(f"invalid document slug: {value!r}")
    return value


def text_sha256(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def chunk_uri(doc_slug: str, section_slug: str, seq: int) -> str:
    return f"potpie://res/{doc_slug}/{section_slug}/{seq:04d}"


def parse_chunk_uri(uri: str) -> tuple[str, str, int]:
    raw = uri.strip()
    prefix = "potpie://res/"
    if not raw.startswith(prefix):
        raise ValueError(f"invalid chunk uri: {uri!r}")
    rest = raw[len(prefix):]
    parts = rest.split("/")
    if len(parts) == 3:
        doc_slug, section_slug, seq_raw = parts
    elif len(parts) == 2:
        doc_slug, section_slug = parts
        seq_raw = "0000"
    else:
        raise ValueError(f"invalid chunk uri: {uri!r}")
    slug_re = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    if not slug_re.match(section_slug):
        raise ValueError(f"invalid section slug in uri: {uri!r}")
    return validate_doc_slug(doc_slug), section_slug, int(seq_raw)


__all__ = [
    "CanonicalChunk",
    "CHUNK_HARD_CAP",
    "CHUNK_TARGET_DEFAULT",
    "ChunkProvenanceRecord",
    "ChunkProvenanceSidecar",
    "chunk_uri",
    "DocumentElementRecord",
    "parse_chunk_uri",
    "ResourceImportReport",
    "ResourceManifest",
    "SectionManifest",
    "validate_doc_slug",
    "text_sha256",
]

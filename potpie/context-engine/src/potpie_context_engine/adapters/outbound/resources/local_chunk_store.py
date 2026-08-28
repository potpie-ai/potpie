"""Local filesystem chunk store under ~/.potpie/resources/."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from potpie_context_engine.adapters.outbound.pots.local_pot_store import default_home
from potpie_context_engine.adapters.outbound.resources.sqlite_registry import (
    SqliteResourceRegistry,
)
from potpie_context_engine.domain.resource_models import (
    CHUNK_HARD_CAP,
    ChunkProvenanceSidecar,
    DocumentElementRecord,
    ResourceImportReport,
    ResourceManifest,
    chunk_uri,
    text_sha256,
    validate_doc_slug,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


@dataclass(slots=True)
class LocalResourceStore:
    home: Path = field(default_factory=default_home)

    def pot_resources_root(self, pot_id: str) -> Path:
        safe = pot_id.replace("/", "_")
        return self.home / "resources" / safe

    def registry(self, pot_id: str) -> SqliteResourceRegistry:
        return SqliteResourceRegistry(self.pot_resources_root(pot_id) / "registry.db")

    def doc_root(self, pot_id: str, doc_slug: str) -> Path:
        return self.pot_resources_root(pot_id) / doc_slug

    def read_manifest(self, staging_dir: Path) -> ResourceManifest:
        meta_path = staging_dir / "meta.json"
        if not meta_path.is_file():
            raise ValueError("staging directory missing meta.json")
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return ResourceManifest.model_validate(data)

    def read_elements(self, *, pot_id: str, doc_slug: str) -> list[DocumentElementRecord]:
        doc_slug = validate_doc_slug(doc_slug)
        elements_path = self.doc_root(pot_id, doc_slug) / "elements.jsonl"
        if not elements_path.is_file():
            return []
        elements: list[DocumentElementRecord] = []
        for line in elements_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            elements.append(DocumentElementRecord.model_validate(json.loads(line)))
        return elements

    def _read_elements_file(self, doc_root: Path) -> list[DocumentElementRecord]:
        elements_path = doc_root / "elements.jsonl"
        if not elements_path.is_file():
            return []
        elements: list[DocumentElementRecord] = []
        for line in elements_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            elements.append(DocumentElementRecord.model_validate(json.loads(line)))
        return elements

    def import_manifest(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        staging_dir: str | Path,
        manifest: ResourceManifest | None = None,
        force: bool = False,
    ) -> ResourceImportReport:
        doc_slug = validate_doc_slug(doc_slug)
        stage = Path(staging_dir).expanduser().resolve()
        if not stage.is_dir():
            raise ValueError(f"staging directory not found: {stage}")

        meta = manifest or self.read_manifest(stage)
        registry = self.registry(pot_id)
        dest = self.doc_root(pot_id, doc_slug)
        backup: Path | None = None

        if dest.exists() and not force:
            # atomic: only proceed if content differs; caller handles dedup at file level
            pass

        if dest.exists():
            backup = dest.parent / f".{doc_slug}.bak-{uuid.uuid4().hex[:8]}"
            shutil.move(dest, backup)

        try:
            shutil.copytree(stage, dest, dirs_exist_ok=False)
            report = self._import_from_stored_tree(
                pot_id=pot_id,
                doc_slug=doc_slug,
                meta=meta,
                registry=registry,
            )
            if backup and backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
            return report
        except Exception:
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            if backup and backup.exists():
                shutil.move(backup, dest)
            raise

    def _import_from_stored_tree(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        meta: ResourceManifest,
        registry: SqliteResourceRegistry,
    ) -> ResourceImportReport:
        errors: list[dict[str, Any]] = []
        summary_pending: list[str] = []
        dest = self.doc_root(pot_id, doc_slug)

        for section in meta.sections:
            section_dir = dest / section.slug
            if not section_dir.is_dir():
                errors.append(
                    {
                        "code": "resource_manifest_invalid",
                        "message": f"missing section directory: {section.slug}",
                    }
                )
                continue
            for chunk_ref in section.chunks:
                chunk_path = section_dir / f"{chunk_ref.seq:04d}.txt"
                if not chunk_path.is_file():
                    errors.append(
                        {
                            "code": "resource_manifest_invalid",
                            "message": f"missing chunk file: {chunk_path.name}",
                        }
                    )
                    continue
                text = chunk_path.read_text(encoding="utf-8")
                if len(text) > CHUNK_HARD_CAP:
                    errors.append(
                        {
                            "code": "resource_chunk_too_large",
                            "message": f"chunk exceeds {CHUNK_HARD_CAP} chars: {section.slug}/{chunk_ref.seq}",
                        }
                    )

        if errors:
            return ResourceImportReport(
                pot_id=pot_id,
                doc_slug=doc_slug,
                graph_written=False,
                errors=errors,
                recommended_next_action="fix staging directory and re-import",
            )

        registry.upsert_document(
            pot_id=pot_id,
            doc_slug=doc_slug,
            source_ref=meta.source_ref,
            source_kind=meta.source_kind,
        )

        section_dicts = []
        for section in meta.sections:
            if not section.summary.strip():
                summary_pending.append(section.slug)
            section_dicts.append(
                {
                    "slug": section.slug,
                    "title": section.title,
                    "summary": section.summary,
                    "content_hash": section.content_hash,
                    "ordinal": section.ordinal,
                }
            )

        added, kept, changed, removed = registry.replace_sections(
            pot_id=pot_id,
            doc_slug=doc_slug,
            sections=section_dicts,
        )

        element_dicts = [
            element.model_dump()
            for element in self._read_elements_file(dest)
        ]
        elements_added, elements_removed = registry.replace_elements(
            pot_id=pot_id,
            doc_slug=doc_slug,
            elements=element_dicts,
        )

        for section in meta.sections:
            section_dir = dest / section.slug
            chunk_rows: list[dict[str, Any]] = []
            for chunk_ref in section.chunks:
                chunk_path = section_dir / f"{chunk_ref.seq:04d}.txt"
                content = chunk_path.read_text(encoding="utf-8")
                ocr_path = section_dir / f"{chunk_ref.seq:04d}.ocr.txt"
                ocr_text = (
                    ocr_path.read_text(encoding="utf-8").strip()
                    if ocr_path.is_file()
                    else ""
                )
                chunk_id = f"chk_{uuid.uuid4().hex[:12]}"
                registry.upsert_chunk_text(
                    pot_id=pot_id,
                    doc_slug=doc_slug,
                    section_slug=section.slug,
                    seq=chunk_ref.seq,
                    content=content,
                    ocr_text=ocr_text,
                )
                chunk_rows.append(
                    {
                        "seq": chunk_ref.seq,
                        "label": chunk_ref.label,
                        "chunk_id": chunk_id,
                        "content_hash": text_sha256(content),
                    }
                )
            registry.replace_section_chunks(
                pot_id=pot_id,
                doc_slug=doc_slug,
                section_slug=section.slug,
                chunks=chunk_rows,
            )
            for chunk_ref in section.chunks:
                prov_path = section_dir / f"{chunk_ref.seq:04d}.prov.json"
                if prov_path.is_file():
                    sidecar = ChunkProvenanceSidecar.model_validate(
                        json.loads(prov_path.read_text(encoding="utf-8"))
                    )
                    registry.replace_chunk_provenance(
                        pot_id=pot_id,
                        doc_slug=doc_slug,
                        section_slug=section.slug,
                        seq=chunk_ref.seq,
                        rows=[row.model_dump() for row in sidecar.provenance],
                    )

        registry.rebuild_fts_for_document(pot_id, doc_slug)

        return ResourceImportReport(
            pot_id=pot_id,
            doc_slug=doc_slug,
            sections_added=added,
            sections_kept=kept,
            sections_changed=changed,
            sections_removed=removed,
            elements_added=elements_added,
            elements_removed=elements_removed,
            provenance_version=meta.provenance_version,
            parser_tier=meta.parser_tier or None,
            summary_pending=summary_pending,
            graph_written=False,
            recommended_next_action=(
                "write section summaries and re-import"
                if summary_pending
                else "run graph bridge commit"
            ),
        )

    def get_chunk_text(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        section_slug: str,
        seq: int,
        with_neighbors: bool = False,
    ) -> dict[str, Any]:
        doc_slug = validate_doc_slug(doc_slug)
        chunk_path = self.doc_root(pot_id, doc_slug) / section_slug / f"{seq:04d}.txt"
        if not chunk_path.is_file():
            raise ValueError(f"chunk not found: {chunk_uri(doc_slug, section_slug, seq)}")
        text = chunk_path.read_text(encoding="utf-8")
        ocr_path = chunk_path.parent / f"{seq:04d}.ocr.txt"
        ocr_text = ocr_path.read_text(encoding="utf-8").strip() if ocr_path.is_file() else ""
        result: dict[str, Any] = {
            "uri": chunk_uri(doc_slug, section_slug, seq),
            "pot_id": pot_id,
            "doc_slug": doc_slug,
            "section_slug": section_slug,
            "seq": seq,
            "content": text,
        }
        if ocr_text:
            result["ocr_text"] = ocr_text
        provenance = self.registry(pot_id).get_chunk_provenance(
            pot_id=pot_id,
            doc_slug=doc_slug,
            section_slug=section_slug,
            seq=seq,
        )
        if provenance:
            result["provenance"] = provenance
        else:
            prov_path = chunk_path.parent / f"{seq:04d}.prov.json"
            if prov_path.is_file():
                sidecar = ChunkProvenanceSidecar.model_validate(
                    json.loads(prov_path.read_text(encoding="utf-8"))
                )
                result["provenance"] = [row.model_dump() for row in sidecar.provenance]
            else:
                result["provenance"] = None
        if with_neighbors:
            neighbors: list[dict[str, Any]] = []
            for neighbor_seq in (seq - 1, seq + 1):
                if neighbor_seq < 0:
                    continue
                neighbor_path = (
                    self.doc_root(pot_id, doc_slug) / section_slug / f"{neighbor_seq:04d}.txt"
                )
                if neighbor_path.is_file():
                    neighbors.append(
                        {
                            "uri": chunk_uri(doc_slug, section_slug, neighbor_seq),
                            "seq": neighbor_seq,
                            "content": neighbor_path.read_text(encoding="utf-8"),
                        }
                    )
            result["neighbors"] = neighbors
        return result

    def list_documents(self, *, pot_id: str) -> list[dict[str, Any]]:
        return self.registry(pot_id).list_documents(pot_id)

    def remove_document(self, *, pot_id: str, doc_slug: str) -> dict[str, Any]:
        doc_slug = validate_doc_slug(doc_slug)
        root = self.doc_root(pot_id, doc_slug)
        self.registry(pot_id).remove_document(pot_id, doc_slug)
        if root.exists():
            shutil.rmtree(root)
        return {"pot_id": pot_id, "doc_slug": doc_slug, "removed": True}

    def search_chunks(
        self,
        *,
        pot_id: str,
        query: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        hits = self.registry(pot_id).search_chunks(pot_id, query, limit=limit)
        enriched: list[dict[str, Any]] = []
        for hit in hits:
            seq = int(hit["seq"])
            section_slug = hit["section_slug"]
            doc_slug = hit["doc_slug"]
            try:
                chunk = self.get_chunk_text(
                    pot_id=pot_id,
                    doc_slug=doc_slug,
                    section_slug=section_slug,
                    seq=seq,
                )
            except ValueError:
                continue
            chunk["score"] = hit.get("score")
            enriched.append(chunk)
        return enriched

    def is_file_imported(self, *, pot_id: str, content_hash: str) -> bool:
        return self.registry(pot_id).is_file_imported(pot_id, content_hash)

    def record_file_hash(
        self,
        *,
        pot_id: str,
        content_hash: str,
        doc_slug: str,
    ) -> None:
        self.registry(pot_id).record_file_hash(pot_id, content_hash, doc_slug)


__all__ = ["LocalResourceStore", "file_sha256", "text_sha256"]

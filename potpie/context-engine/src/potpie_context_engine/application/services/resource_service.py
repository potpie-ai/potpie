"""Resource ingestion orchestration (parse, import, get, list, remove).

The service depends on two ports, never on concrete storage:
``ResourceStorePort`` owns bytes + catalog (local disk today, blob storage
later), ``ResourceIndexPort`` owns retrieval (SQLite FTS today, hosted or
semantic profiles later). Swapping either is wiring, not surgery, and the
CLI/RPC payload shapes produced here do not change with the backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from typing import Any

from potpie_context_engine.adapters.outbound.resources.graph_bridge import (
    build_retract_mutations,
    parse_import_request,
)
from potpie_context_engine.adapters.outbound.resources.local_chunk_store import (
    LocalResourceStore,
    file_sha256,
)
from potpie_context_engine.adapters.outbound.resources.parsers.dispatch import (
    ParseOptions,
    parse_file_to_staging,
)
from potpie_context_engine.domain.resource_models import (
    ResourceImportReport,
    ResourceManifest,
    parse_chunk_uri,
    validate_doc_slug,
)
from potpie_context_engine.core.ports.graph_service import GraphService
from potpie_context_engine.core.ports.resource_index import ResourceIndexPort
from potpie_context_engine.core.ports.resource_store import (
    format_resource_id,
    parse_resource_id,
    ResourceStoreError,
    ResourceStorePort,
)
from potpie_context_engine.application.services.resource_facade import ResourceFacade


@dataclass(slots=True)
class ResourceService:
    store: ResourceStorePort = field(default_factory=LocalResourceStore)
    index: ResourceIndexPort | None = None
    graph: GraphService | None = None
    facade: ResourceFacade | None = None

    def __post_init__(self) -> None:
        if self.index is None and self.facade is None:
            self.index = self.store.default_index()

    def parse_to_staging(
        self,
        *,
        source_path: str | Path,
        out_dir: str | Path,
        chunk_size: int = 4000,
        force: bool = False,
        pot_id: str | None = None,
        vision_provider: str = "none",
        allow_degraded: bool = False,
    ) -> ResourceManifest:
        source = Path(source_path).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"source file not found: {source}")
        content_hash = file_sha256(source)
        if (
            pot_id
            and not force
            and self.store.is_file_imported(pot_id=pot_id, content_hash=content_hash)
        ):
            raise ValueError(
                f"file already imported (content_hash={content_hash}); use --force to re-parse"
            )
        out = Path(out_dir).expanduser().resolve()
        return parse_file_to_staging(
            source,
            out,
            options=ParseOptions(
                chunk_target=chunk_size,
                vision_provider=vision_provider,
                allow_degraded=allow_degraded,
            ),
        )

    def parse_and_import(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        source_path: str | Path,
        staging_dir: str | Path | None = None,
        chunk_size: int = 4000,
        force: bool = False,
        source_ref: str | None = None,
        vision_provider: str = "none",
        allow_degraded: bool = False,
    ) -> ResourceImportReport:
        doc_slug = validate_doc_slug(doc_slug)
        source = Path(source_path).expanduser().resolve()
        if self.facade is not None and staging_dir is None:
            with tempfile.TemporaryDirectory(prefix="potpie-document-") as scratch:
                return self._parse_and_import_facade(
                    pot_id=pot_id,
                    doc_slug=doc_slug,
                    source=source,
                    stage=Path(scratch),
                    chunk_size=chunk_size,
                    source_ref=source_ref,
                    vision_provider=vision_provider,
                    allow_degraded=allow_degraded,
                )
        if self.facade is not None:
            stage = Path(staging_dir).expanduser().resolve()
            return self._parse_and_import_facade(
                pot_id=pot_id,
                doc_slug=doc_slug,
                source=source,
                stage=stage,
                chunk_size=chunk_size,
                source_ref=source_ref,
                vision_provider=vision_provider,
                allow_degraded=allow_degraded,
            )
        if staging_dir:
            stage = Path(staging_dir).expanduser().resolve()
        else:
            stage = self.store.staging_root(pot_id=pot_id) / doc_slug
        if stage.exists():
            import shutil

            shutil.rmtree(stage)
        self.parse_to_staging(
            source_path=source,
            out_dir=stage,
            chunk_size=chunk_size,
            force=force,
            pot_id=pot_id,
            vision_provider=vision_provider,
            allow_degraded=allow_degraded,
        )
        content_hash = file_sha256(source)
        self.store.record_file_hash(
            pot_id=pot_id, content_hash=content_hash, doc_slug=doc_slug
        )
        return self.import_staging(
            pot_id=pot_id,
            doc_slug=doc_slug,
            staging_dir=stage,
            source_ref=source_ref,
            force=force,
        )

    def _parse_and_import_facade(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        source: Path,
        stage: Path,
        chunk_size: int,
        source_ref: str | None,
        vision_provider: str,
        allow_degraded: bool,
    ) -> ResourceImportReport:
        if not source.is_file():
            raise ValueError(f"source file not found: {source}")
        if stage.exists() and (not stage.is_dir() or any(stage.iterdir())):
            raise ValueError(f"staging directory must be empty: {stage}")
        manifest = parse_file_to_staging(
            source,
            stage,
            options=ParseOptions(
                chunk_target=chunk_size,
                vision_provider=vision_provider,
                allow_degraded=allow_degraded,
            ),
        )
        return self.import_staging(
            pot_id=pot_id,
            doc_slug=doc_slug,
            staging_dir=stage,
            source_ref=source_ref or manifest.source_ref,
        )

    def import_staging(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        staging_dir: str | Path,
        source_ref: str | None = None,
        force: bool = False,
        write_graph: bool = True,
    ) -> ResourceImportReport:
        doc_slug = validate_doc_slug(doc_slug)
        stage = Path(staging_dir).expanduser().resolve()
        if self.facade is not None:
            parsed = self.store.read_manifest(stage)
            element_reader = getattr(self.facade.store, "read_elements", None)
            previous_elements = (
                {row.element_id for row in element_reader(pot_id=pot_id, slug=doc_slug)}
                if callable(element_reader)
                else set()
            )
            result = self.facade.import_dir(
                pot_id=pot_id,
                slug=doc_slug,
                source_dir=stage,
                source_ref=source_ref,
                source_kind=parsed.source_kind,
            )
            stored = result.manifest
            current_elements = (
                {row.element_id for row in element_reader(pot_id=pot_id, slug=doc_slug)}
                if callable(element_reader)
                else set()
            )
            from potpie_context_engine.adapters.outbound.resources.fact_inbox import (
                FactInbox,
            )

            FactInbox(
                home=getattr(self.facade.store, "home", self.store.home)
            ).close_superseded(
                pot_id=pot_id,
                doc=doc_slug,
                revision=stored.revision,
            )
            current_slugs = {section.slug for section in stored.sections}
            changed = sorted(
                current_slugs - set(stored.sections_added) - set(stored.sections_kept)
            )
            return ResourceImportReport(
                pot_id=pot_id,
                doc_slug=doc_slug,
                sections_added=list(stored.sections_added),
                sections_kept=list(stored.sections_kept),
                sections_changed=changed,
                sections_removed=list(stored.sections_removed),
                elements_added=sorted(current_elements - previous_elements),
                elements_removed=sorted(previous_elements - current_elements),
                provenance_version=parsed.provenance_version,
                parser_tier=parsed.parser_tier,
                ingestion_diagnostics=dict(parsed.ingestion_diagnostics),
                summary_pending=[
                    section.slug
                    for section in stored.sections
                    if section.summary_pending
                ],
                graph_written=result.graph_written,
                missing_claim_keys=list(result.missing_claim_keys),
                recommended_next_action=(
                    "verify with potpie search --include docs"
                    if result.graph_written
                    else "inspect the resource graph mutation"
                ),
            )
        manifest = self.store.read_manifest(stage)
        if source_ref:
            manifest = manifest.model_copy(update={"source_ref": source_ref})

        report = self.store.import_dir(
            pot_id=pot_id,
            doc_slug=doc_slug,
            source_dir=stage,
            manifest=manifest,
            force=force,
        )
        if report.errors:
            return report

        if self.index is not None:
            self.index.index_document(
                pot_id=pot_id,
                doc_slug=doc_slug,
                chunks=self.store.iter_chunks(pot_id=pot_id, doc_slug=doc_slug),
            )

        if write_graph and self.graph is not None:
            elements = self.store.read_elements(pot_id=pot_id, doc_slug=doc_slug)
            mutation = parse_import_request(
                pot_id=pot_id,
                doc_slug=doc_slug,
                manifest=manifest,
                elements=elements,
                retract_sections=report.sections_removed,
                retract_elements=report.elements_removed,
            )
            result = self.graph.mutate(mutation)
            report.graph_written = result.ok and result.status == "applied"
            report.missing_claim_keys = (
                [] if report.graph_written else list(result.claim_keys)
            )
            if not report.graph_written:
                report.recommended_next_action = (
                    result.detail or "fix graph mutations and re-import"
                )
            else:
                report.recommended_next_action = (
                    "verify with potpie search --include docs"
                )
        return report

    def ingest_file(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        source_path: str | Path,
        chunk_size: int = 4000,
        force: bool = False,
        source_ref: str | None = None,
        vision_provider: str = "none",
        allow_degraded: bool = False,
        extract_facts: bool = False,
    ) -> ResourceImportReport:
        report = self.parse_and_import(
            pot_id=pot_id,
            doc_slug=doc_slug,
            source_path=source_path,
            chunk_size=chunk_size,
            force=force,
            source_ref=source_ref,
            vision_provider=vision_provider,
            allow_degraded=allow_degraded,
        )
        if extract_facts and self.facade is not None and report.graph_written:
            from potpie_context_engine.application.services.fact_extraction import (
                FactExtractor,
            )

            try:
                facts = FactExtractor(self.facade).run(pot_id=pot_id, doc=doc_slug)
                report.facts_verified = facts["verified"]
                report.facts_proposed = facts["proposed"]
                report.facts_suppressed = facts["suppressed"]
                report.fact_extraction_error = facts.get("error")
            except Exception as exc:
                report.fact_extraction_error = str(exc)
        return report

    def get_chunk(
        self,
        *,
        pot_id: str,
        uri: str,
        with_neighbors: bool = False,
    ) -> dict[str, Any]:
        if self.facade is not None:
            resource = parse_resource_id(uri)
            revision = resource.revision
            if revision is None:
                manifest = self.facade.current_manifest(
                    pot_id=pot_id, slug=resource.doc
                )
                revision = manifest.revision
            resource_id = format_resource_id(
                resource.doc,
                resource.section,
                resource.seq,
                revision=revision,
            )
            result = self.facade.get(
                pot_id=pot_id,
                resource_ids=(resource_id,),
                with_neighbors=with_neighbors,
            )
            if isinstance(result, tuple) and result:
                payload = result[0].to_payload()
                artifact_reader = getattr(self.facade.store, "figure_artifacts", None)
                provenance = result[0].provenance or []
                if callable(artifact_reader) and provenance:
                    current = self.facade.current_manifest(
                        pot_id=pot_id, slug=resource.doc
                    )
                    if current.revision == revision:
                        payload["figure_artifacts"] = artifact_reader(
                            pot_id=pot_id,
                            slug=resource.doc,
                            element_ids={
                                row["element_id"]
                                for row in provenance
                                if isinstance(row, dict) and row.get("element_id")
                            },
                        )
                return payload
            raise ResourceStoreError("resource_not_found", f"chunk not found: {uri}")
        doc_slug, section_slug, seq = parse_chunk_uri(uri)
        return self.store.get(
            pot_id=pot_id,
            doc_slug=doc_slug,
            section_slug=section_slug,
            seq=seq,
            with_neighbors=with_neighbors,
        ).to_payload()

    def list_documents(self, *, pot_id: str) -> list[dict[str, Any]]:
        if self.facade is not None:
            listing = getattr(self.facade.store, "list_documents", None)
            if callable(listing):
                return [row.to_payload() for row in listing(pot_id=pot_id)]
            return []
        return [info.to_payload() for info in self.store.list_documents(pot_id=pot_id)]

    def remove_document(self, *, pot_id: str, doc_slug: str) -> dict[str, Any]:
        doc_slug = validate_doc_slug(doc_slug)
        if self.facade is not None:
            result = self.facade.delete(pot_id=pot_id, slug=doc_slug)
            return {
                "doc_slug": doc_slug,
                "removed": result.removed,
                "graph_retracted": result.graph_retracted,
            }
        section_slugs = [
            section.slug
            for section in self.store.list_sections(pot_id=pot_id, doc_slug=doc_slug)
        ]
        element_ids = [
            element.element_id
            for element in self.store.read_elements(pot_id=pot_id, doc_slug=doc_slug)
        ]

        if self.graph is not None and (section_slugs or element_ids):
            payload = {
                "pot_id": pot_id,
                "operations": build_retract_mutations(
                    pot_id=pot_id,
                    doc_slug=doc_slug,
                    section_slugs=section_slugs,
                    element_ids=element_ids,
                ),
                "created_by": {"surface": "cli", "harness": "resource-remove"},
            }
            from potpie_context_engine.core.semantic_mutations import (
                SemanticMutationRequest,
            )

            mutation = SemanticMutationRequest.parse(
                payload, approved_by="resource-remove"
            )
            self.graph.mutate(mutation)

        if self.index is not None:
            self.index.remove_document(pot_id=pot_id, doc_slug=doc_slug)
        result = self.store.delete(pot_id=pot_id, doc_slug=doc_slug)
        result["sections_retracted"] = section_slugs
        result["elements_retracted"] = element_ids
        return result

    def search_chunks(
        self, *, pot_id: str, query: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        if self.index is None:
            return []
        enriched: list[dict[str, Any]] = []
        for hit in self.index.search(pot_id=pot_id, query=query, limit=limit):
            try:
                chunk = self.store.get(
                    pot_id=pot_id,
                    doc_slug=hit.doc_slug,
                    section_slug=hit.section_slug,
                    seq=hit.seq,
                )
            except ResourceStoreError:
                continue
            payload = chunk.to_payload()
            payload["score"] = hit.score
            enriched.append(payload)
        return enriched


__all__ = ["ResourceService"]

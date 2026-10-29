"""Resource ingestion orchestration (parse, import, get, list, remove).

The service depends on two ports, never on concrete storage:
``ResourceStorePort`` owns bytes + catalog (local disk today, blob storage
later), ``ResourceIndexPort`` owns retrieval (SQLite FTS today, hosted or
semantic profiles later). Swapping either is wiring, not surgery, and the
CLI/RPC payload shapes produced here do not change with the backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import json
from pathlib import Path
import re
import shutil
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
from potpie_context_engine.core.ports.resource_index import (
    ResourceIndexPort,
    ResourceSearchFilters,
)
from potpie_context_engine.core.ports.resource_store import (
    ResourceStoreError,
    ResourceStorePort,
)


@dataclass(slots=True)
class ResourceService:
    store: ResourceStorePort = field(default_factory=LocalResourceStore)
    index: ResourceIndexPort | None = None
    graph: GraphService | None = None

    def __post_init__(self) -> None:
        if self.index is None:
            self.index = self.store.default_index()

    def parse_to_staging(
        self,
        *,
        source_path: str | Path,
        out_dir: str | Path,
        chunk_size: int = 4000,
        force: bool = False,
        pot_id: str | None = None,
        vision_provider: str = "local",
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
        vision_provider: str = "local",
        allow_degraded: bool = False,
    ) -> ResourceImportReport:
        doc_slug = validate_doc_slug(doc_slug)
        source = Path(source_path).expanduser().resolve()
        if staging_dir:
            stage = Path(staging_dir).expanduser().resolve()
        else:
            stage = self.store.staging_root(pot_id=pot_id) / doc_slug
        content_hash = file_sha256(source)
        if (
            not force
            and self.store.is_file_imported(
                pot_id=pot_id, content_hash=content_hash
            )
        ):
            raise ValueError(
                f"file already imported (content_hash={content_hash}); use --force to re-parse"
            )
        if stage.exists() and not force and self._stage_matches_source(stage, content_hash):
            pass
        elif stage.exists():
            shutil.rmtree(stage)
        if not stage.exists():
            parsed = self.parse_to_staging(
                source_path=source,
                out_dir=stage,
                chunk_size=chunk_size,
                force=force,
                pot_id=pot_id,
                vision_provider=vision_provider,
                allow_degraded=allow_degraded,
            )
            meta_path = stage / "meta.json"
            if meta_path.is_file():
                persisted = parsed.model_copy(
                    update={"source_content_hash": content_hash}
                )
                meta_path.write_text(
                    json.dumps(persisted.model_dump(mode="json"), indent=2),
                    encoding="utf-8",
                )
        return self.import_staging(
            pot_id=pot_id,
            doc_slug=doc_slug,
            staging_dir=stage,
            source_ref=source_ref,
            force=force,
            require_graph=True,
            content_hash=content_hash,
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
        require_graph: bool = False,
        content_hash: str | None = None,
    ) -> ResourceImportReport:
        doc_slug = validate_doc_slug(doc_slug)
        stage = Path(staging_dir).expanduser().resolve()
        manifest = self.store.read_manifest(stage)
        if source_ref:
            manifest = manifest.model_copy(update={"source_ref": source_ref})

        snapshot = None
        snapshot_fn = getattr(self.store, "snapshot_document", None)
        if callable(snapshot_fn):
            snapshot = snapshot_fn(pot_id=pot_id, doc_slug=doc_slug)
        try:
            report = self.store.import_dir(
                pot_id=pot_id,
                doc_slug=doc_slug,
                source_dir=stage,
                manifest=manifest,
                force=force,
            )
        except Exception:
            if snapshot is not None:
                restore_fn = getattr(self.store, "restore_document_snapshot", None)
                if callable(restore_fn):
                    restore_fn(pot_id=pot_id, doc_slug=doc_slug, snapshot=snapshot)
            raise
        if report.errors:
            if snapshot is not None:
                discard_fn = getattr(self.store, "discard_document_snapshot", None)
                if callable(discard_fn):
                    discard_fn(snapshot)
            return report

        if self.index is None and require_graph:
            if snapshot is not None:
                restore_fn = getattr(self.store, "restore_document_snapshot", None)
                if callable(restore_fn):
                    restore_fn(pot_id=pot_id, doc_slug=doc_slug, snapshot=snapshot)
            report.errors.append(
                {
                    "code": "resource_index_unavailable",
                    "message": "document ingest requires a searchable resource index",
                }
            )
            report.recommended_next_action = "configure the resource index and retry"
            return report

        try:
            if self.index is not None:
                chunks = list(
                    self.store.iter_chunks(pot_id=pot_id, doc_slug=doc_slug)
                )
                indexed = self.index.index_document(
                    pot_id=pot_id,
                    doc_slug=doc_slug,
                    chunks=chunks,
                )
                if indexed != len(chunks):
                    raise RuntimeError(
                        f"indexed {indexed} chunks, expected {len(chunks)}"
                    )
        except Exception as exc:  # noqa: BLE001
            if snapshot is not None:
                restore_fn = getattr(self.store, "restore_document_snapshot", None)
                if callable(restore_fn):
                    restore_fn(pot_id=pot_id, doc_slug=doc_slug, snapshot=snapshot)
            report.errors.append(
                {
                    "code": "resource_index_failed",
                    "message": str(exc),
                }
            )
            report.recommended_next_action = "retry the import to rebuild the index"
            return report

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
            try:
                result = self.graph.mutate(mutation)
            except Exception as exc:  # noqa: BLE001
                report.errors.append(
                    {
                        "code": "resource_graph_write_failed",
                        "message": str(exc),
                    }
                )
                report.recommended_next_action = (
                    "retry the import to resume the graph write"
                )
                if snapshot is not None:
                    discard_fn = getattr(self.store, "discard_document_snapshot", None)
                    if callable(discard_fn):
                        discard_fn(snapshot)
                return report
            report.graph_written = result.ok and result.status == "applied"
            report.missing_claim_keys = (
                [] if report.graph_written else list(getattr(result, "claim_keys", ()))
            )
            if not report.graph_written:
                report.errors.append(
                    {
                        "code": "resource_graph_write_failed",
                        "message": result.detail or "graph mutation was not applied",
                    }
                )
                report.recommended_next_action = (
                    result.detail or "fix graph mutations and re-import"
                )
            else:
                report.recommended_next_action = (
                    "verify with potpie search --include docs"
                )
        elif require_graph:
            report.errors.append(
                {
                    "code": "resource_graph_unavailable",
                    "message": "document ingest requires a successful graph write",
                }
            )
            report.recommended_next_action = (
                "run document ingest with the graph service available"
            )

        if report.errors:
            if snapshot is not None:
                discard_fn = getattr(self.store, "discard_document_snapshot", None)
                if callable(discard_fn):
                    discard_fn(snapshot)
            return report

        if content_hash is not None:
            try:
                self.store.record_file_hash(
                    pot_id=pot_id, content_hash=content_hash, doc_slug=doc_slug
                )
            except Exception as exc:  # noqa: BLE001
                report.errors.append(
                    {
                        "code": "resource_completion_marker_failed",
                        "message": str(exc),
                    }
                )
                report.recommended_next_action = "retry the import completion step"
                return report
        if snapshot is not None:
            discard_fn = getattr(self.store, "discard_document_snapshot", None)
            if callable(discard_fn):
                discard_fn(snapshot)
        return report

    def _stage_matches_source(self, stage: Path, content_hash: str) -> bool:
        try:
            manifest = self.store.read_manifest(stage)
        except Exception:  # noqa: BLE001
            return False
        return manifest.source_content_hash == content_hash

    def ingest_file(
        self,
        *,
        pot_id: str,
        doc_slug: str,
        source_path: str | Path,
        chunk_size: int = 4000,
        force: bool = False,
        source_ref: str | None = None,
        vision_provider: str = "local",
        allow_degraded: bool = False,
    ) -> ResourceImportReport:
        return self.parse_and_import(
            pot_id=pot_id,
            doc_slug=doc_slug,
            source_path=source_path,
            chunk_size=chunk_size,
            force=force,
            source_ref=source_ref,
            vision_provider=vision_provider,
            allow_degraded=allow_degraded,
        )

    def get_chunk(
        self,
        *,
        pot_id: str,
        uri: str,
        with_neighbors: bool = False,
    ) -> dict[str, Any]:
        doc_slug, section_slug, seq = parse_chunk_uri(uri)
        store_parameters = inspect.signature(self.store.get).parameters
        if "resource_id" in store_parameters:
            resource_id = uri
            return self.store.get(
                pot_id=pot_id, resource_id=resource_id
            ).to_payload()
        return self.store.get(
            pot_id=pot_id,
            doc_slug=doc_slug,
            section_slug=section_slug,
            seq=seq,
            with_neighbors=with_neighbors,
        ).to_payload()

    def list_documents(self, *, pot_id: str) -> list[dict[str, Any]]:
        return [info.to_payload() for info in self.store.list_documents(pot_id=pot_id)]

    def remove_document(self, *, pot_id: str, doc_slug: str) -> dict[str, Any]:
        doc_slug = validate_doc_slug(doc_slug)
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
        self,
        *,
        pot_id: str,
        query: str,
        limit: int = 20,
        doc_slug: str | None = None,
        source_kind: str | None = None,
        page_number: int | None = None,
    ) -> list[dict[str, Any]]:
        if self.index is None:
            return []
        filters = ResourceSearchFilters(
            doc_slug=doc_slug,
            source_kind=source_kind,
            page_number=page_number,
        )
        enriched: list[dict[str, Any]] = []
        search_parameters = inspect.signature(self.index.search).parameters
        if "filters" in search_parameters:
            hits = self.index.search(
                pot_id=pot_id, query=query, limit=limit, filters=filters
            )
        elif "doc" in search_parameters:
            # New typed indexes use a document selector and leave source/page
            # filtering to the hydrated store payload below.  A **kwargs
            # signature alone is not evidence that an index understands the
            # legacy ``filters`` object.
            hits = self.index.search(
                pot_id=pot_id, query=query, limit=limit, doc=doc_slug
            )
        else:
            hits = self.index.search(pot_id=pot_id, query=query, limit=limit)
        # Typed indexes return an ``IndexSearchResult`` envelope; the legacy
        # parser service historically consumed a plain hit list.  Normalize at
        # this compatibility boundary so both retrieval generations remain
        # usable during the CLI migration.
        if hasattr(hits, "hits"):
            hits = hits.hits
        for hit in hits:
            try:
                store_parameters = inspect.signature(self.store.get).parameters
                if "resource_id" in store_parameters:
                    chunk = self.store.get(
                        pot_id=pot_id,
                        resource_id=hit.resource_id,
                    )
                else:
                    chunk = self.store.get(
                        pot_id=pot_id,
                        doc_slug=hit.doc_slug,
                        section_slug=hit.section_slug,
                        seq=hit.seq,
                    )
            except ResourceStoreError:
                continue
            payload = chunk.to_payload()
            if doc_slug is not None and payload.get("doc_slug") != doc_slug:
                continue
            if source_kind is not None and payload.get("source_kind") != source_kind:
                continue
            if page_number is not None and page_number not in payload.get(
                "page_numbers", []
            ):
                continue
            if not _supports_exact_metric_query(query, payload):
                continue
            payload["score"] = hit.score
            if hit.fts_score is not None:
                payload["fts_score"] = hit.fts_score
            if hit.semantic_score is not None:
                payload["semantic_score"] = hit.semantic_score
            if hit.retrieval_source is not None:
                payload["retrieval_source"] = hit.retrieval_source
            payload["facet_coverage"] = _facet_coverage(query, payload)
            payload["structural_match"] = _structural_match(query, payload)
            enriched.append(payload)
        facets = _query_facets(query)
        covered = {
            facet_index
            for payload in enriched
            for facet_index in payload.get("facet_coverage", [])
        }
        bundle_complete = not facets or covered == set(range(len(facets)))
        for payload in enriched:
            payload["answer_bundle_complete"] = bundle_complete
        enriched.sort(
            key=lambda payload: (
                -float(payload["structural_match"].get("score", 0.0)),
                -float(payload.get("score") or 0.0),
            )
        )
        return enriched


def _supports_exact_metric_query(query: str, payload: dict[str, Any]) -> bool:
    """Abstain when an exact named benchmark/metric is absent from evidence."""
    lowered_query = query.casefold()
    if "exact" not in lowered_query and "p-value" not in lowered_query:
        return True
    metric_terms = {
        token
        for token in re.findall(r"[\w-]+", lowered_query)
        if token in {"accuracy", "score", "p-value", "perplexity", "bleu", "f1"}
    }
    if not metric_terms:
        return True
    searchable = " ".join(
        str(payload.get(key) or "")
        for key in ("content", "section_title", "ocr_text")
    ).casefold()
    if not any(term in searchable for term in metric_terms):
        return False
    named_terms = {
        token.casefold()
        for token in re.findall(r"\b[A-Z][\w-]{2,}\b", query)
        if token.casefold() not in {"What", "Which", "GPT"}
    }
    return all(term in searchable for term in named_terms)


_FACET_STOPWORDS = frozenset(
    "a an and are as at be but by did do does for from had has have how in into "
    "is it its no not of on or than that the their then there these this those "
    "to was were what when where which who why will with would"
    .split()
)
_FACET_SYNONYMS = {
    "omitted": {"removed", "excluded"},
    "excluded": {"removed", "omitted"},
    "encyclopedia": {"wikipedia"},
    "wikipedia": {"encyclopedia"},
    "underfits": {"underfitting", "underfit"},
    "underfitting": {"underfits", "underfit"},
}


def _query_facets(query: str) -> list[str]:
    return [
        facet.strip()
        for facet in re.split(r"\band\b|[;|]", query, flags=re.IGNORECASE)
        if facet.strip()
    ]


def _facet_coverage(query: str, payload: dict[str, Any]) -> list[int]:
    searchable = str(
        payload.get("evidence_text")
        or payload.get("content")
        or ""
    ).casefold()
    coverage: list[int] = []
    for index, facet in enumerate(_query_facets(query)):
        terms = {
            token
            for token in re.findall(r"[\w-]+", facet.casefold())
            if len(token) > 1 and token not in _FACET_STOPWORDS
        }
        expanded = terms | {
            variant
            for term in terms
            for variant in _FACET_SYNONYMS.get(term, set())
        }
        if any(term in searchable for term in expanded):
            coverage.append(index)
    return coverage


def _structural_match(query: str, payload: dict[str, Any]) -> dict[str, Any]:
    searchable = str(
        payload.get("evidence_text")
        or payload.get("content")
        or ""
    ).casefold()
    identifiers = [
        f"{kind.casefold()} {number}"
        for kind, number in re.findall(
            r"\b(table|figure)\s+(\d+)\b", query, flags=re.IGNORECASE
        )
    ]
    numeric_tokens = re.findall(
        r"\b\d+(?:\.\d+)?%?\b", query.casefold()
    )
    identifier_matches = sum(identifier in searchable for identifier in identifiers)
    numeric_matches = sum(token in searchable for token in numeric_tokens)
    header_terms = {
        token
        for token in re.findall(r"[\w-]+", query.casefold())
        if token in {"model", "accuracy", "score", "value", "parameters", "bleu"}
    }
    header_matches = sum(term in searchable for term in header_terms)
    score = min(
        1.0,
        0.50 * min(1.0, identifier_matches / max(1, len(identifiers)))
        + 0.35 * min(1.0, numeric_matches / max(1, len(numeric_tokens)))
        + 0.15 * min(1.0, header_matches / max(1, len(header_terms))),
    )
    return {
        "score": score,
        "identifier_matches": identifier_matches,
        "numeric_matches": numeric_matches,
        "header_matches": header_matches,
    }


__all__ = ["ResourceService"]

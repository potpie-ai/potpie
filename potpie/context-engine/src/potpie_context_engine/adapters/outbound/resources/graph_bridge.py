"""Build semantic mutations for resource import graph projection."""

from __future__ import annotations

from typing import Any

from potpie_context_engine.domain.resource_models import (
    DocumentElementRecord,
    ResourceManifest,
    chunk_uri,
)
from potpie_context_core.semantic_mutations import SemanticMutationRequest


def document_entity_key(doc_slug: str) -> str:
    return f"document:{doc_slug}"


def section_entity_key(doc_slug: str, section_slug: str) -> str:
    return f"document-section:{doc_slug}/{section_slug}"


def element_entity_key(doc_slug: str, element_id: str) -> str:
    return f"document-element:{doc_slug}/{element_id}"


def build_retract_mutations(
    *,
    pot_id: str,
    doc_slug: str,
    section_slugs: list[str],
    element_ids: list[str],
) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    doc_key = document_entity_key(doc_slug)

    for slug in section_slugs:
        sec_key = section_entity_key(doc_slug, slug)
        for predicate in ("SECTION_OF", "RELATED_TO"):
            operations.append(
                {
                    "op": "retract_claim",
                    "subgraph": "knowledge",
                    "subject": {"key": sec_key, "type": "DocumentSection"},
                    "predicate": predicate,
                    "object": {"key": doc_key, "type": "Document"},
                    "reason": "document re-import or removal dropped section",
                }
            )

    for element_id in element_ids:
        elem_key = element_entity_key(doc_slug, element_id)
        operations.append(
            {
                "op": "retract_claim",
                "subgraph": "knowledge",
                "subject": {"key": elem_key, "type": "DocumentElement"},
                "predicate": "ELEMENT_OF",
                "object": {"key": doc_key, "type": "Document"},
                "reason": "document re-import or removal dropped element",
            }
        )

    return operations


def build_import_mutations(
    *,
    pot_id: str,
    doc_slug: str,
    manifest: ResourceManifest,
    elements: list[DocumentElementRecord] | None = None,
    retract_sections: list[str] | None = None,
    retract_elements: list[str] | None = None,
) -> dict[str, Any]:
    operations: list[dict[str, Any]] = []
    doc_key = document_entity_key(doc_slug)
    doc_summary = manifest.source_ref or doc_slug

    if retract_sections or retract_elements:
        operations.extend(
            build_retract_mutations(
                pot_id=pot_id,
                doc_slug=doc_slug,
                section_slugs=retract_sections or [],
                element_ids=retract_elements or [],
            )
        )

    for section in manifest.sections:
        sec_key = section_entity_key(doc_slug, section.slug)
        summary = section.summary.strip() or section.title
        evidence = [
            {"source_ref": chunk_uri(doc_slug, section.slug, chunk_ref.seq)}
            for chunk_ref in section.chunks
        ]
        if manifest.source_ref:
            evidence.insert(0, {"source_ref": manifest.source_ref})
        operations.append(
            {
                "op": "link_entities",
                "subgraph": "knowledge",
                "subject": {
                    "key": sec_key,
                    "type": "DocumentSection",
                    "name": section.title,
                    "summary": summary[:2000],
                    "description": summary[:2000],
                },
                "predicate": "SECTION_OF",
                "object": {
                    "key": doc_key,
                    "type": "Document",
                    "name": doc_slug,
                    "summary": doc_summary[:2000],
                    "description": doc_summary[:2000],
                    "properties": {
                        "source_kind": manifest.source_kind,
                        "provenance_version": manifest.provenance_version,
                        "parser_tier": manifest.parser_tier,
                        "source_content_hash": manifest.source_content_hash,
                    },
                },
                "truth": "source_observation",
                "confidence": 1.0,
                "evidence": evidence,
            }
        )
        operations.append(
            {
                "op": "assert_claim",
                "subgraph": "knowledge",
                "subject": {
                    "key": sec_key,
                    "type": "DocumentSection",
                    "summary": summary[:2000],
                    "description": summary[:2000],
                },
                "predicate": "RELATED_TO",
                "object": {"key": doc_key, "type": "Document"},
                "truth": "source_observation",
                "confidence": 0.95,
                "description": summary[:2000],
                "evidence": evidence,
            }
        )

    for element in elements or []:
        elem_key = element_entity_key(doc_slug, element.element_id)
        label = element.element_type
        summary_bits = [label, element.text[:200] if element.text else ""]
        if element.page_number is not None:
            summary_bits.append(f"page {element.page_number}")
        summary = " — ".join([b for b in summary_bits if b]).strip() or element.element_id
        evidence: list[dict[str, Any]] = []
        if manifest.source_ref:
            evidence.append({"source_ref": manifest.source_ref})
        if element.artifact_ref:
            evidence.append({"source_ref": element.artifact_ref})
        properties: dict[str, Any] = {
            "element_type": element.element_type,
            "page_number": element.page_number,
            "bbox": element.bbox,
            "text_hash": element.text_hash,
        }
        if element.artifact_ref:
            properties["artifact_ref"] = element.artifact_ref
        operations.append(
            {
                "op": "link_entities",
                "subgraph": "knowledge",
                "subject": {
                    "key": elem_key,
                    "type": "DocumentElement",
                    "name": element.element_id,
                    "summary": summary[:2000],
                    "description": summary[:2000],
                    "properties": properties,
                },
                "predicate": "ELEMENT_OF",
                "object": {
                    "key": doc_key,
                    "type": "Document",
                    "name": doc_slug,
                },
                "truth": "source_observation",
                "confidence": 1.0,
                "evidence": evidence,
            }
        )

    return {
        "pot_id": pot_id,
        "operations": operations,
        "created_by": {"surface": "cli", "harness": "resource-import"},
    }


def parse_import_request(
    *,
    pot_id: str,
    doc_slug: str,
    manifest: ResourceManifest,
    elements: list[DocumentElementRecord] | None = None,
    retract_sections: list[str] | None = None,
    retract_elements: list[str] | None = None,
) -> SemanticMutationRequest:
    payload = build_import_mutations(
        pot_id=pot_id,
        doc_slug=doc_slug,
        manifest=manifest,
        elements=elements,
        retract_sections=retract_sections,
        retract_elements=retract_elements,
    )
    return SemanticMutationRequest.parse(payload, approved_by="resource-import")


__all__ = [
    "build_import_mutations",
    "build_retract_mutations",
    "document_entity_key",
    "element_entity_key",
    "parse_import_request",
    "section_entity_key",
]

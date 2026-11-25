"""Stable resource revision intent and graph-dependent preparation effects."""

from dataclasses import replace
from .resource_review import review_operations
from typing import TYPE_CHECKING

from potpie_context_engine.core.errors import CapabilityNotImplemented
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.ports.resource_store import (
    DocumentManifest,
    ResourceStoreError,
)
from potpie_context_engine.core.resource_to_semantic import (
    SECTION_PREDICATE,
    document_key,
    section_key,
    resource_import_to_semantic_request,
    resource_delete_to_semantic_request,
    DependentClaim,
    _dependent_retractions,
)
from potpie_context_engine.core.semantic_mutations import SemanticMutation

ELEMENT_PREDICATE = "ELEMENT_OF"

if TYPE_CHECKING:
    from .resource_facade import ResourceFacade


def _check_source(facade: "ResourceFacade", manifest: DocumentManifest) -> None:
    current = facade.store.current_manifest(pot_id=manifest.pot_id, slug=manifest.doc)
    if current.revision != manifest.revision:
        raise ResourceStoreError(
            "resource_revision_changed",
            "Document changed during graph preparation; import or delete the current revision explicitly.",
        )


def import_graph(facade: "ResourceFacade", manifest: DocumentManifest):
    # Import reports contain transient diffs. Only the revision's desired graph
    # assertions bind the request; cleanup is resolved under preparation's
    # revision bracket, including recovery before a plan was ever reserved.
    intent = resource_import_to_semantic_request(replace(manifest, sections_removed=()))
    intent = replace(
        intent, operations=intent.operations + _element_operations(facade, manifest)
    )

    def resolve():
        _check_source(facade, manifest)
        if facade.claims is None:
            raise CapabilityNotImplemented("resources.graph_reconciliation")
        desired = {
            section_key(manifest.doc, section.slug) for section in manifest.sections
        }
        rows = facade.claims.find_claims(
            ClaimQueryFilter(
                pot_id=manifest.pot_id,
                predicate_in=(SECTION_PREDICATE,),
                object_key_in=(document_key(manifest.doc),),
            )
        )
        prefix = section_key(manifest.doc, "")
        removed = tuple(
            sorted(
                {
                    row.subject_key[len(prefix) :]
                    for row in rows
                    if row.subject_key.startswith(prefix)
                    and row.subject_key not in desired
                }
            )
        )
        request = resource_import_to_semantic_request(
            replace(manifest, sections_removed=removed),
            dependent_claims=facade._dependent_claims(
                pot_id=manifest.pot_id,
                section_keys=tuple(section_key(manifest.doc, slug) for slug in removed),
            ),
        )
        element_rows = facade.claims.find_claims(
            ClaimQueryFilter(
                pot_id=manifest.pot_id,
                predicate_in=(ELEMENT_PREDICATE,),
                object_key_in=(document_key(manifest.doc),),
            )
        )
        desired_elements = {
            op.subject.key
            for op in _element_operations(facade, manifest)
            if op.subject is not None
        }
        removed_elements = [
            row for row in element_rows if row.subject_key not in desired_elements
        ]
        return _with_exact_structure(
            request.operations + _element_operations(facade, manifest),
            [row for row in rows if row.subject_key not in desired] + removed_elements,
        ) + review_operations(facade.claims, manifest)

    return facade.graph.mutate(intent, resolve_operations=resolve)


def delete_graph(facade: "ResourceFacade", manifest: DocumentManifest):
    slugs = tuple(sorted(section.slug for section in manifest.sections))
    intent = resource_delete_to_semantic_request(
        pot_id=manifest.pot_id,
        doc=manifest.doc,
        revision=manifest.revision,
        section_slugs=slugs,
    )

    def resolve():
        _check_source(facade, manifest)
        if facade.claims is None:
            raise CapabilityNotImplemented("resources.graph_reconciliation")
        request = resource_delete_to_semantic_request(
            pot_id=manifest.pot_id,
            doc=manifest.doc,
            revision=manifest.revision,
            section_slugs=slugs,
            dependent_claims=facade._dependent_claims(
                pot_id=manifest.pot_id,
                section_keys=tuple(section_key(manifest.doc, slug) for slug in slugs),
            ),
        )
        rows = facade.claims.find_claims(
            ClaimQueryFilter(
                pot_id=manifest.pot_id,
                predicate_in=(SECTION_PREDICATE,),
                object_key_in=(document_key(manifest.doc),),
            )
        )
        element_rows = facade.claims.find_claims(
            ClaimQueryFilter(
                pot_id=manifest.pot_id,
                predicate_in=(ELEMENT_PREDICATE,),
                object_key_in=(document_key(manifest.doc),),
            )
        )
        return _with_exact_structure(
            request.operations, list(rows) + list(element_rows)
        ) + review_operations(facade.claims, manifest)

    return facade.graph.mutate(intent, resolve_operations=resolve)


def _with_exact_structure(operations, rows):
    """Resolve structural cleanup to the same physical contract as dependencies."""
    from potpie_context_engine.core.semantic_mutations import SemanticMutation

    retained = tuple(
        op
        for op in operations
        if not (
            op.op == "retract_claim"
            and op.predicate in {SECTION_PREDICATE, ELEMENT_PREDICATE}
        )
    )
    if any(not row.record_id or not row.claim_key for row in rows):
        raise ResourceStoreError(
            "migration_required", "resource structure lacks incarnation identity"
        )
    claims = tuple(
        DependentClaim(
            predicate=row.predicate,
            subject_key=row.subject_key,
            object_key=row.object_key,
            subgraph=row.subgraph or "knowledge",
            claim_key=row.claim_key,
            record_id=row.record_id,
            environment=row.environment,
            source_system=row.source_system,
            source_ref=row.source_ref,
        )
        for row in rows
    )
    return retained + tuple(
        SemanticMutation.parse(raw)
        for raw in _dependent_retractions(
            claims, reason="resource section is no longer present"
        )
    )


def _element_operations(
    facade: "ResourceFacade", manifest: DocumentManifest
) -> tuple[SemanticMutation, ...]:
    reader = getattr(facade.store, "read_elements", None)
    if not callable(reader):
        return ()
    from potpie_context_engine.adapters.outbound.resources.graph_bridge import (
        element_entity_key,
    )

    operations = []
    for element in reader(pot_id=manifest.pot_id, slug=manifest.doc):
        summary = f"{element.element_type} {element.text[:180]}".strip()
        if element.page_number is not None:
            summary += f" (page {element.page_number})"
        evidence = [manifest.source_ref] if manifest.source_ref else []
        operations.append(
            SemanticMutation.parse(
                {
                    "op": "link_entities",
                    "subgraph": "knowledge",
                    "subject": {
                        "key": element_entity_key(manifest.doc, element.element_id),
                        "type": "DocumentElement",
                        "name": element.element_id,
                        "summary": summary[:2000],
                        "properties": {
                            "element_type": element.element_type,
                            "page_number": element.page_number,
                            "bbox": element.bbox,
                            "text_hash": element.text_hash,
                            "artifact_ref": element.artifact_ref,
                        },
                    },
                    "predicate": ELEMENT_PREDICATE,
                    "object": {"key": document_key(manifest.doc), "type": "Document"},
                    "truth": "source_observation",
                    "confidence": 1.0,
                    "evidence": evidence,
                }
            )
        )
    return tuple(operations)

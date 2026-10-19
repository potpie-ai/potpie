"""Stable resource revision intent and graph-dependent preparation effects."""

from dataclasses import replace
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
)

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
        return resource_import_to_semantic_request(
            replace(manifest, sections_removed=removed),
            dependent_claims=facade._dependent_claims(
                pot_id=manifest.pot_id,
                section_keys=tuple(section_key(manifest.doc, slug) for slug in removed),
            ),
        ).operations

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
        return resource_delete_to_semantic_request(
            pot_id=manifest.pot_id,
            doc=manifest.doc,
            revision=manifest.revision,
            section_slugs=slugs,
            dependent_claims=facade._dependent_claims(
                pot_id=manifest.pot_id,
                section_keys=tuple(section_key(manifest.doc, slug) for slug in slugs),
            ),
        ).operations

    return facade.graph.mutate(intent, resolve_operations=resolve)

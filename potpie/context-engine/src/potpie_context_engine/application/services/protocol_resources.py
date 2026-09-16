"""Protect source bytes used by protocol extractions until explicit reconciliation."""

from potpie_context_core.ports.claim_query import (
    ClaimQueryFilter,
    entity_properties_many,
)
from potpie_context_core.ports.resource_store import (
    RESOURCE_NOT_FOUND,
    ResourceStoreError,
    format_resource_id,
    read_import_files,
)
from potpie_context_core.protocols import (
    PREFIXES,
    normalize_properties,
    property_evidence,
)


def protect_protocol_source(
    store, claims, *, pot_id: str, slug: str, files=None, source_dir=None
) -> None:
    if claims is None:
        return
    try:
        sections = store.list(pot_id=pot_id, slug=slug)
    except ResourceStoreError as exc:
        if exc.code == RESOURCE_NOT_FOUND:
            return
        raise
    refs = tuple(
        format_resource_id(slug, section.slug, ref.seq)
        for section in sections
        for ref in section.chunks
    )
    if not refs:
        return
    rows = claims.find_claims(
        ClaimQueryFilter(
            pot_id=pot_id, subgraph_in=("protocols",), source_ref_in=refs, limit=1
        )
    )
    if not rows:
        # Corrections and source_coverage are current entity properties, not new
        # predicates. On this infrequent destructive path, inspect every live
        # protocol endpoint so correction-only sources cannot slip through.
        live = claims.find_claims(
            ClaimQueryFilter(pot_id=pot_id, subgraph_in=("protocols",))
        )
        labels = {prefix: label for label, prefix in PREFIXES.items()}
        keys = sorted(
            {
                key
                for row in live
                for key in (row.subject_key, row.object_key)
                if key.partition(":")[0] in labels
            }
        )
        for start in range(0, len(keys), 256):
            properties = entity_properties_many(
                claims, pot_id=pot_id, entity_keys=keys[start : start + 256]
            )
            if any(
                ev.get("source_ref") in refs
                for key, props in properties.items()
                for ev in property_evidence(
                    normalize_properties(props, labels[key.partition(":")[0]])
                )
            ):
                rows = live[:1]
                break
    if rows:
        incoming = (
            files
            if files is not None
            else read_import_files(source_dir)
            if source_dir is not None
            else None
        )
        if incoming is not None:
            chunks = store.get_many(pot_id=pot_id, resource_ids=refs)
            if len(chunks) == len(refs) and all(
                incoming.get(f"{chunk.section}/{chunk.seq:04}.txt") == chunk.text
                for chunk in chunks
            ):
                return  # Identical evidence bytes are safe to import again.
        raise ResourceStoreError(
            "protocol_source_in_use",
            "Resource bytes are evidence for live protocol definitions.",
            recommended_next_action="Import a new immutable slug (include the source digest). Before removal, explicitly retract dependent protocol claims and correct source coverage through graph propose/commit; retain the original bytes for history.",
        )

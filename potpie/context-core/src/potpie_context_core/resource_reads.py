"""Batch resource receipts with deduplicated immutable neighbor bodies."""

from __future__ import annotations

from dataclasses import replace
from itertools import islice
from typing import Callable, Sequence

from potpie_context_core.ports.resource_store import (
    Chunk,
    DocumentManifest,
    ResourceBatchResult,
    ResourceId,
    ResourceReadOutcome,
    ResourceStoreError,
    RESOURCE_READ_BUDGET_EXCEEDED,
    SectionManifest,
    format_resource_id,
    parse_resource_id,
)


def resource_choices(manifest: DocumentManifest, resource: ResourceId) -> dict:
    preferred = [
        section for section in manifest.sections if section.slug == resource.section
    ]
    sections = preferred or manifest.sections
    candidates = list(
        islice(
            (
                {
                    "resource_id": format_resource_id(
                        manifest.doc, section.slug, ref.seq, revision=manifest.revision
                    ),
                    "section": section.slug,
                    "revision": manifest.revision,
                    "label": ref.label,
                }
                for section in sections
                for ref in section.chunks
            ),
            5,
        )
    )
    return {
        "candidates": candidates,
        "requested_revision": resource.revision,
        "candidate_status": "choices_only",
        "revision_substituted": False,
    }


def read_batch(
    resource_ids: tuple[str, ...],
    *,
    pot_id: str,
    read: Callable[[str], Chunk],
    sections: Callable[[Chunk], Sequence[SectionManifest]],
    with_neighbors: bool,
) -> ResourceBatchResult:
    from potpie_context_core.cli_commands import join_command

    def error_payload(resource_id, error):
        try:
            resource = parse_resource_id(resource_id)
            command = join_command(
                ["potpie", "resource", "list", "--doc", resource.doc, "--pot", pot_id]
            )
        except ResourceStoreError:
            command = "Use a potpie://res/<doc>/<section>/<seq>@revN id."
        if error.code == RESOURCE_READ_BUDGET_EXCEEDED:
            command = join_command(
                ["potpie", "resource", "get", resource_id, "--pot", pot_id]
            )
        detail = error.detail
        if isinstance(detail, dict):
            detail = {
                **detail,
                "candidates": [
                    {
                        **candidate,
                        "fetch_command": join_command(
                            [
                                "potpie",
                                "resource",
                                "get",
                                candidate["resource_id"],
                                "--pot",
                                pot_id,
                            ]
                        ),
                    }
                    for candidate in detail.get("candidates", ())
                ],
            }
        return {
            "resource_id": resource_id,
            "code": error.code,
            "message": str(error),
            "detail": detail,
            "recommended_next_action": command,
        }

    chunks: dict[str, Chunk] = {}
    cache: dict[str, Chunk | ResourceStoreError] = {}
    outcomes = []
    section_cache: dict[tuple[str, int], Sequence[SectionManifest]] = {}

    def fetch(resource_id):
        if resource_id not in cache:
            try:
                cache[resource_id] = read(resource_id)
            except ResourceStoreError as error:
                cache[resource_id] = error
        value = cache[resource_id]
        if isinstance(value, ResourceStoreError):
            raise value
        immutable = format_resource_id(
            value.doc, value.section, value.seq, revision=value.revision
        )
        cache.setdefault(immutable, value)
        return value, immutable

    for root in resource_ids:
        associated = []
        errors = []

        def include(resource_id):
            try:
                chunk, immutable = fetch(resource_id)
                requested = parse_resource_id(root)
                display_id = format_resource_id(
                    chunk.doc, chunk.section, chunk.seq, revision=requested.revision
                )
                chunks.setdefault(immutable, replace(chunk, resource_id=display_id))
                if immutable not in associated:
                    associated.append(immutable)
                return chunk
            except ResourceStoreError as error:
                errors.append(error_payload(resource_id, error))
                return None

        # Read root first to select exactly its immutable revision. Neighbors
        # never use the latest manifest, or a manifest cached for another revision.
        try:
            chunk, _ = fetch(root)
        except ResourceStoreError:
            include(root)
            chunk = None
        if chunk is not None:
            ids = [root]
            if with_neighbors:
                try:
                    key = (chunk.doc, chunk.revision)
                    if key not in section_cache:
                        section_cache[key] = sections(chunk)
                    seqs = sorted(
                        ref.seq
                        for section in section_cache[key]
                        if section.slug == chunk.section
                        for ref in section.chunks
                    )
                    position = seqs.index(chunk.seq)
                    ids = [
                        format_resource_id(
                            chunk.doc, chunk.section, seq, revision=chunk.revision
                        )
                        for seq in seqs[max(0, position - 1) : position + 2]
                    ]
                except ResourceStoreError as error:
                    errors.append(error_payload(root, error))
            for resource_id in ids:
                include(resource_id)
        outcomes.append(
            ResourceReadOutcome(
                resource_id=root,
                status="partial"
                if errors and associated
                else "error"
                if errors
                else "success",
                chunk_ids=tuple(associated),
                errors=tuple(errors),
            )
        )
    failed = any(outcome.errors for outcome in outcomes)
    status = "partial" if failed and chunks else "error" if failed else "success"
    # Preserve old unversioned body ids on all-success reads. Partial receipts
    # use immutable ids so associations point directly to the returned bodies.
    bodies = tuple(
        replace(chunk, resource_id=key) if failed else chunk
        for key, chunk in chunks.items()
    )
    return ResourceBatchResult(chunks=bodies, outcomes=tuple(outcomes), status=status)

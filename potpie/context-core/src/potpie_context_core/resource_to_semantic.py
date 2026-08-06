"""Turn a stored document manifest into semantic mutations (resources P4).

The resource store holds a document's *bytes*; this module produces the
mutation that puts its *structure* in the graph — a ``Document`` entity owning
one ``DocumentSection`` per division, joined by ``SECTION_OF``.

Like ``record_to_semantic``, this deliberately stops at the public semantic
write vocabulary. Import does not touch the backend or lower to a
``MutationBatch`` itself: it emits ops and hands them to ``GraphService.mutate``
so ``apply`` stays the single write door, and an imported document is validated,
risk-classified, and embedded by exactly the code path an agent's own write
uses.

**R1 lives here.** Nothing in this module reads chunk text — the manifest it
consumes carries slugs, titles, summaries, hashes, and chunk *ids*, so there is
no payload available to leak into a node property even by accident. Chunk ids
travel as claim evidence, which is what makes a section's search hit already
carry everything ``resource get`` needs (R13).
"""

from __future__ import annotations

from dataclasses import dataclass

from potpie_context_core.ports.resource_store import (
    DocumentManifest,
    SectionManifest,
    format_resource_id,
)
from potpie_context_core.semantic_mutations import (
    MutationActor,
    SemanticMutation,
    SemanticMutationRequest,
    SemanticMutationResult,
)

# An import is a deliberate write the user asked for by running the command, so
# the request is pre-approved the way ``context_record`` is: re-import emits
# medium-risk retractions for sections that disappeared, and dead-ending those
# on an approval prompt would leave the graph describing a document that no
# longer exists. ``resource rm`` is the same shape — a user-asked teardown whose
# retractions must apply or the graph keeps describing deleted chunks.
_APPROVED_BY = "resource_import"
_APPROVED_BY_DELETE = "resource_rm"
_SURFACE = "cli"

DOCUMENT_TYPE = "Document"
SECTION_TYPE = "DocumentSection"
SECTION_PREDICATE = "SECTION_OF"
# The scope edge: what the document (or one of its sections) is reference
# material *for*. Import never writes one — only an agent knows what a document
# covers — so import's job is to notice when none exists.
SCOPE_PREDICATE = "DOCUMENTS"
RESOURCE_SUBGRAPH = "knowledge"

# The structure is read off the source document itself, not judged — a
# source observation, which is also the truth class that requires evidence.
# Every section carries its chunk ids as evidence, so that requirement is met
# by construction.
_STRUCTURE_TRUTH = "source_observation"
# A section with no chunks has nothing to cite. It is a degenerate case the
# store allows, so name it honestly rather than asserting a fact with no
# evidence behind it.
_UNEVIDENCED_TRUTH = "agent_claim"


@dataclass(frozen=True, slots=True)
class ResourceImportResult:
    """What ``resource import`` did, in both halves of the split.

    ``manifest`` is the store's report (bytes written, sections added/kept/
    removed). ``graph`` is the outcome of the structure write, and is ``None``
    only when the host was built without a graph service — the CLI reports that
    as a document stored but not findable, rather than implying success.
    """

    manifest: DocumentManifest
    graph: SemanticMutationResult | None = None
    missing_claim_keys: tuple[str, ...] = ()
    """Claims the write reported as applied that a read cannot see.

    Non-empty means the mutation succeeded on paper and the document is still
    not findable — which is the exact failure this whole report block exists to
    surface, and one ``status == "applied"`` alone cannot detect.
    """
    scope_claim_count: int | None = None
    """Live ``DOCUMENTS`` claims on this document or any of its sections.

    ``None`` means nobody looked (no claim query wired), which is different
    from zero and must not be reported as "unlinked". Zero is the state the
    import nudge exists for: chunks and structure landed, but nothing says what
    the document is *about*, so it is findable by semantic luck alone.
    """

    @property
    def graph_written(self) -> bool:
        """True only when the structure both applied *and* reads back."""
        return (
            self.graph is not None
            and self.graph.ok
            and self.graph.status == "applied"
            and not self.missing_claim_keys
        )


@dataclass(frozen=True, slots=True)
class ResourceDeleteResult:
    """What ``resource rm`` did, in both halves of the split.

    ``removed`` is the store's answer; ``graph`` is the retraction's own result
    and is ``None`` when there was nothing to retract or no graph was wired.
    Kept separate on purpose: the CLI used to derive its ``graph_retracted``
    field from ``removed``, which reported a successful retraction on a write
    that never happened.
    """

    removed: bool
    graph: SemanticMutationResult | None = None

    @property
    def graph_retracted(self) -> bool:
        """True only when a retraction was applied — the honest version."""
        return (
            self.graph is not None and self.graph.ok and self.graph.status == "applied"
        )


@dataclass(frozen=True, slots=True)
class DependentClaim:
    """A live claim, written by somebody else, about a section that is leaving.

    ``SECTION_OF`` is the store's own structure and this module knows how to
    retract it. Everything *else* pointing at a section — the ``DOCUMENTS``
    links the skills teach agents to write, or anything citing the section as
    evidence — was written by an agent through the normal write door, so the
    only way to find it is to ask the graph. The caller does that lookup and
    passes the answers in, keeping this module free of a backend dependency.
    """

    predicate: str
    subject_key: str
    object_key: str
    subgraph: str = RESOURCE_SUBGRAPH


def document_key(doc: str) -> str:
    return f"document:{doc}"


def section_key(doc: str, section: str) -> str:
    return f"docsection:{doc}:{section}"


def _dependent_retractions(
    claims: tuple[DependentClaim, ...], *, reason: str
) -> list[dict]:
    """Retract claims left dangling by a section's removal.

    Without this a departed section keeps answering: its ``SECTION_OF`` edge is
    gone, but an agent-written ``DOCUMENTS`` claim stays live, keeps ranking in
    ``search --include docs``, and keeps handing out chunk ids that now resolve
    to ``resource_not_found``. Structure cleanup that only cleans up the
    structure *this module* wrote is not cleanup.
    """
    return [
        {
            "op": "retract_claim",
            "subgraph": claim.subgraph,
            "predicate": claim.predicate,
            # Keys only: these entities already exist with their real types, and
            # asserting a type here would just be this module guessing at one.
            "subject": {"key": claim.subject_key},
            "object": {"key": claim.object_key},
            "reason": reason,
        }
        for claim in claims
    ]


def resource_import_to_semantic_request(
    manifest: DocumentManifest,
    *,
    dependent_claims: tuple[DependentClaim, ...] = (),
) -> SemanticMutationRequest:
    """Build the graph write for one completed import.

    One ``upsert_entity`` for the document (carrying the revision this import
    produced), one ``assert_claim``/``SECTION_OF`` per section, and one
    ``retract_claim`` per section the prior revision had and this one does not —
    that last group is what stops a re-import from leaving the graph pointing at
    sections whose chunks were just deleted.

    ``dependent_claims`` are the *other* live claims about those departing
    sections, which the caller looked up; they are retracted in the same write
    so a re-import can never leave a live claim citing a deleted chunk.
    """
    document = {
        "key": document_key(manifest.doc),
        "type": DOCUMENT_TYPE,
        "name": manifest.doc,
        "description": _document_description(manifest),
        "properties": _drop_empty(
            {
                "revision": manifest.revision,
                "source_ref": manifest.source_ref,
                "source_kind": manifest.source_kind,
                "section_count": len(manifest.sections),
            }
        ),
    }

    ops: list[dict] = [
        {
            "op": "upsert_entity",
            "subgraph": RESOURCE_SUBGRAPH,
            "subject": document,
            "description": document["description"],
        }
    ]

    # Object refs are keys only. Repeating the document's properties on every
    # section op would re-upsert the same node N times with the same values.
    document_ref = {"key": document["key"], "type": DOCUMENT_TYPE}

    for section in manifest.sections:
        evidence = [
            {"source_ref": format_resource_id(manifest.doc, section.slug, ref.seq)}
            for ref in section.chunks
        ]
        ops.append(
            {
                "op": "assert_claim",
                "subgraph": RESOURCE_SUBGRAPH,
                "predicate": SECTION_PREDICATE,
                "truth": _STRUCTURE_TRUTH if evidence else _UNEVIDENCED_TRUTH,
                "subject": {
                    "key": section_key(manifest.doc, section.slug),
                    "type": SECTION_TYPE,
                    "name": section.title or section.slug,
                    # The retrieval card: a section is found by its summary and
                    # nothing else, so the summary *is* the description.
                    "description": _section_description(manifest, section),
                    "summary": section.summary or None,
                    "properties": _drop_empty(
                        {
                            "title": section.title,
                            "ordinal": section.ordinal,
                            "chunk_count": len(section.chunks),
                            "content_hash": section.content_hash,
                            "revision": manifest.revision,
                            "summary_pending": section.summary_pending,
                        }
                    ),
                },
                "object": document_ref,
                "evidence": evidence,
                "description": _section_description(manifest, section),
            }
        )

    for slug in manifest.sections_removed:
        ops.append(
            {
                "op": "retract_claim",
                "subgraph": RESOURCE_SUBGRAPH,
                "predicate": SECTION_PREDICATE,
                "subject": {
                    "key": section_key(manifest.doc, slug),
                    "type": SECTION_TYPE,
                },
                "object": document_ref,
                "reason": (
                    f"section '{slug}' is not in revision {manifest.revision} of "
                    f"document '{manifest.doc}'; its chunks were deleted"
                ),
            }
        )

    ops.extend(
        _dependent_retractions(
            dependent_claims,
            reason=(
                f"the section it describes left revision {manifest.revision} of "
                f"document '{manifest.doc}'; its chunks were deleted"
            ),
        )
    )

    return SemanticMutationRequest(
        pot_id=manifest.pot_id,
        operations=tuple(SemanticMutation.parse(op) for op in ops),
        # Same document, same revision, same write: a retried import must not
        # double-apply.
        idempotency_key=f"resource-import:{manifest.doc}:{manifest.revision}",
        created_by=MutationActor(surface=_SURFACE, harness="resource_import"),
        allow_review_required=True,
        approved_by=_APPROVED_BY,
    )


def resource_delete_to_semantic_request(
    *,
    pot_id: str,
    doc: str,
    section_slugs: tuple[str, ...],
    dependent_claims: tuple[DependentClaim, ...] = (),
) -> SemanticMutationRequest:
    """Build the graph write that retracts every section of a removed document.

    ``resource rm`` deletes bytes from the store; this is the matching structure
    cleanup so search cannot land on a section whose chunks are gone. There is
    no ``delete_entity`` in the semantic vocabulary — retracting each
    ``SECTION_OF`` claim is what stops the document from answering reads. The
    Document node may linger as an orphan until a pot reset; that is cheaper
    than inventing a hard-delete op just for this path.

    ``dependent_claims`` carries whatever else was live about those sections
    (see :class:`DependentClaim`). Retracting only ``SECTION_OF`` would leave an
    agent's ``DOCUMENTS`` claim answering searches with chunk ids that no longer
    resolve — a deleted document still returning citations.
    """
    document_ref = {"key": document_key(doc), "type": DOCUMENT_TYPE}
    ops = [
        {
            "op": "retract_claim",
            "subgraph": RESOURCE_SUBGRAPH,
            "predicate": SECTION_PREDICATE,
            "subject": {
                "key": section_key(doc, slug),
                "type": SECTION_TYPE,
            },
            "object": document_ref,
            "reason": (
                f"document '{doc}' was removed; section '{slug}' chunks were deleted"
            ),
        }
        for slug in section_slugs
    ]
    ops.extend(
        _dependent_retractions(
            dependent_claims,
            reason=(
                f"document '{doc}' was removed; the section it describes no "
                "longer exists and its chunks were deleted"
            ),
        )
    )
    return SemanticMutationRequest(
        pot_id=pot_id,
        operations=tuple(SemanticMutation.parse(op) for op in ops),
        idempotency_key=f"resource-delete:{doc}:{','.join(section_slugs)}",
        created_by=MutationActor(surface=_SURFACE, harness="resource_rm"),
        allow_review_required=True,
        approved_by=_APPROVED_BY_DELETE,
    )


def _document_description(manifest: DocumentManifest) -> str:
    """The document's retrieval card: what it is, plus its section titles.

    Section titles come free from the source's own structure and carry real
    signal, so a document whose summaries are still pending is not completely
    invisible to search.
    """
    parts = [f"Document '{manifest.doc}'"]
    if manifest.source_kind:
        parts.append(f"({manifest.source_kind})")
    titles = [row.title for row in manifest.sections if row.title]
    if titles:
        parts.append("— sections: " + ", ".join(titles))
    elif manifest.sections:
        parts.append(f"— {len(manifest.sections)} section(s)")
    return " ".join(parts)


def _section_description(manifest: DocumentManifest, section: SectionManifest) -> str:
    summary = (section.summary or "").strip()
    heading = " · ".join(p for p in (manifest.doc, section.title or section.slug) if p)
    if summary:
        return f"{heading} — {summary}"
    # Nothing to index but the heading. The claim is still written, so the
    # section is reachable structurally and a later pass can fill the summary.
    return heading


def _drop_empty(values: dict) -> dict:
    return {k: v for k, v in values.items() if v not in (None, "", [], {})}


__all__ = [
    "DOCUMENT_TYPE",
    "RESOURCE_SUBGRAPH",
    "SECTION_PREDICATE",
    "SECTION_TYPE",
    "DependentClaim",
    "ResourceDeleteResult",
    "ResourceImportResult",
    "document_key",
    "resource_delete_to_semantic_request",
    "resource_import_to_semantic_request",
    "section_key",
]

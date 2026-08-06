"""The import → semantic-mutation mapping (resources P4).

These assert the shape the graph actually receives, and — through the real
validator — that ``resource import`` cannot emit a batch an agent's own write
would be rejected for.
"""

from __future__ import annotations

import pytest

from potpie_context_core.ports.resource_store import (
    ChunkRef,
    DocumentManifest,
    SectionManifest,
)
from potpie_context_core.resource_to_semantic import (
    resource_delete_to_semantic_request,
    resource_import_to_semantic_request,
)
from potpie_context_core.semantic_mutation_validator import validate_semantic_request

pytestmark = pytest.mark.unit


def _section(slug="capacity", *, summary="how much headroom Q3 leaves", chunks=2):
    return SectionManifest(
        slug=slug,
        title=slug.replace("-", " ").title(),
        summary=summary,
        ordinal=0,
        content_hash=f"{slug}-hash",
        chunks=tuple(ChunkRef(seq=seq, label=f"part {seq}") for seq in range(chunks)),
        summary_pending=not summary,
    )


def _manifest(*, sections=None, removed=(), revision=1):
    return DocumentManifest(
        pot_id="pot-1",
        doc="q3-review",
        revision=revision,
        source_ref="file:///q3.pdf",
        source_kind="pdf",
        sections=tuple(sections if sections is not None else [_section()]),
        sections_removed=tuple(removed),
    )


def test_emits_a_document_upsert_and_one_claim_per_section():
    request = resource_import_to_semantic_request(
        _manifest(sections=[_section("capacity"), _section("limits")])
    )

    assert [op.op for op in request.operations] == [
        "upsert_entity",
        "assert_claim",
        "assert_claim",
    ]
    document = request.operations[0].subject
    assert document.key == "document:q3-review"
    assert document.properties["revision"] == 1
    assert document.properties["source_ref"] == "file:///q3.pdf"
    for op in request.operations[1:]:
        assert op.predicate == "SECTION_OF"
        assert op.object.key == "document:q3-review"
        assert op.subject.key.startswith("docsection:q3-review:")


def test_section_claims_cite_their_chunk_ids():
    request = resource_import_to_semantic_request(_manifest())

    claim = request.operations[1]
    assert [e.source_ref for e in claim.evidence] == [
        "potpie://res/q3-review/capacity/0000",
        "potpie://res/q3-review/capacity/0001",
    ]


def test_the_summary_is_the_section_description():
    """The description is the retrieval card, and a section has no other index."""
    request = resource_import_to_semantic_request(_manifest())

    assert "how much headroom Q3 leaves" in request.operations[1].description


def test_a_pending_summary_still_indexes_on_the_title():
    request = resource_import_to_semantic_request(
        _manifest(sections=[_section("limits", summary="")])
    )

    claim = request.operations[1]
    assert claim.description == "q3-review · Limits"
    assert claim.subject.properties["summary_pending"] is True


def test_a_section_with_no_chunks_does_not_claim_to_be_evidenced():
    """``source_observation`` requires evidence; a chunkless section has none,
    so it downgrades rather than asserting a fact nothing backs."""
    request = resource_import_to_semantic_request(
        _manifest(sections=[_section("empty", chunks=0)])
    )

    assert request.operations[1].truth == "agent_claim"
    assert request.operations[1].evidence == ()


def test_removed_sections_are_retracted():
    request = resource_import_to_semantic_request(
        _manifest(removed=("appendix",), revision=2)
    )

    retraction = request.operations[-1]
    assert retraction.op == "retract_claim"
    assert retraction.subject.key == "docsection:q3-review:appendix"
    assert retraction.object.key == "document:q3-review"
    assert "revision 2" in retraction.reason


def test_the_batch_passes_the_real_validator_with_no_errors():
    plan = validate_semantic_request(
        resource_import_to_semantic_request(
            _manifest(
                sections=[_section("capacity")], removed=("appendix",), revision=2
            )
        )
    )

    assert plan.ok, [issue.message for issue in plan.errors]
    assert plan.decision == "apply"


def test_idempotency_key_is_the_document_revision():
    """Re-running an import that already applied must not double-write."""
    request = resource_import_to_semantic_request(_manifest(revision=3))

    assert request.idempotency_key == "resource-import:q3-review:3"


def test_delete_retracts_every_named_section():
    request = resource_delete_to_semantic_request(
        pot_id="pot-1",
        doc="q3-review",
        section_slugs=("capacity", "limits"),
    )

    assert [op.op for op in request.operations] == ["retract_claim", "retract_claim"]
    assert [op.subject.key for op in request.operations] == [
        "docsection:q3-review:capacity",
        "docsection:q3-review:limits",
    ]
    assert all(op.object.key == "document:q3-review" for op in request.operations)
    assert request.approved_by == "resource_rm"
    plan = validate_semantic_request(request)
    assert plan.ok, [issue.message for issue in plan.errors]

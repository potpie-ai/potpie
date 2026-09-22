from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

from potpie_context_core.ports.claim_query import ClaimRow
from potpie_context_core.semantic_mutation_validator import validate_semantic_request
from potpie_context_core.semantic_mutations import SemanticMutationRequest
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.application.services.resource_facade import (
    ResourceFacade,
    _claim_snapshot,
)

pytestmark = pytest.mark.unit

POT = "snapshot-pot"


def _row() -> ClaimRow:
    return ClaimRow(
        pot_id=POT,
        claim_key="claim:guide-api",
        predicate="DOCUMENTS",
        subject_key="document:guide",
        object_key="service:api",
        fact="Guide documents API behavior",
        description="Stable API guidance",
        source_system="resource",
        source_ref="resource:guide/overview/1@rev1",
        source_refs=("resource:guide/overview/1@rev1",),
        evidence=(
            {
                "source_ref": "resource:guide/overview/1@rev1",
                "authority": "external_system",
            },
        ),
        properties={"audience": "operators"},
        valid_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        observed_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
        truth="source_observation",
        confidence=0.9,
    )


def _backend(row: ClaimRow) -> InMemoryGraphBackend:
    backend = InMemoryGraphBackend()
    backend.store.add(row)
    backend.store.set_entity_label(
        pot_id=POT, entity_key=row.subject_key, labels=("Entity", "Document")
    )
    backend.store.set_entity_label(
        pot_id=POT, entity_key=row.object_key, labels=("Entity", "Service")
    )
    return backend


class _UnusedStore:
    pass


class _RacingGraph:
    def __init__(self, backend: InMemoryGraphBackend) -> None:
        self.backend = backend
        self.delegate = DefaultGraphService(backend=backend)

    def mutate(self, request):
        current = self.backend.store.rows[0]
        self.backend.store.rows[0] = dataclasses.replace(
            current,
            fact="Newer API behavior",
            description="Updated after the marker row was copied",
        )
        return self.delegate.mutate(request)


class _CapturingGraph:
    def __init__(self) -> None:
        self.requests = []

    def mutate(self, request):
        self.requests.append(request)
        return type("Result", (), {"ok": True})()


def test_intervening_content_update_rejects_marker_and_keeps_newer_fact() -> None:
    copied = _row()
    backend = _backend(copied)
    facade = ResourceFacade(
        store=_UnusedStore(), graph=_RacingGraph(backend), claims=backend.claim_query
    )

    errors = facade._mark_claims_for_review(
        pot_id=POT,
        rows=[copied],
        refs=("resource:guide/overview/1@rev1",),
        reason="source revision changed",
    )

    assert errors
    assert "changed before evidence review metadata" in errors[0]
    assert backend.store.rows[0].fact == "Newer API behavior"
    assert backend.store.rows[0].properties == {"audience": "operators"}


def test_duplicate_exact_claim_key_rejects_preservation_marker() -> None:
    row = _row()
    backend = _backend(row)
    backend.store.add(dataclasses.replace(row, fact="duplicate row"))
    facade = ResourceFacade(
        store=_UnusedStore(),
        graph=DefaultGraphService(backend=backend),
        claims=backend.claim_query,
    )

    errors = facade._mark_claims_for_review(
        pot_id=POT,
        rows=[row],
        refs=("resource:guide/overview/1@rev1",),
        reason="source revision changed",
    )

    assert errors
    assert "changed before evidence review metadata" in errors[0]
    assert len(backend.store.rows) == 2


@pytest.mark.parametrize(
    ("preserve_key", "subject_key", "object_key"),
    [
        ("claim:other", "document:guide", "service:api"),
        ("claim:guide-api", "document:other", "service:api"),
        ("claim:guide-api", "document:guide", "service:other"),
    ],
)
def test_preservation_guard_rejects_malicious_identity_mismatch(
    preserve_key: str, subject_key: str, object_key: str
) -> None:
    row = _row()
    backend = _backend(row)
    request = SemanticMutationRequest.parse(
        {
            "operations": [
                {
                    "op": "assert_claim",
                    "subgraph": "knowledge",
                    "predicate": row.predicate,
                    "subject": {"key": subject_key, "type": "Document"},
                    "object": {"key": object_key, "type": "Service"},
                    "truth": row.truth,
                    "description": row.description,
                    "evidence": [{"source_ref": row.source_refs[0]}],
                    "extra": {
                        "evidence_review_required": True,
                        "_preserve_claim_key": preserve_key,
                        "_expected_claim_snapshot": _claim_snapshot(row),
                    },
                }
            ]
        },
        pot_id=POT,
        allow_review_required=True,
        approved_by="resource_evidence_lifecycle",
    )

    plan = validate_semantic_request(request, claim_query=backend.claim_query)

    assert plan.decision == "rejected"
    assert any(issue.code == "claim_preservation_conflict" for issue in plan.issues)


def test_unchanged_exact_marker_succeeds_and_preserves_claim() -> None:
    row = _row()
    backend = _backend(row)
    facade = ResourceFacade(
        store=_UnusedStore(),
        graph=DefaultGraphService(backend=backend),
        claims=backend.claim_query,
    )

    errors = facade._mark_claims_for_review(
        pot_id=POT,
        rows=[row],
        refs=("resource:guide/overview/1@rev1",),
        reason="source revision changed",
    )

    assert errors == ()
    assert len(backend.store.rows) == 1
    marked = backend.store.rows[0]
    assert marked.claim_key == row.claim_key
    assert marked.fact == row.fact
    assert marked.subject_key == row.subject_key
    assert marked.object_key == row.object_key
    assert marked.properties["evidence_review_required"] is True


def test_marker_idempotency_key_tracks_exact_claim_snapshot() -> None:
    row = _row()
    backend = _backend(row)
    graph = _CapturingGraph()
    facade = ResourceFacade(store=_UnusedStore(), graph=graph, claims=backend.claim_query)
    args = {
        "pot_id": POT,
        "refs": ("resource:guide/overview/1@rev1",),
        "reason": "source revision changed",
    }

    assert facade._mark_claims_for_review(rows=[row], **args) == ()
    assert facade._mark_claims_for_review(rows=[row], **args) == ()
    changed = dataclasses.replace(row, fact="new fact", properties={"generation": 2})
    assert facade._mark_claims_for_review(rows=[changed], **args) == ()

    first, repeated, updated = (request.idempotency_key for request in graph.requests)
    assert first == repeated
    assert updated != first

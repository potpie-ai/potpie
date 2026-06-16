"""Tests for the public semantic-mutation DTOs (Graph V1.5 Step 3).

The DTO layer must parse JSON into typed objects without importing any adapter
or backend, and must normalize the single-op alias into a batch.
"""

from __future__ import annotations

import pytest

from domain.semantic_mutations import (
    GraphEntityRef,
    GraphEvidenceRef,
    SemanticMutationParseError,
    SemanticMutationRequest,
)

pytestmark = pytest.mark.unit


def _batch_payload() -> dict:
    return {
        "graph_contract_version": "v1.5",
        "pot_id": "local/default",
        "idempotency_key": "mutation:preference:pytest-fixtures",
        "created_by": {"surface": "cli", "harness": "codex"},
        "operations": [
            {
                "op": "link_entities",
                "subgraph": "infra_topology",
                "subject": {
                    "key": "service:local:payments-api",
                    "type": "Service",
                    "summary": "Payments API service.",
                    "properties": {"name": "payments-api"},
                    "description": "payments service that processes refunds",
                },
                "predicate": "DEPENDS_ON",
                "object": {
                    "key": "service:local:ledger-api",
                    "type": "Service",
                    "properties": {"name": "ledger-api"},
                },
                "truth": "source_observation",
                "confidence": 1.0,
                "evidence": [
                    {
                        "source_ref": "repo:manifest:services/payments/service.yaml",
                        "authority": "repository_metadata",
                    }
                ],
                "valid_from": "2026-06-08T00:00:00+05:30",
            }
        ],
    }


def test_parse_batch_payload() -> None:
    req = SemanticMutationRequest.parse(_batch_payload())
    assert req.pot_id == "local/default"
    assert req.graph_contract_version == "v1.5"
    assert req.created_by.surface == "cli"
    assert req.created_by.harness == "codex"
    assert len(req.operations) == 1
    op = req.operations[0]
    assert op.op == "link_entities"
    assert op.subgraph == "infra_topology"
    assert op.subject.key == "service:local:payments-api"
    assert op.subject.type == "Service"
    assert op.subject.summary == "Payments API service."
    assert op.subject.description == "payments service that processes refunds"
    assert op.predicate == "DEPENDS_ON"
    assert op.object.key == "service:local:ledger-api"
    assert op.truth == "source_observation"
    assert op.confidence == 1.0
    assert op.evidence[0].source_ref == "repo:manifest:services/payments/service.yaml"
    assert op.evidence[0].authority == "repository_metadata"
    assert op.valid_from == "2026-06-08T00:00:00+05:30"


def test_single_op_alias_normalizes_to_batch() -> None:
    payload = {
        "operation": "assert_claim",
        "pot_id": "p",
        "subgraph": "decisions",
        "subject": {"key": "preference:wrap-retries", "type": "Preference"},
        "predicate": "POLICY_APPLIES_TO",
        "object": {"key": "service:payments", "type": "Service"},
        "truth": "preference",
        "description": "wrap external calls in tenacity retry",
    }
    req = SemanticMutationRequest.parse(payload)
    assert len(req.operations) == 1
    assert req.operations[0].op == "assert_claim"
    assert req.operations[0].subject.key == "preference:wrap-retries"
    # Request-level keys must not leak into the op.
    assert "pot_id" not in req.operations[0].raw or req.operations[0].raw.get("op")


def test_pot_id_override_wins() -> None:
    payload = _batch_payload()
    req = SemanticMutationRequest.parse(payload, pot_id="override/pot")
    assert req.pot_id == "override/pot"


def test_missing_pot_id_raises() -> None:
    payload = _batch_payload()
    del payload["pot_id"]
    with pytest.raises(SemanticMutationParseError):
        SemanticMutationRequest.parse(payload)


def test_empty_operations_raises() -> None:
    with pytest.raises(SemanticMutationParseError):
        SemanticMutationRequest.parse({"pot_id": "p", "operations": []})


def test_no_operations_or_operation_raises() -> None:
    with pytest.raises(SemanticMutationParseError):
        SemanticMutationRequest.parse({"pot_id": "p"})


def test_entity_ref_requires_key() -> None:
    with pytest.raises(SemanticMutationParseError):
        GraphEntityRef.parse({"type": "Service"})


def test_entity_ref_from_string() -> None:
    ref = GraphEntityRef.parse("service:foo")
    assert ref.key == "service:foo"


def test_evidence_requires_source_ref() -> None:
    with pytest.raises(SemanticMutationParseError):
        GraphEvidenceRef.parse({"authority": "repository_metadata"})


def test_evidence_from_string() -> None:
    ev = GraphEvidenceRef.parse("repo:x")
    assert ev.source_ref == "repo:x"


def test_confidence_must_be_number() -> None:
    payload = {
        "pot_id": "p",
        "operations": [{"op": "assert_claim", "confidence": "high"}],
    }
    with pytest.raises(SemanticMutationParseError):
        SemanticMutationRequest.parse(payload)


def test_append_event_fields_parse() -> None:
    payload = {
        "pot_id": "p",
        "operation": "append_event",
        "subgraph": "recent_changes",
        "verb": "merged_pr",
        "occurred_at": "2026-06-08T00:00:00Z",
        "actor": {"key": "person:alice", "type": "Person"},
        "targets": [{"key": "service:payments", "type": "Service"}],
        "mentions": [{"key": "bug_pattern:refund-race", "type": "BugPattern"}],
        "description": "merged PR #42 fixing the refund race",
    }
    req = SemanticMutationRequest.parse(payload)
    op = req.operations[0]
    assert op.verb == "merged_pr"
    assert op.actor.key == "person:alice"
    assert op.targets[0].key == "service:payments"
    assert op.mentions[0].key == "bug_pattern:refund-race"


def test_patch_and_transition_fields_parse() -> None:
    payload = {
        "pot_id": "p",
        "operations": [
            {
                "op": "patch_entity",
                "subject": {"key": "service:payments", "type": "Service"},
                "patch": {"summary": "Payments service"},
                "expected_entity_version": "entity-version:1",
                "reason": "tighten display metadata",
            },
            {
                "op": "transition_state",
                "subject": {"key": "decision:adr-1", "type": "Decision"},
                "from_state": "proposed",
                "to_state": "accepted",
                "expected_version": "entity-version:2",
                "reason": "ADR approved",
            },
        ],
    }

    req = SemanticMutationRequest.parse(payload)

    patch = req.operations[0]
    assert patch.patch == {"summary": "Payments service"}
    assert patch.expected_entity_version == "entity-version:1"
    transition = req.operations[1]
    assert transition.from_state == "proposed"
    assert transition.to_state == "accepted"
    assert transition.expected_entity_version == "entity-version:2"


def test_supersede_and_merge_fields_parse() -> None:
    payload = {
        "pot_id": "p",
        "operations": [
            {
                "op": "supersede_claim",
                "subject": {"key": "service:payments", "type": "Service"},
                "predicate": "DEPENDS_ON",
                "object": {"key": "service:ledger-old", "type": "Service"},
                "superseded_by": {"key": "service:ledger-new", "type": "Service"},
                "reason": "dependency target changed",
            },
            {
                "op": "merge_duplicate_entities",
                "subject": {"key": "service:payments-api-v1", "type": "Service"},
                "object": {"key": "service:payments-api", "type": "Service"},
                "external_ids": {
                    "losing": {"github_service": "payments-api-v1"},
                    "winning": {"github_service": "payments-api"},
                },
                "reason": "same service in source manifests",
            },
        ],
    }

    req = SemanticMutationRequest.parse(payload)

    supersede = req.operations[0]
    assert supersede.superseded_by.key == "service:ledger-new"
    merge = req.operations[1]
    assert merge.external_ids["losing"]["github_service"] == "payments-api-v1"


def test_patch_field_must_be_object() -> None:
    with pytest.raises(SemanticMutationParseError, match="patch"):
        SemanticMutationRequest.parse(
            {
                "pot_id": "p",
                "operation": "patch_entity",
                "subject": {"key": "service:payments", "type": "Service"},
                "patch": ["summary", "Payments service"],
            }
        )


def test_external_ids_field_must_be_object() -> None:
    with pytest.raises(SemanticMutationParseError, match="external_ids"):
        SemanticMutationRequest.parse(
            {
                "pot_id": "p",
                "operation": "merge_duplicate_entities",
                "subject": {"key": "service:payments-v1", "type": "Service"},
                "object": {"key": "service:payments", "type": "Service"},
                "external_ids": ["service:payments-v1"],
            }
        )

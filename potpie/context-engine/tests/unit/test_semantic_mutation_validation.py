"""Tests for the semantic validator + risk policy + lowerer (V1.5 Steps 4, 5)."""

from __future__ import annotations

import pytest

from application.services.semantic_mutation_lowering import lower_semantic_request
from application.services.semantic_mutation_validator import (
    subgraph_for_predicate,
    validate_semantic_request,
)
from domain.graph_contract import GRAPH_CONTRACT_VERSION, ONTOLOGY_VERSION
from domain.semantic_mutations import SemanticMutationRequest

pytestmark = pytest.mark.unit


def _req(op: dict, **kw) -> SemanticMutationRequest:
    return SemanticMutationRequest.parse({"pot_id": "p", "operations": [op]}, **kw)


def _link(**over) -> dict:
    base = {
        "op": "link_entities",
        "subgraph": "infra_topology",
        "subject": {"key": "service:payments-api", "type": "Service"},
        "predicate": "DEPENDS_ON",
        "object": {"key": "service:ledger-api", "type": "Service"},
        "truth": "source_observation",
        "evidence": [{"source_ref": "repo:manifest", "authority": "repository_metadata"}],
        "description": "payments calls ledger to post entries",
    }
    base.update(over)
    return base


# --- validation -------------------------------------------------------------


def test_valid_low_risk_link_is_accepted_and_apply() -> None:
    plan = validate_semantic_request(_req(_link()))
    assert plan.ok
    assert plan.decision == "apply"
    assert plan.risk == "low"
    assert len(plan.accepted_ops) == 1


def test_unsupported_contract_version_rejected() -> None:
    req = SemanticMutationRequest.parse(
        {"pot_id": "p", "graph_contract_version": "v9", "operations": [_link()]}
    )
    plan = validate_semantic_request(req)
    assert not plan.ok
    assert plan.decision == "rejected"
    assert any(i.code == "unsupported_contract_version" for i in plan.errors)


def test_unknown_predicate_rejected() -> None:
    plan = validate_semantic_request(_req(_link(predicate="FROBNICATES")))
    assert not plan.ok
    assert any(i.code == "unknown_predicate" for i in plan.errors)


def test_invalid_endpoints_rejected() -> None:
    # DEPENDS_ON is Service->Service; Service->Repository is invalid.
    plan = validate_semantic_request(
        _req(_link(object={"key": "repo:foo", "type": "Repository"}))
    )
    assert not plan.ok
    assert any(i.code == "invalid_endpoints" for i in plan.errors)


def test_key_prefix_mismatch_rejected() -> None:
    plan = validate_semantic_request(
        _req(_link(subject={"key": "repo:wrong", "type": "Service"}))
    )
    assert any(i.code == "key_prefix_mismatch" for i in plan.errors)


def test_durable_claim_without_evidence_rejected() -> None:
    # authoritative_fact is not low-authority → must carry evidence.
    plan = validate_semantic_request(
        _req(_link(truth="authoritative_fact", evidence=[]))
    )
    assert any(i.code == "missing_evidence" for i in plan.errors)


def test_agent_claim_without_evidence_allowed() -> None:
    # A low-authority agent_claim needs no evidence.
    plan = validate_semantic_request(_req(_link(truth="agent_claim", evidence=[])))
    assert plan.ok
    assert plan.decision == "apply"


def test_code_asset_endpoint_is_valid_for_preferences() -> None:
    op = {
        "op": "assert_claim",
        "subgraph": "preferences",
        "subject": {
            "key": "preference:cli-errors",
            "type": "Preference",
            "properties": {
                "policy_kind": "error_handling",
                "prescription": "CLI commands should surface validation errors with next actions.",
                "strength": "strong",
            },
        },
        "predicate": "POLICY_APPLIES_TO",
        "object": {
            "key": "code:potpie-context-engine:adapters/inbound/cli",
            "type": "CodeAsset",
        },
        "truth": "preference",
        "description": "CLI graph commands should return structured errors with next actions.",
        "extra": {"file_path": "adapters/inbound/cli", "language": "python"},
    }

    plan = validate_semantic_request(_req(op))

    assert plan.ok
    assert plan.decision == "apply"


def test_missing_description_warns_not_rejects() -> None:
    plan = validate_semantic_request(_req(_link(description=None)))
    assert plan.ok  # still applies
    assert any(i.code == "missing_description" and not i.is_error for i in plan.issues)


def test_bad_confidence_rejected() -> None:
    plan = validate_semantic_request(_req(_link(confidence=1.5)))
    assert any(i.code == "bad_confidence" for i in plan.errors)


def test_bad_timestamp_rejected() -> None:
    plan = validate_semantic_request(_req(_link(valid_from="not-a-date")))
    assert any(i.code == "bad_timestamp" for i in plan.errors)


def test_deferred_op_rejected_honestly() -> None:
    plan = validate_semantic_request(
        _req({"op": "patch_entity", "subject": {"key": "service:x", "type": "Service"}})
    )
    assert not plan.ok
    assert any(i.code == "op_deferred" for i in plan.errors)
    assert plan.deferred_ops


def test_review_required_op() -> None:
    plan = validate_semantic_request(
        _req(
            {
                "op": "supersede_claim",
                "subject": {"key": "service:a", "type": "Service"},
                "predicate": "DEPENDS_ON",
                "object": {"key": "service:b", "type": "Service"},
            }
        )
    )
    assert plan.decision == "review_required"
    assert plan.review_required_ops


def test_append_event_validates_entity_refs_and_endpoints() -> None:
    plan = validate_semantic_request(
        _req(
            {
                "op": "append_event",
                "verb": "merged_pr",
                "occurred_at": "2026-06-08T00:00:00Z",
                "actor": {"key": "repo:not-person"},
                "targets": [{"key": "service:payments-api", "type": "Nope"}],
                "description": "merged PR #42",
            }
        )
    )

    assert not plan.ok
    assert plan.decision == "rejected"
    assert any(i.code == "invalid_endpoints" for i in plan.errors)
    assert any(i.code == "unknown_entity_type" for i in plan.errors)


def test_end_relation_validity_requires_exact_object() -> None:
    plan = validate_semantic_request(
        _req(
            {
                "op": "end_relation_validity",
                "subgraph": "infra_topology",
                "subject": {"key": "service:payments-api", "type": "Service"},
                "predicate": "DEPENDS_ON",
                "reason": "dependency removed",
            },
            allow_review_required=True,
            approved_by="user:alice",
        )
    )

    assert not plan.ok
    assert plan.decision == "rejected"
    assert any(i.code == "missing_object" for i in plan.errors)


def test_review_required_supersede_claim_validates_refs_first() -> None:
    plan = validate_semantic_request(
        _req(
            {
                "op": "supersede_claim",
                "subject": {"key": "repo:not-person", "type": "Person"},
                "predicate": "DEPENDS_ON",
                "object": {"key": "service:payments-api", "type": "Service"},
            }
        )
    )

    assert not plan.ok
    assert plan.decision == "rejected"
    assert not plan.review_required_ops
    assert any(i.code == "key_prefix_mismatch" for i in plan.errors)
    assert any(i.code == "invalid_endpoints" for i in plan.errors)


def test_user_decision_is_medium_and_needs_approval() -> None:
    op = {
        "op": "assert_claim",
        "subgraph": "decisions",
        "subject": {"key": "decision:use-postgres", "type": "Decision"},
        "predicate": "DECIDED",
        "object": {"key": "service:payments-api", "type": "Service"},
        "truth": "user_decision",
        "evidence": [{"source_ref": "thread:123", "authority": "user_statement"}],
        "description": "we will use postgres for payments",
    }
    # Without approval → review_required.
    plan = validate_semantic_request(_req(op))
    assert plan.risk == "medium"
    assert plan.decision == "review_required"
    # With approval → apply.
    plan2 = validate_semantic_request(
        _req(op, allow_review_required=True, approved_by="user:alice")
    )
    assert plan2.decision == "apply"


def test_subgraph_for_predicate() -> None:
    assert subgraph_for_predicate("PROVIDES") == "features"
    assert subgraph_for_predicate("IMPLEMENTED_IN") == "features"
    assert subgraph_for_predicate("POLICY_APPLIES_TO") == "preferences"
    assert subgraph_for_predicate("RESOLVED") == "bugs"
    assert subgraph_for_predicate("DEPENDS_ON") == "infra_topology"
    assert subgraph_for_predicate("DECIDED") == "decisions"


# --- lowering ---------------------------------------------------------------


def test_lowering_produces_batch_with_metadata() -> None:
    req = _req(_link())
    plan = validate_semantic_request(req)
    lower_semantic_request(req, plan)
    assert plan.batch is not None
    # link_entities → two entity upserts + one edge upsert.
    assert len(plan.batch.entity_upserts) == 2
    assert len(plan.batch.edge_upserts) == 1
    edge = plan.batch.edge_upserts[0]
    assert edge.edge_type == "DEPENDS_ON"
    props = edge.properties
    # Full V1.5 claim metadata is stamped.
    for key in (
        "claim_key",
        "subgraph",
        "truth",
        "confidence",
        "source_refs",
        "evidence",
        "valid_at",
        "observed_at",
        "created_by",
        "graph_contract_version",
        "ontology_version",
        "fact",
    ):
        assert key in props, key
    assert props["truth"] == "source_observation"
    assert props["evidence_strength"] == "deterministic"  # truth→strength map
    assert props["graph_contract_version"] == GRAPH_CONTRACT_VERSION
    assert props["ontology_version"] == ONTOLOGY_VERSION
    assert props["source_refs"] == ["repo:manifest"]
    # claim_keys surfaced on the accepted op.
    assert plan.accepted_ops[0].claim_keys
    # No EventRef fabricated for a non-event write.
    assert plan.batch.event_ref is None


def test_lowering_stamps_environment_into_identity() -> None:
    req = _req(_link(environment="prod"))
    plan = validate_semantic_request(req)
    lower_semantic_request(req, plan)
    edge = plan.batch.edge_upserts[0]
    assert edge.properties["environment"] == "prod"
    assert edge.properties["identity_key"][-1] == "prod"


def test_lowering_derives_subgraph_when_omitted() -> None:
    req = _req(
        _link(
            subgraph=None,
            predicate="USES_ADAPTER",
            object={"key": "adapter:graph-backend:embedded", "type": "Adapter"},
            environment="local",
        )
    )
    plan = validate_semantic_request(req)
    assert plan.ok
    lower_semantic_request(req, plan)

    edge = plan.batch.edge_upserts[0]
    assert edge.properties["subgraph"] == "infra_topology"
    assert ":infra_topology:" in edge.properties["claim_key"]


def test_lowering_keeps_canonical_label_for_referenced_entity() -> None:
    req = SemanticMutationRequest.parse(
        {
            "pot_id": "p",
            "operations": [
                {
                    "op": "upsert_entity",
                    "subject": {
                        "key": "repo:github.com/potpie-ai/potpie",
                        "type": "Repository",
                        "summary": "Potpie repo",
                    },
                },
                {
                    "op": "assert_claim",
                    "subject": {
                        "key": "feature:context-graph",
                        "type": "Feature",
                    },
                    "predicate": "IMPLEMENTED_IN",
                    "object": {"key": "repo:github.com/potpie-ai/potpie"},
                    "truth": "source_observation",
                    "description": "context graph is implemented in potpie",
                    "evidence": [{"source_ref": "repo:README"}],
                },
            ],
        }
    )
    plan = validate_semantic_request(req)
    assert plan.ok
    lower_semantic_request(req, plan)

    labels = {e.entity_key: e.labels for e in plan.batch.entity_upserts}
    assert labels["repo:github.com/potpie-ai/potpie"] == ("Repository",)


def test_lowering_carries_preference_fields_on_claim() -> None:
    op = {
        "op": "assert_claim",
        "subgraph": "preferences",
        "subject": {
            "key": "preference:cli-errors",
            "type": "Preference",
            "properties": {
                "policy_kind": "error_handling",
                "prescription": "Emit structured CLI errors with recommended next actions.",
                "strength": "strong",
                "audience": "path",
            },
        },
        "predicate": "POLICY_APPLIES_TO",
        "object": {"key": "repo:github.com/potpie-ai/potpie", "type": "Repository"},
        "truth": "preference",
        "description": "CLI graph commands should return structured errors with next actions.",
        "extra": {
            "repo": "github.com/potpie-ai/potpie",
            "file_path": "potpie/context-engine/adapters/inbound/cli",
            "language": "python",
        },
    }
    req = _req(op)
    plan = validate_semantic_request(req)
    assert plan.ok
    lower_semantic_request(req, plan)

    props = plan.batch.edge_upserts[0].properties
    assert props["policy_kind"] == "error_handling"
    assert props["prescription"].startswith("Emit structured CLI errors")
    assert props["strength"] == "strong"
    assert props["code_scope"]["file_path"] == "potpie/context-engine/adapters/inbound/cli"


def test_lowering_value_object_creates_observation() -> None:
    op = {
        "op": "assert_claim",
        "subgraph": "bugs",
        "subject": {"key": "bug_pattern:refund-race", "type": "BugPattern"},
        "predicate": "REPRODUCES",
        "value": "fails under concurrent settle calls",
        "truth": "agent_claim",
        "description": "refund race: deadlock on concurrent settle",
    }
    req = _req(op)
    plan = validate_semantic_request(req)
    assert plan.ok
    lower_semantic_request(req, plan)
    labels = {lbl for e in plan.batch.entity_upserts for lbl in e.labels}
    assert "Observation" in labels


def test_lowering_append_event_creates_activity_and_edges() -> None:
    op = {
        "op": "append_event",
        "subgraph": "recent_changes",
        "verb": "merged_pr",
        "occurred_at": "2026-06-08T00:00:00Z",
        "actor": {"key": "person:alice", "type": "Person"},
        "targets": [{"key": "service:payments-api", "type": "Service"}],
        "description": "merged PR #42",
    }
    req = _req(op)
    plan = validate_semantic_request(req)
    assert plan.ok
    lower_semantic_request(req, plan)
    edge_types = {e.edge_type for e in plan.batch.edge_upserts}
    assert "PERFORMED" in edge_types
    assert "TOUCHED" in edge_types
    for edge in plan.batch.edge_upserts:
        assert edge.properties["valid_at"] == "2026-06-08T00:00:00Z"
    labels = {lbl for e in plan.batch.entity_upserts for lbl in e.labels}
    assert "Activity" in labels


def test_lowering_retract_produces_invalidation() -> None:
    op = {
        "op": "retract_claim",
        "subgraph": "infra_topology",
        "subject": {"key": "service:payments-api", "type": "Service"},
        "predicate": "DEPENDS_ON",
        "object": {"key": "service:ledger-api", "type": "Service"},
        "reason": "dependency removed",
    }
    req = _req(op, allow_review_required=True, approved_by="user:alice")
    plan = validate_semantic_request(req)
    assert plan.decision == "apply"  # medium + approved
    lower_semantic_request(req, plan)
    assert len(plan.batch.invalidations) == 1
    assert plan.batch.invalidations[0].target_edge[0] == "DEPENDS_ON"

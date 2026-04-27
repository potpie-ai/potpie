"""Deterministic ontology classifier (signals → canonical labels to add)."""

from __future__ import annotations

import pytest

from domain.ontology_classifier import (
    EntitySignals,
    build_signals,
    classify_entity,
)

pytestmark = pytest.mark.unit


def _labels(*, properties=None, outgoing=(), incoming=(), labels=()) -> tuple[str, ...]:
    signals = build_signals(
        labels=labels,
        properties=dict(properties or {}),
        outgoing_edge_names=outgoing,
        incoming_edge_names=incoming,
    )
    return classify_entity(signals)


class TestBuildSignals:
    def test_normalizes_edge_names(self) -> None:
        signals = build_signals(
            labels=(),
            properties={},
            outgoing_edge_names=["deployed-to", "  member of  "],
            incoming_edge_names=["AUTHORED BY"],
        )
        assert "DEPLOYED_TO" in signals.outgoing_edge_names
        assert "MEMBER_OF" in signals.outgoing_edge_names
        assert "AUTHORED_BY" in signals.incoming_edge_names

    def test_filters_empty_edge_names(self) -> None:
        signals = build_signals(
            labels=(),
            properties={},
            outgoing_edge_names=["", None, "DEPLOYED_TO"],  # type: ignore[list-item]
            incoming_edge_names=(),
        )
        assert signals.outgoing_edge_names == frozenset({"DEPLOYED_TO"})

    def test_returns_entity_signals(self) -> None:
        out = build_signals(labels=("Entity",), properties={})
        assert isinstance(out, EntitySignals)
        assert out.labels == ("Entity",)


# --- Property signatures ---------------------------------------------------


class TestPropertyClassification:
    def test_pr_number_implies_pull_request(self) -> None:
        assert "PullRequest" in _labels(properties={"pr_number": 42})

    def test_issue_number_implies_issue(self) -> None:
        assert "Issue" in _labels(properties={"issue_number": 5})

    def test_valid_sha_implies_commit(self) -> None:
        assert "Commit" in _labels(properties={"sha": "abc1234"})  # 7 hex chars

    def test_full_sha_implies_commit(self) -> None:
        assert "Commit" in _labels(properties={"sha": "0123456789abcdef0123456789abcdef01234567"})

    def test_short_sha_does_not_imply_commit(self) -> None:
        # Fewer than 7 chars is not accepted.
        assert "Commit" not in _labels(properties={"sha": "abc12"})

    def test_non_hex_sha_does_not_imply_commit(self) -> None:
        assert "Commit" not in _labels(properties={"sha": "ZZZZZZZ"})

    def test_github_login_implies_person(self) -> None:
        assert "Person" in _labels(properties={"github_login": "alice"})

    def test_display_name_implies_person(self) -> None:
        assert "Person" in _labels(properties={"display_name": "Alice"})

    @pytest.mark.parametrize(
        "key,label",
        [
            ("interface_type", "Interface"),
            ("store_type", "DataStore"),
            ("agent_type", "Agent"),
            ("fix_type", "Fix"),
            ("signal_type", "DiagnosticSignal"),
            ("integration_type", "Integration"),
            ("dependency_type", "Dependency"),
            ("component_type", "Component"),
            ("workflow_type", "LocalWorkflow"),
            ("constraint_type", "Constraint"),
            ("instruction_type", "AgentInstruction"),
            ("role_type", "Role"),
            ("strategy_type", "DeploymentStrategy"),
            ("environment_type", "Environment"),
            ("asset_type", "CodeAsset"),
            ("ref_type", "SourceReference"),
            ("job_type", "MaintenanceJob"),
            ("metric_type", "Metric"),
            ("pattern_type", "MaterializedAccessPath"),
        ],
    )
    def test_string_type_signatures(self, key: str, label: str) -> None:
        assert label in _labels(properties={key: "anything"})

    def test_empty_string_type_does_not_fire(self) -> None:
        assert "Interface" not in _labels(properties={"interface_type": "   "})

    def test_non_string_type_does_not_fire(self) -> None:
        assert "Interface" not in _labels(properties={"interface_type": 42})

    def test_preference_requires_both_props(self) -> None:
        assert "Preference" not in _labels(properties={"preference_type": "style"})
        assert "Preference" in _labels(
            properties={"preference_type": "style", "scope_kind": "team"}
        )


# --- Text patterns ---------------------------------------------------------


class TestTextClassification:
    @pytest.mark.parametrize(
        "phrase,label",
        [
            ("we decided to use graphql", "Decision"),
            ("the team chose Postgres", "Decision"),
            ("ADR-3", "Decision"),
            ("ADR 4", "Decision"),
            ("architecture decision", "Decision"),
            ("incident report", "Incident"),
            ("p1 incident", "Incident"),
            ("sev-2", "Incident"),
            ("postmortem", "Incident"),
            ("hotfix landed", "Fix"),
            ("bug fix", "Fix"),
            ("workaround for the regression", "Fix"),
            ("paged on-call yesterday", "Alert"),
            ("alert fired in prod", "Alert"),
            ("alerting rule was tightened", "Alert"),
            ("see runbook for steps", "Runbook"),
            ("flaky test in CI", "BugPattern"),
            ("recurring failure", "BugPattern"),
            ("known issue", "BugPattern"),
            ("the team prefers tabs", "Preference"),
            ("we prefer pytest", "Preference"),
            ("hard constraint on memory", "Constraint"),
            ("must not be logged anywhere", "Constraint"),
            ("never commit credentials", "Constraint"),
            ("AGENTS.md", "AgentInstruction"),
            ("CLAUDE.md", "AgentInstruction"),
        ],
    )
    def test_text_cues_fire(self, phrase: str, label: str) -> None:
        assert label in _labels(properties={"summary": phrase})

    def test_no_text_no_label(self) -> None:
        assert _labels(properties={"summary": ""}) == ()


# --- Edge-endpoint inference -----------------------------------------------


class TestEdgeEndpointClassification:
    def test_outgoing_made_in_implies_decision(self) -> None:
        # ``MADE_IN`` source → Decision.
        assert "Decision" in _labels(outgoing=["MADE_IN"])

    def test_incoming_decides_for_implies_decision(self) -> None:
        # ``DECIDES_FOR`` target → Decision.
        assert "Decision" in _labels(incoming=["DECIDES_FOR"])

    def test_unknown_edge_yields_no_label(self) -> None:
        assert _labels(outgoing=["UNKNOWN_EDGE"]) == ()


# --- Canonical type hint ---------------------------------------------------


class TestCanonicalTypeHint:
    def test_canonical_type_pins_label(self) -> None:
        assert "Decision" in _labels(properties={"canonical_type": "Decision"})

    def test_unknown_canonical_type_ignored(self) -> None:
        assert _labels(properties={"canonical_type": "NotAnOntologyLabel"}) == ()

    def test_non_string_canonical_type_ignored(self) -> None:
        assert _labels(properties={"canonical_type": 123}) == ()


# --- Idempotency ------------------------------------------------------------


class TestIdempotency:
    def test_existing_label_not_returned(self) -> None:
        assert _labels(
            labels=("PullRequest",),
            properties={"pr_number": 1},
        ) == ()

    def test_only_canonical_labels_returned(self) -> None:
        # Even if a non-ontology label is somehow synthesized, it must be filtered.
        out = _labels(properties={"pr_number": 1})
        for label in out:
            # Every returned label must be a real ontology entity type.
            from domain.ontology import ENTITY_TYPES
            assert label in ENTITY_TYPES

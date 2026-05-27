"""Deterministic ontology classifier rules (edge / property / text)."""

from __future__ import annotations

import pytest

from domain.ontology_classifier import build_signals, classify_entity

pytestmark = pytest.mark.unit


def _labels(signals_labels, properties, *, out=(), incoming=()):
    return classify_entity(
        build_signals(
            labels=signals_labels,
            properties=properties,
            outgoing_edge_names=out,
            incoming_edge_names=incoming,
        )
    )


# --- Edge-endpoint rules -----------------------------------------------------------


def test_incoming_decides_for_adds_decision() -> None:
    labels = _labels(("Entity",), {}, incoming=("DECIDES_FOR",))
    assert "Decision" in labels


def test_outgoing_affects_marks_source_as_decision() -> None:
    labels = _labels(("Entity",), {}, out=("AFFECTS",))
    assert "Decision" in labels


def test_incoming_has_review_decision_adds_decision() -> None:
    labels = _labels(("Entity",), {}, incoming=("HAS_REVIEW_DECISION",))
    assert "Decision" in labels


def test_outgoing_resolved_marks_source_as_fix() -> None:
    labels = _labels(("Entity",), {}, out=("RESOLVED",))
    assert "Fix" in labels


def test_incoming_matches_pattern_adds_bugpattern() -> None:
    labels = _labels(("Entity",), {}, incoming=("MATCHES_PATTERN",))
    assert "BugPattern" in labels


def test_edge_name_is_normalized() -> None:
    assert "Decision" in _labels(("Entity",), {}, incoming=("decides_for",))
    assert "Decision" in _labels(("Entity",), {}, incoming=(" Decides-For ",))


def test_ambiguous_edge_emits_no_inference() -> None:
    assert _labels(("Entity",), {}, out=("FIXES",)) == ()
    assert _labels(("Entity",), {}, out=("OWNS",)) == ()


# --- Property signatures -----------------------------------------------------------


def test_pr_number_forces_pull_request() -> None:
    assert "PullRequest" in _labels(("Entity",), {"pr_number": 42})


def test_sha_shaped_value_forces_commit() -> None:
    assert "Commit" in _labels(("Entity",), {"sha": "a1b2c3d4e5f6"})


def test_non_sha_string_does_not_force_commit() -> None:
    assert "Commit" not in _labels(("Entity",), {"sha": "not-a-sha"})


def test_fix_type_forces_fix() -> None:
    assert "Fix" in _labels(("Entity",), {"fix_type": "code"})


def test_signal_type_forces_diagnostic_signal() -> None:
    assert "DiagnosticSignal" in _labels(("Entity",), {"signal_type": "metric"})


def test_interface_type_forces_interface() -> None:
    assert "Interface" in _labels(("Entity",), {"interface_type": "rest"})


def test_canonical_type_hint_is_respected() -> None:
    assert "Incident" in _labels(
        ("Entity",), {"canonical_type": "Incident", "title": "x", "severity": "high", "status": "open"}
    )


def test_canonical_type_hint_ignored_when_non_canonical() -> None:
    assert _labels(("Entity",), {"canonical_type": "NotAThing"}) == ()


# --- Text cues ---------------------------------------------------------------------


def test_decision_text_cue() -> None:
    labels = _labels(
        ("Entity",),
        {"summary": "We decided to split the ingestion pipeline into two workers."},
    )
    assert "Decision" in labels


def test_adr_mention_adds_decision() -> None:
    assert "Decision" in _labels(("Entity",), {"title": "ADR-0007 background workers"})


def test_incident_text_cue() -> None:
    assert "Incident" in _labels(
        ("Entity",), {"summary": "Postmortem for the billing outage on 2026-04-10."}
    )


def test_fix_text_cue() -> None:
    assert "Fix" in _labels(
        ("Entity",), {"summary": "Hotfix for the webhook retry regression."}
    )


def test_runbook_text_cue() -> None:
    assert "Runbook" in _labels(("Entity",), {"title": "On-call runbook for ledger"})


def test_agent_instruction_text_cue() -> None:
    assert "AgentInstruction" in _labels(
        ("Entity",), {"title": "AGENTS.md — testing conventions"}
    )


def test_constraint_text_cue() -> None:
    assert "Constraint" in _labels(
        ("Entity",), {"statement": "Never store access tokens in plaintext."}
    )


def test_preference_text_cue() -> None:
    assert "Preference" in _labels(
        ("Entity",), {"summary": "The team prefers pytest fixtures over unittest."}
    )


def test_bug_pattern_text_cue() -> None:
    assert "BugPattern" in _labels(
        ("Entity",), {"summary": "Flaky tests caused by unordered sets in fixtures."}
    )


def test_non_matching_text_does_not_infer() -> None:
    assert _labels(("Entity",), {"summary": "Refactored the ingestion pipeline."}) == ()


# --- Idempotence and existing labels ----------------------------------------------


def test_does_not_readd_existing_label() -> None:
    labels = _labels(
        ("Entity", "Decision"),
        {"summary": "We decided to adopt the new worker model."},
    )
    assert "Decision" not in labels


def test_classifier_only_returns_canonical_labels() -> None:
    labels = _labels(("Entity",), {"summary": "We decided to ship v2."})
    assert all(label in ("Decision",) for label in labels)


# --- Integration of multiple signals ----------------------------------------------


def test_multi_signal_classification() -> None:
    labels = _labels(
        ("Entity", "Feature"),
        {
            "name": "Background worker rewrite",
            "summary": "We decided to adopt Hatchet over Celery for background jobs.",
            "canonical_type": "Decision",
        },
        out=("AFFECTS",),
    )
    assert "Decision" in labels

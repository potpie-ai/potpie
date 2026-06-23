"""Tests for opt-in coverage/precision floor gating."""

from __future__ import annotations

from context_engine.benchmarks.core.graph_inspect import GraphSnapshot
from context_engine.benchmarks.core.scenario import (
    EntityAssertion,
    PostIngestAssertions,
    ReconciliationAssertion,
    RetrievalAssertions,
)
from context_engine.benchmarks.evaluators.ingestion_quality import evaluate_ingestion_quality
from context_engine.benchmarks.evaluators.retrieval import evaluate_retrieval


def test_ingestion_coverage_floor_gates_otherwise_passing_axis() -> None:
    """Even if structural score is high, a coverage floor violation fails the axis."""
    # Set up assertions that would normally pass (single entity, satisfied)
    # but with a high coverage_floor of 100 that allows no missing matches.
    assertions = PostIngestAssertions(
        graph_must_contain_entities=(
            EntityAssertion(label="Service", min_count=1),
            EntityAssertion(label="Decision", min_count=1),  # not satisfied
        ),
        reconciliation=ReconciliationAssertion(),
        coverage_floor=80,
    )
    snapshot = GraphSnapshot()
    # Add one Service that satisfies the first assertion.
    from context_engine.benchmarks.core.graph_inspect import GraphEntity

    snapshot.entities.append(GraphEntity(label="Service", key="s1", properties={}))
    result = evaluate_ingestion_quality(
        snapshot=snapshot, outcomes=[], assertions=assertions
    )
    # Coverage = 50 (1 of 2 satisfied), floor = 80 → fail
    assert result.coverage == 50.0
    assert not result.passed
    assert any("coverage" in e and "below floor" in e for e in result.errors)


def test_ingestion_no_floor_means_report_only() -> None:
    """Without floors, sub-axes are reported but don't gate."""
    assertions = PostIngestAssertions(
        graph_must_contain_entities=(EntityAssertion(label="Decision"),),
        reconciliation=ReconciliationAssertion(),
    )
    result = evaluate_ingestion_quality(
        snapshot=GraphSnapshot(), outcomes=[], assertions=assertions
    )
    # Coverage = 0 but no floor set → passing decision is purely about primary.
    # Primary score = 0 (nothing matched), so it fails primary too; but the
    # *reason* must not be a floor violation.
    assert result.coverage == 0.0
    assert not any("below floor" in e for e in result.errors)


def test_retrieval_precision_floor_gates() -> None:
    """A response that cites distractors fails when precision_floor is set."""
    from context_engine.benchmarks.evaluators.retrieval import set_fixture_source_id_lookup

    set_fixture_source_id_lookup(
        {
            "linear/signal.json": "linear:signal",
            "linear/noise.json": "linear:noise",
        }
    )
    response = {
        "answer": {"summary": ""},
        "source_refs": [
            {"source_id": "linear:signal"},
            {"source_id": "linear:noise"},
        ],
    }
    assertions = RetrievalAssertions(
        must_cite_event_ids=("linear/signal.json",),
        must_not_cite_event_ids=("linear/noise.json",),
        precision_floor=80,
    )
    result = evaluate_retrieval(response, assertions)
    # The response cited the signal AND the distractor → precision = 50.
    assert result.precision == 50.0
    assert not result.passed
    assert any("precision" in e and "below floor" in e for e in result.errors)
    set_fixture_source_id_lookup({})

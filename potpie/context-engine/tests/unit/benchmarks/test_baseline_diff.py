"""Tests for the baseline-diff renderer."""

from __future__ import annotations

from context_engine.benchmarks.core.reporting import (
    aggregate_report,
    diff_reports,
    render_diff_markdown,
)
from context_engine.benchmarks.core.result import AxisScore, ScenarioResult


def _scenario(id: str, agg: float, use_case: str = "PREF") -> ScenarioResult:
    return ScenarioResult(
        id=id,
        use_case=use_case,
        tier="quick",
        aggregate_score=agg,
        aggregate_passed=agg >= 70,
        ingestion=AxisScore(score=agg, passed=True, coverage=80, precision=90),
        retrieval=AxisScore(score=agg, passed=True, coverage=80, precision=90),
        synthesis=AxisScore(score=agg, passed=True),
        difficulty="easy",
        source_mix="single",
        dimensions=[use_case],
    )


def _report(label: str, results):
    return aggregate_report(
        schema_version="v3",
        started_at=label,
        finished_at=label,
        engine_url="x",
        tier="quick",
        use_case_filter=None,
        results=results,
    )


def test_diff_surfaces_regression_and_recovery() -> None:
    base = _report("t0", [_scenario("a", 80), _scenario("b", 50)])
    cur = _report("t1", [_scenario("a", 60), _scenario("b", 70)])
    diff = diff_reports(cur, base)
    by_id = {r["id"]: r for r in diff["scenarios"]}
    assert by_id["a"]["delta_aggregate"] == -20.0
    assert by_id["a"]["pass_change"] == "regressed"
    assert by_id["b"]["delta_aggregate"] == 20.0
    assert by_id["b"]["pass_change"] == "recovered"


def test_diff_flags_new_and_dropped_scenarios() -> None:
    base = _report("t0", [_scenario("kept", 70), _scenario("dropped", 60)])
    cur = _report("t1", [_scenario("kept", 70), _scenario("added", 80)])
    diff = diff_reports(cur, base)
    by_id = {r["id"]: r for r in diff["scenarios"]}
    assert by_id["added"]["new"] is True
    assert by_id["dropped"]["dropped"] is True
    assert by_id["kept"]["pass_change"] == "stable"


def test_render_diff_markdown_is_well_formed() -> None:
    base = _report("t0", [_scenario("a", 80)])
    cur = _report("t1", [_scenario("a", 60)])
    out = render_diff_markdown(diff_reports(cur, base))
    assert "# Benchmark diff" in out
    assert "By use case" in out
    assert "regressed" in out

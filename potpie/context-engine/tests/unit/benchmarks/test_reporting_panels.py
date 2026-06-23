"""Tests for the new reporting panels (by_dimension / by_difficulty / by_source_mix)."""

from __future__ import annotations

from potpie.context_engine.benchmarks.core.reporting import aggregate_report
from potpie.context_engine.benchmarks.core.result import AxisScore, DimensionScore, ScenarioResult


def _scenario(
    *,
    id: str,
    use_case: str = "PREF",
    aggregate: float = 70.0,
    difficulty: str = "easy",
    source_mix: str = "single",
    dimensions: list[str] | None = None,
    by_dimension: list[DimensionScore] | None = None,
) -> ScenarioResult:
    return ScenarioResult(
        id=id,
        use_case=use_case,
        tier="quick",
        aggregate_score=aggregate,
        aggregate_passed=aggregate >= 70,
        ingestion=AxisScore(score=aggregate, passed=True, coverage=80, precision=90),
        retrieval=AxisScore(score=aggregate, passed=True, coverage=80, precision=90),
        synthesis=AxisScore(score=aggregate, passed=True),
        difficulty=difficulty,
        source_mix=source_mix,
        dimensions=dimensions or [use_case],
        by_dimension=by_dimension or [],
    )


def test_by_difficulty_and_source_mix_aggregate() -> None:
    results = [
        _scenario(id="a", aggregate=80, difficulty="easy", source_mix="single"),
        _scenario(id="b", aggregate=60, difficulty="hard", source_mix="full"),
        _scenario(id="c", aggregate=40, difficulty="hard", source_mix="full"),
    ]
    report = aggregate_report(
        schema_version="v3",
        started_at="t0",
        finished_at="t1",
        engine_url="x",
        tier="quick",
        use_case_filter=None,
        results=results,
    )
    assert report.by_difficulty["easy"]["aggregate"] == 80.0
    assert report.by_difficulty["easy"]["pass_rate"] == 1.0
    assert report.by_difficulty["hard"]["count"] == 2
    assert report.by_difficulty["hard"]["aggregate"] == 50.0
    assert report.by_difficulty["hard"]["pass_rate"] == 0.0

    assert report.by_source_mix["full"]["count"] == 2
    assert report.by_source_mix["full"]["aggregate"] == 50.0


def test_by_dimension_breaks_out_composite() -> None:
    composite = _scenario(
        id="combo",
        use_case="COMBO",
        aggregate=65,
        difficulty="medium",
        source_mix="dual",
        dimensions=["PREF", "INFRA"],
        by_dimension=[
            DimensionScore(
                dimension="PREF", ingestion=70, retrieval=70, synthesis=80, aggregate=73
            ),
            DimensionScore(
                dimension="INFRA",
                ingestion=70,
                retrieval=70,
                synthesis=40,
                aggregate=58,
            ),
        ],
    )
    pref_only = _scenario(id="pref", aggregate=78)
    report = aggregate_report(
        schema_version="v3",
        started_at="t0",
        finished_at="t1",
        engine_url="x",
        tier="quick",
        use_case_filter=None,
        results=[composite, pref_only],
    )
    # PREF gets credit from both the composite (via dimension breakout) and the standalone PREF scenario.
    assert report.by_dimension["PREF"]["count"] == 2
    # INFRA only the composite.
    assert report.by_dimension["INFRA"]["count"] == 1
    # INFRA's score should reflect the composite's INFRA-dim score (58), not the scenario aggregate (65).
    assert abs(report.by_dimension["INFRA"]["aggregate"] - 58.0) < 0.01

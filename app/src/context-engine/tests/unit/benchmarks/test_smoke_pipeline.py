"""Tests for the in-process smoke pipeline."""

from __future__ import annotations

from pathlib import Path

from benchmarks.core.smoke import run_smoke


def test_smoke_passes_for_authored_corpus() -> None:
    """All currently-authored scenarios load, resolve their fixtures, and
    run through the in-process evaluator pipeline without raising."""
    bench_root = Path(__file__).resolve().parents[3] / "benchmarks"
    report = run_smoke(bench_root)
    failures = [f for f in report.findings if not f.ok]
    assert not failures, "\n".join(f"{f.scenario_id}: {f.detail}" for f in failures)
    assert report.passed

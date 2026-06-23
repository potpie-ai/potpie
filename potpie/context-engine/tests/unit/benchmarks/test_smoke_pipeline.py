"""Tests for the in-process smoke pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from potpie.context_engine.benchmarks.core.smoke import run_smoke

_BENCH_ROOT = Path(__file__).resolve().parents[3] / "benchmarks"
_UNIVERSE_ROOT = _BENCH_ROOT / "fixtures" / "raw_events" / "universe" / "acme"
_CORPUS_SHIPPED = _UNIVERSE_ROOT.is_dir()


@pytest.mark.skipif(
    not _CORPUS_SHIPPED,
    reason="Benchmark fixture corpus is not shipped in this checkout",
)
def test_smoke_passes_for_authored_corpus() -> None:
    """All currently-authored scenarios load, resolve their fixtures, and
    run through the in-process evaluator pipeline without raising."""
    bench_root = Path(__file__).resolve().parents[3] / "benchmarks"
    report = run_smoke(bench_root)
    failures = [f for f in report.findings if not f.ok]
    assert not failures, "\n".join(f"{f.scenario_id}: {f.detail}" for f in failures)
    assert report.passed

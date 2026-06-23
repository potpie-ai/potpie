"""Tests for the coverage + precision sub-axis evaluators."""

from __future__ import annotations

from context_engine.benchmarks.evaluators.coverage import coverage_score
from context_engine.benchmarks.evaluators.precision import precision_score


def test_coverage_full_recall_is_100() -> None:
    assert coverage_score(expected=5, found=5) == 100.0


def test_coverage_zero_expected_is_100() -> None:
    # No expectations means nothing to miss.
    assert coverage_score(expected=0, found=0) == 100.0
    assert coverage_score(expected=0, found=7) == 100.0


def test_coverage_partial() -> None:
    assert coverage_score(expected=4, found=3) == 75.0


def test_precision_no_distractors_is_100() -> None:
    assert precision_score(relevant=10, distractors=0) == 100.0


def test_precision_collapses_with_distractors() -> None:
    assert precision_score(relevant=1, distractors=3) == 25.0


def test_precision_zero_both_is_100() -> None:
    assert precision_score(relevant=0, distractors=0) == 100.0

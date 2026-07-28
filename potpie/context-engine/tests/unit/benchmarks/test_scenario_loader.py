"""Tests for the v3 scenario loader.

Covers the schema additions in the bench-plan v3 rewrite: new
USE_CASES enum, dimensions / difficulty / source_mix, distractor +
seed parsing, must_cite list, temporal block, and per-use-case
axis-weight defaults.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from potpie_context_engine.benchmarks.core.scenario import (
    DEFAULT_AXIS_WEIGHTS,
    AxisWeights,
    ScenarioLoadError,
    load_scenario,
)


def _write(tmp_path: Path, body: str, name: str = "s.yaml") -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


_MINIMAL = """\
id: t_minimal
use_case: BUG
ingest:
  - { event: linear/x.json, at: "0d" }
query:
  intent: debugging
judge:
  pass_score: 60
  criteria:
    - { name: c, weight: 100, pass_threshold: 3, prompt: "ok?" }
"""


def test_loads_minimal_v3_scenario(tmp_path: Path) -> None:
    s = load_scenario(_write(tmp_path, _MINIMAL))
    assert s.use_case == "BUG"
    assert s.difficulty == "easy"  # default
    assert s.source_mix == "single"  # default
    assert s.effective_dimensions == ("BUG",)


def test_per_use_case_axis_weight_defaults(tmp_path: Path) -> None:
    """Each use case picks its declared default unless overridden."""
    for use_case, expected in DEFAULT_AXIS_WEIGHTS.items():
        body = _MINIMAL.replace("use_case: BUG", f"use_case: {use_case}")
        if use_case == "COMBO":
            body = body.replace("ingest:", "dimensions: [PREF, INFRA]\ningest:")
        s = load_scenario(_write(tmp_path, body, name=f"s_{use_case}.yaml"))
        ing, ret, syn = expected
        # Defaults already sum to 1.0, so normalized() is a no-op.
        assert s.axis_weights == AxisWeights(
            ingestion=ing, retrieval=ret, synthesis=syn
        )


def test_combo_requires_at_least_two_dimensions(tmp_path: Path) -> None:
    body = _MINIMAL.replace("use_case: BUG", "use_case: COMBO")
    with pytest.raises(ScenarioLoadError, match="at least 2"):
        load_scenario(_write(tmp_path, body))


def test_unknown_dimension_is_rejected(tmp_path: Path) -> None:
    body = _MINIMAL.replace(
        "use_case: BUG", "use_case: COMBO\ndimensions: [PREF, BANANA]"
    )
    with pytest.raises(ScenarioLoadError, match="dimension 'BANANA'"):
        load_scenario(_write(tmp_path, body))


def test_must_cite_accepts_list_or_string(tmp_path: Path) -> None:
    body_str = (
        _MINIMAL
        + """\
retrieval_assertions:
  must_cite_event_id: linear/x.json
"""
    )
    s = load_scenario(_write(tmp_path, body_str, name="s_str.yaml"))
    assert s.retrieval_assertions.must_cite_event_ids == ("linear/x.json",)

    body_list = (
        _MINIMAL
        + """\
retrieval_assertions:
  must_cite_event_id:
    - linear/x.json
    - linear/y.json
"""
    )
    s2 = load_scenario(_write(tmp_path, body_list, name="s_list.yaml"))
    assert s2.retrieval_assertions.must_cite_event_ids == (
        "linear/x.json",
        "linear/y.json",
    )


def test_must_not_cite_and_temporal_window_parse(tmp_path: Path) -> None:
    body = (
        _MINIMAL
        + """\
retrieval_assertions:
  must_not_cite_event_id:
    - github/noise.json
  temporal:
    must_order_correctly: true
    window: { from: "-14d", to: "0d" }
    out_of_window_refs_max: 0
"""
    )
    s = load_scenario(_write(tmp_path, body))
    assert s.retrieval_assertions.must_not_cite_event_ids == ("github/noise.json",)
    assert s.retrieval_assertions.temporal is not None
    assert s.retrieval_assertions.temporal.window_from == "-14d"
    assert s.retrieval_assertions.temporal.must_order_correctly is True


def test_distractor_count_and_range(tmp_path: Path) -> None:
    body = (
        _MINIMAL
        + """\
distractor_events:
  - { event: github/n.json, at: "-21d..-7d", count: 12, shape: "noise/random" }
"""
    )
    s = load_scenario(_write(tmp_path, body))
    assert len(s.distractor_events) == 1
    d = s.distractor_events[0]
    assert d.count == 12
    assert d.at == "-21d..-7d"
    assert d.shape == "noise/random"


def test_judge_criterion_dimensions_validated(tmp_path: Path) -> None:
    body = _MINIMAL.replace(
        '- { name: c, weight: 100, pass_threshold: 3, prompt: "ok?" }',
        '- { name: c, weight: 100, pass_threshold: 3, prompt: "ok?", dimensions: [TIME] }',
    )
    s = load_scenario(_write(tmp_path, body))
    assert s.judge.criteria[0].dimensions == ("TIME",)


def test_invalid_difficulty_rejected(tmp_path: Path) -> None:
    body = _MINIMAL + "difficulty: extreme\n"
    with pytest.raises(ScenarioLoadError, match="difficulty 'extreme'"):
        load_scenario(_write(tmp_path, body))

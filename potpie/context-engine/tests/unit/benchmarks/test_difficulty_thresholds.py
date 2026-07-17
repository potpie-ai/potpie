"""Tests for difficulty-adjusted default ``pass_score``."""

from __future__ import annotations

from pathlib import Path

from potpie_context_engine.benchmarks.core.scenario import DEFAULT_PASS_SCORE_BY_DIFFICULTY, load_scenario


_MINIMAL = """\
id: t_minimal
use_case: BUG
difficulty: {difficulty}
ingest:
  - {{ event: linear/x.json, at: "0d" }}
query:
  intent: debugging
judge:
  criteria:
    - {{ name: c, weight: 100, pass_threshold: 3, prompt: "ok?" }}
"""


def _write(tmp_path: Path, difficulty: str) -> Path:
    p = tmp_path / f"s_{difficulty}.yaml"
    p.write_text(_MINIMAL.format(difficulty=difficulty), encoding="utf-8")
    return p


def test_default_pass_score_per_difficulty(tmp_path: Path) -> None:
    for difficulty, expected in DEFAULT_PASS_SCORE_BY_DIFFICULTY.items():
        s = load_scenario(_write(tmp_path, difficulty))
        assert s.judge.pass_score == expected, (
            f"difficulty={difficulty} -> pass_score={s.judge.pass_score}, "
            f"expected {expected}"
        )


def test_explicit_pass_score_wins(tmp_path: Path) -> None:
    body = _MINIMAL.format(difficulty="hard").replace(
        "judge:\n  criteria:",
        "judge:\n  pass_score: 90\n  criteria:",
    )
    p = tmp_path / "s_explicit.yaml"
    p.write_text(body, encoding="utf-8")
    s = load_scenario(p)
    # Explicit overrides the difficulty-default (which would be 55 for hard).
    assert s.judge.pass_score == 90

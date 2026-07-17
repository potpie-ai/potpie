"""Tests for the curated light subset used by ``benchmarks run-light``.

Two contracts:

1. The scenario loader picks up the optional ``light: true`` field and
   exposes it on the ``Scenario`` dataclass (default ``False``).
2. The shipped corpus has exactly one ``light: true`` scenario per
   dimension (PREF / INFRA / TIME / BUG / COMBO) — five total — so a
   ``benchmarks run-light`` always exercises every dimension once.

These tests run in <1 s and are part of the smoke layer authors get
back from ``pytest tests/unit/benchmarks/``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from potpie_context_engine.benchmarks.core.scenario import discover_scenarios, load_scenario

# tests/unit/benchmarks/test_light_subset.py → ../../../ = context-engine/
BENCHMARKS_ROOT = Path(__file__).resolve().parents[3] / "src" / "potpie_context_engine" / "benchmarks"


def _write(tmp_path: Path, body: str, name: str = "s.yaml") -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


_MINIMAL = """\
id: t_light
use_case: BUG
ingest:
  - {{ event: linear/x.json, at: "0d" }}
query:
  intent: debugging
judge:
  pass_score: 60
  criteria:
    - {{ name: c, weight: 100, pass_threshold: 3, prompt: "ok?" }}
{extra}
"""


def test_loader_defaults_light_to_false(tmp_path: Path):
    s = load_scenario(_write(tmp_path, _MINIMAL.format(extra="")))
    assert s.light is False


def test_loader_picks_up_light_true(tmp_path: Path):
    s = load_scenario(_write(tmp_path, _MINIMAL.format(extra="light: true\n")))
    assert s.light is True


def test_loader_picks_up_light_false_explicit(tmp_path: Path):
    s = load_scenario(_write(tmp_path, _MINIMAL.format(extra="light: false\n")))
    assert s.light is False


def test_corpus_has_one_light_scenario_per_dimension():
    """The shipped corpus must keep run-light at five scenarios — one
    per dimension. Editing this test is fine; let it remind you.
    """
    scenarios = discover_scenarios(BENCHMARKS_ROOT)
    light = [s for s in scenarios if s.light]
    assert len(light) == 5, (
        f"expected 5 light scenarios; got {len(light)}: {sorted(s.id for s in light)}"
    )
    by_dim = Counter(s.use_case for s in light)
    assert dict(by_dim) == {"PREF": 1, "INFRA": 1, "TIME": 1, "BUG": 1, "COMBO": 1}, (
        f"each dimension must have exactly one light scenario; got {dict(by_dim)}"
    )

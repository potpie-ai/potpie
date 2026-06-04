"""Golden cases for episodic edge collapse (docs 02-edge-type-collapse)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from domain.entity_schema import normalized_episodic_edge_allowlist
from domain.extraction_edges import classify_episodic_edge

pytestmark = pytest.mark.unit

_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "edge_collapse_golden.json"


def _cases() -> list[dict[str, object]]:
    raw = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return list(raw["cases"])


@pytest.mark.parametrize("case", _cases(), ids=lambda c: str(c["id"]))
def test_golden_classify_episodic_edge(case: dict[str, object]) -> None:
    allowed = normalized_episodic_edge_allowlist()
    name, ls = classify_episodic_edge(
        str(case["relation_name"]),
        str(case["fact"]),
        tuple(case["source_labels"]),  # type: ignore[arg-type]
        tuple(case["target_labels"]),  # type: ignore[arg-type]
        allowed_normalized_names=allowed,
        existing_lifecycle=None,
    )
    assert name == case["expected_name"]
    assert ls == case["expected_lifecycle"]

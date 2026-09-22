"""The requested/effective adjustment contract."""

from __future__ import annotations

import pytest

from potpie_context_core.adjustments import (
    Adjustment,
    adjustment_lines,
    adjustments_from_dicts,
    dedupe_adjustments,
    with_adjustments,
)

pytestmark = pytest.mark.unit


def _depth(requested: int = 100) -> Adjustment:
    return Adjustment(
        field="depth",
        requested=requested,
        effective=4,
        reason="maximum_supported",
        message=f"Depth {requested} exceeds the supported maximum 4; returned depth-4 context.",
        max_supported=4,
    )


def test_payload_without_adjustments_is_returned_unchanged() -> None:
    payload = {"ok": True, "items": [1]}
    assert with_adjustments(payload, ()) == payload
    assert "status" not in with_adjustments(payload, ())


def test_payload_with_adjustments_gains_status_and_records() -> None:
    out = with_adjustments({"ok": True, "items": []}, [_depth()])
    assert out["status"] == "adjusted"
    assert out["adjustments"] == [_depth().to_dict()]
    assert out["adjustments"][0]["max_supported"] == 4


def test_a_non_ok_status_is_never_relabelled_as_adjusted() -> None:
    out = with_adjustments({"ok": False, "status": "partial"}, [_depth()])
    assert out["status"] == "partial"
    assert out["adjustments"]


def test_duplicate_adjustments_collapse_in_order() -> None:
    other = Adjustment(
        field="view",
        requested="Docs",
        effective="docs",
        reason="canonical_case",
        message="m",
    )
    assert dedupe_adjustments([_depth(), other, _depth(), other]) == (_depth(), other)


def test_text_lines_use_the_tilde_marker_once_per_adjustment() -> None:
    assert adjustment_lines([_depth(), _depth()]) == [
        "~ Depth 100 exceeds the supported maximum 4; returned depth-4 context."
    ]


def test_dicts_round_trip_and_ignore_garbage() -> None:
    rehydrated = adjustments_from_dicts([_depth().to_dict(), {"nope": 1}, "x"])
    assert rehydrated == (_depth(),)

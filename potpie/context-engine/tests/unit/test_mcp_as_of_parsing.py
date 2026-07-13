"""MCP ``as_of`` parsing: every accepted form must yield a tz-aware datetime.

A naive datetime here becomes the ranker's ``now`` (via ``ReadRequest.as_of``
-> ``TaskContext.now``) and crashed recency scoring against tz-normalized
claim times with ``TypeError: can't subtract offset-naive and offset-aware
datetimes``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from adapters.inbound.mcp.server import _parse_as_of_iso

pytestmark = pytest.mark.unit


def test_none_and_blank_return_none() -> None:
    assert _parse_as_of_iso(None) is None
    assert _parse_as_of_iso("") is None
    assert _parse_as_of_iso("   ") is None


@pytest.mark.parametrize(
    "value",
    [
        "2024-06-01",
        "2024-06-01T10:30:00",
        "2024-06-01T10:30:00Z",
        "2024-06-01T10:30:00+05:30",
    ],
)
def test_parsed_as_of_is_always_tz_aware(value: str) -> None:
    parsed = _parse_as_of_iso(value)
    assert parsed is not None
    assert parsed.tzinfo is not None


def test_naive_input_interpreted_as_utc() -> None:
    parsed = _parse_as_of_iso("2024-06-01T10:30:00")
    assert parsed == datetime(2024, 6, 1, 10, 30, tzinfo=timezone.utc)


def test_explicit_offset_preserved() -> None:
    parsed = _parse_as_of_iso("2024-06-01T10:30:00+05:30")
    assert parsed is not None
    assert parsed.utcoffset() is not None
    assert parsed.utcoffset().total_seconds() == 5.5 * 3600


def test_z_suffix_is_utc() -> None:
    parsed = _parse_as_of_iso("2024-06-01T10:30:00Z")
    assert parsed == datetime(2024, 6, 1, 10, 30, tzinfo=timezone.utc)

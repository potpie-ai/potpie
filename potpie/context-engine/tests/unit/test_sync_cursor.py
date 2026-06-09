"""Unit tests for the diff-sync cursor parser."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from domain.sync_cursor import parse_cursor_since

pytestmark = pytest.mark.unit


def test_none_and_empty_return_none() -> None:
    assert parse_cursor_since(None) is None
    assert parse_cursor_since("") is None
    assert parse_cursor_since("   ") is None


def test_garbage_returns_none_not_raises() -> None:
    # A bad cursor recorded in history must never crash an enumeration.
    assert parse_cursor_since("not-a-date") is None
    assert parse_cursor_since("2026-13-99") is None


def test_z_suffix_parses_as_utc() -> None:
    dt = parse_cursor_since("2026-06-01T12:00:00Z")
    assert dt == datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def test_offset_is_normalized_to_utc() -> None:
    dt = parse_cursor_since("2026-06-01T12:00:00+02:00")
    assert dt == datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)


def test_naive_string_assumed_utc() -> None:
    dt = parse_cursor_since("2026-06-01T00:00:00")
    assert dt == datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)

"""Tests for sync wait helper."""

from __future__ import annotations

from unittest.mock import MagicMock

from application.use_cases.wait_ingestion_event import wait_for_terminal_ingestion_event


def test_wait_returns_immediately_on_terminal() -> None:
    ev = MagicMock()
    ev.status = "done"
    store = MagicMock()
    store.get_event.return_value = ev
    out = wait_for_terminal_ingestion_event(
        store, "e1", timeout_seconds=1.0, poll_interval_seconds=0.01
    )
    assert out is ev
    store.get_event.assert_called_with("e1")


def test_wait_returns_none_when_missing() -> None:
    store = MagicMock()
    store.get_event.return_value = None
    out = wait_for_terminal_ingestion_event(store, "missing", timeout_seconds=0.2, poll_interval_seconds=0.05)
    assert out is None

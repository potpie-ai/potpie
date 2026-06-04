"""Phase 6: compatibility alias ``POST /events/ingest`` observability."""

from __future__ import annotations

import logging

import pytest
from fastapi import Response

from adapters.inbound.http.api.v1.context.router import (
    _EVENTS_INGEST_ALIAS_WARNING,
    _apply_events_ingest_alias_headers,
    _mark_events_ingest_alias,
    events_ingest_alias_call_count,
    reset_events_ingest_alias_counter,
)

pytestmark = pytest.mark.unit


class _StubUrl:
    def __init__(self, path: str) -> None:
        self.path = path


class _StubRequest:
    def __init__(self, path: str) -> None:
        self.url = _StubUrl(path)


def test_mark_alias_increments_counter_and_logs(caplog):
    reset_events_ingest_alias_counter()
    assert events_ingest_alias_call_count() == 0

    with caplog.at_level(logging.WARNING):
        assert _mark_events_ingest_alias(
            _StubRequest("/api/v2/context/events/ingest")
        )

    assert events_ingest_alias_call_count() == 1
    assert any(
        "deprecated alias POST /events/ingest" in rec.message
        for rec in caplog.records
    )


def test_mark_canonical_path_does_not_increment_counter():
    reset_events_ingest_alias_counter()
    assert not _mark_events_ingest_alias(
        _StubRequest("/api/v2/context/events/reconcile")
    )
    assert events_ingest_alias_call_count() == 0


def test_alias_headers_include_deprecation_warning_and_link():
    resp = Response()
    _apply_events_ingest_alias_headers(resp)

    assert resp.headers["Deprecation"] == "true"
    assert resp.headers["Warning"] == _EVENTS_INGEST_ALIAS_WARNING
    assert "events/reconcile" in resp.headers["Link"]
    assert 'rel="successor-version"' in resp.headers["Link"]


def test_counter_is_reset_between_tests():
    reset_events_ingest_alias_counter()
    for _ in range(3):
        _mark_events_ingest_alias(_StubRequest("/api/v2/context/events/ingest"))
    assert events_ingest_alias_call_count() == 3
    reset_events_ingest_alias_counter()
    assert events_ingest_alias_call_count() == 0

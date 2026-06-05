"""Phase 6: compatibility alias ``POST /events/ingest`` observability."""

from __future__ import annotations

import logging

import pytest
from fastapi import Response

from adapters.inbound.http.api.v1.context.router import (
    ContextEventHttpBody,
    _EVENTS_INGEST_ALIAS_WARNING,
    _apply_events_ingest_alias_headers,
    _mark_events_ingest_alias,
    ContextEventHttpBody,
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


def test_context_event_http_body_accepts_non_repo_scoped_linear_event():
    body = ContextEventHttpBody.model_validate(
        {
            "pot_id": "pot-1",
            "source_system": "linear",
            "event_type": "linear_team",
            "action": "one_shot_ingest",
            "source_id": "one_shot_ingest:linear:eng:42",
            "payload": {"team": "ENG", "count": 120},
        }
    )

    assert body.repo_name is None
    assert body.provider is None
    assert body.provider_host is None


def test_context_event_body_allows_repo_less_jira_event():
    body = ContextEventHttpBody.model_validate(
        {
            "pot_id": "11111111-1111-4111-8111-111111111111",
            "source_system": "jira",
            "event_type": "jira_project",
            "action": "one_shot_ingest",
            "source_id": "one_shot_ingest:jira:proj:1",
            "payload": {"project_key": "PROJ", "count": 3},
        }
    )

    assert body.provider is None
    assert body.provider_host is None
    assert body.repo_name is None

"""Unit tests for MCP context_ingest tool implementation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from adapters.inbound.mcp.tool_impl import run_context_ingest
from application.use_cases.submit_raw_episode import RawEpisodeSubmissionResult
from domain.actor import Actor

pytestmark = pytest.mark.unit


def test_run_context_ingest_rejects_empty_episode_body() -> None:
    container = MagicMock()
    db = MagicMock()
    actor = Actor(user_id="u1", surface="mcp", client_name="test", auth_method="api_key")
    with patch(
        "adapters.inbound.mcp.tool_impl.policy_error_response", return_value=None
    ):
        out = run_context_ingest(
            container=container,
            db=db,
            actor=actor,
            pot_id="pot-1",
            name="episode",
            episode_body="   ",
            source_description="mcp",
        )
    assert out["ok"] is False
    assert out["error"] == "validation_error"
    db.commit.assert_not_called()


def test_run_context_ingest_maps_duplicate_as_ok() -> None:
    container = MagicMock()
    db = MagicMock()
    actor = Actor(user_id="u1", surface="mcp", client_name="test", auth_method="api_key")
    result = RawEpisodeSubmissionResult(
        ok=False,
        status="duplicate",
        event_id="ev-1",
        duplicate_reason="idempotency",
    )
    with (
        patch(
            "adapters.inbound.mcp.tool_impl.policy_error_response", return_value=None
        ),
        patch(
            "adapters.inbound.mcp.tool_impl.submit_raw_episode", return_value=result
        ),
    ):
        out = run_context_ingest(
            container=container,
            db=db,
            actor=actor,
            pot_id="pot-1",
            name="episode",
            episode_body="body",
            source_description="mcp",
            idempotency_key="key-1",
        )
    assert out["ok"] is True
    assert out["status"] == "duplicate"
    assert out["event_id"] == "ev-1"
    db.rollback.assert_called_once()


def test_run_context_ingest_maps_queued_with_fallback() -> None:
    container = MagicMock()
    db = MagicMock()
    actor = Actor(user_id="u1", surface="mcp", client_name="test", auth_method="api_key")
    result = RawEpisodeSubmissionResult(
        ok=True,
        status="queued",
        event_id="ev-2",
        job_id="job-2",
    )
    with (
        patch(
            "adapters.inbound.mcp.tool_impl.policy_error_response", return_value=None
        ),
        patch(
            "adapters.inbound.mcp.tool_impl.submit_raw_episode", return_value=result
        ),
    ):
        out = run_context_ingest(
            container=container,
            db=db,
            actor=actor,
            pot_id="pot-1",
            name="episode",
            episode_body="body",
            source_description="mcp",
        )
    assert out["ok"] is True
    assert out["status"] == "queued"
    assert out["fallbacks"]
    db.commit.assert_called_once()

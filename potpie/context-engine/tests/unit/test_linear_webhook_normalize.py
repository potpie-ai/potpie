"""normalize_linear_webhook: webhook payloads → IngestionSubmissionRequest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from application.use_cases.normalize_linear_webhook import (
    LinearWebhookError,
    normalize_linear_webhook,
)
from domain.ingestion_kinds import INGESTION_KIND_AGENT_RECONCILIATION

pytestmark = pytest.mark.unit

DATA = Path(__file__).resolve().parent.parent / "data" / "linear"


def _load(name: str) -> dict[str, Any]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def test_issue_create_webhook_maps_to_agent_reconciliation_event() -> None:
    body = _load("webhook_issue_create.json")
    req = normalize_linear_webhook(body, pot_id="pot-1")
    assert req.ingestion_kind == INGESTION_KIND_AGENT_RECONCILIATION
    assert req.source_system == "linear"
    assert req.event_type == "linear_issue"
    assert req.action == "create"
    assert req.provider == "linear"
    assert req.provider_host == "linear.app"
    assert req.repo_name == "team:team_eng"
    assert req.source_id == "linear:issue:ENG-42:create"
    assert req.payload["action"] == "create"
    assert req.payload["issue"]["identifier"] == "ENG-42"
    # Occurred-at comes from the webhook envelope.
    assert req.occurred_at is not None
    assert req.occurred_at.tzinfo is not None


def test_issue_state_change_webhook_synthesizes_state_change_action() -> None:
    body = _load("webhook_issue_state_change.json")
    req = normalize_linear_webhook(body, pot_id="pot-1")
    assert req.action == "state_change"
    assert req.payload["action"] == "state_change"
    # Previous state extracted from updatedFrom.
    assert req.payload["previous_state"]["name"] == "Triage"
    assert req.source_id == "linear:issue:ENG-42:state_change"


def test_comment_create_webhook_maps_to_comment_added() -> None:
    body = _load("webhook_comment_create.json")
    req = normalize_linear_webhook(body, pot_id="pot-1")
    assert req.event_type == "linear_comment"
    assert req.action == "comment_added"
    assert req.payload["issue"]["identifier"] == "ENG-42"
    assert req.payload["comment"]["id"] == "comment_01"
    assert req.source_id == "linear:comment:comment_01"


def test_unsupported_type_raises_webhook_error() -> None:
    with pytest.raises(LinearWebhookError):
        normalize_linear_webhook(
            {"type": "Project", "action": "create", "data": {"id": "p1"}},
            pot_id="pot-1",
        )


def test_comment_update_is_unsupported() -> None:
    body = _load("webhook_comment_create.json")
    body["action"] = "update"
    with pytest.raises(LinearWebhookError):
        normalize_linear_webhook(body, pot_id="pot-1")


def test_body_without_data_raises() -> None:
    with pytest.raises(LinearWebhookError):
        normalize_linear_webhook({"type": "Issue", "action": "create"}, pot_id="pot-1")


def test_issue_update_without_state_change_stays_as_update() -> None:
    body = _load("webhook_issue_state_change.json")
    # Strip updatedFrom to simulate a non-state update (e.g. priority change).
    body.pop("updatedFrom", None)
    req = normalize_linear_webhook(body, pot_id="pot-1")
    assert req.action == "update"
    assert req.source_id == "linear:issue:ENG-42:update"

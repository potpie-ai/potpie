"""Linear sync → context graph: episodic.add_episode is invoked for each issue."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from integrations.adapters.outbound.postgres.integration_model import Integration
from integrations.adapters.outbound.postgres.project_source_model import ProjectSource
from integrations.adapters.outbound.linear.linear_sync import sync_linear_project_source


@pytest.fixture
def ps_row():
    p = MagicMock(spec=ProjectSource)
    p.id = "ps-test-1"
    p.provider = "linear"
    p.integration_id = "int-linear-1"
    p.project_id = "pot-test-1"
    p.scope_json = {"team_id": "team-99"}
    return p


@pytest.fixture
def int_row():
    i = MagicMock(spec=Integration)
    i.integration_id = "int-linear-1"
    i.integration_type = "linear"
    i.active = True
    i.auth_data = {"access_token": "enc", "refresh_token": "", "expires_at": None}
    return i


def test_linear_sync_calls_episodic_add_episode(ps_row, int_row):
    sync_state = MagicMock()
    sync_state.last_synced_at = None

    ledger = MagicMock()
    ledger.get_or_create_sync_state.return_value = sync_state
    ledger.get_ingestion_log.return_value = None
    ledger.try_append_ingestion_and_raw_event.return_value = True

    episodic = MagicMock()
    episodic.add_episode.return_value = "episode-uuid-linear-1"

    settings = MagicMock()
    settings.is_enabled.return_value = True

    container = MagicMock()
    container.settings = settings
    container.ledger.return_value = ledger
    container.episodic = episodic

    ref = MagicMock()
    ref.id = "linear-issue-id-1"
    ref.updated_at = datetime(2025, 1, 2, tzinfo=timezone.utc)

    adapter_instance = MagicMock()
    adapter_instance.iter_issues.return_value = [ref]
    adapter_instance.get_issue.return_value = {
        "id": "linear-issue-id-1",
        "identifier": "ENG-1",
        "title": "Test issue",
        "description": "Body",
        "url": "https://linear.app/x/issue/ENG-1",
        "updatedAt": "2025-01-02T00:00:00.000Z",
        "state": {"name": "Todo"},
        "assignee": None,
        "labels": {"nodes": []},
        "comments": {"nodes": []},
    }

    def query_side(model):
        q = MagicMock()
        fl = MagicMock()
        q.filter.return_value = fl
        if model is ProjectSource:
            fl.first.return_value = ps_row
        elif model is Integration:
            fl.first.return_value = int_row
        else:
            fl.first.return_value = None
        return q

    db = MagicMock()
    db.query.side_effect = query_side

    with (
        patch(
            "integrations.adapters.outbound.linear.linear_sync.build_container_for_session",
            return_value=container,
        ),
        patch(
            "integrations.adapters.outbound.linear.linear_sync.decrypt_token",
            return_value="plain-token",
        ),
        patch(
            "integrations.adapters.outbound.linear.linear_sync.LinearIssueTrackerAdapter",
            return_value=adapter_instance,
        ),
        patch("integrations.adapters.outbound.linear.linear_sync.touch_source_sync"),
    ):
        out = sync_linear_project_source(db, "ps-test-1")

    assert out["status"] == "success"
    assert out["ingested"] == 1
    episodic.add_episode.assert_called_once()
    call_kw = episodic.add_episode.call_args.kwargs
    assert call_kw["pot_id"] == "pot-test-1"
    assert "ENG-1" in call_kw["episode_body"]
    assert call_kw["source_description"] == "Linear Issue ENG-1"
    ledger.update_sync_state_success.assert_called_once()

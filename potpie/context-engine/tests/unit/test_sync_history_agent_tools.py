"""Unit tests for the read_sync_history / write_sync_history agent tools."""

from __future__ import annotations

import pytest

from adapters.outbound.agent_tools.sync_history import build_sync_history_tools
from adapters.outbound.sync_history.filesystem import FileSystemSyncHistoryStore

pytestmark = pytest.mark.unit


class _State:
    def __init__(self, pot_id: str | None) -> None:
        self.pot_id = pot_id


def _tools(store, pot_id="pot-1"):
    builder = build_sync_history_tools(store)
    tools = builder(_State(pot_id))
    return {t.name: getattr(t, "function", t) for t in tools}


def test_builder_surfaces_both_tools(tmp_path) -> None:
    names = set(_tools(FileSystemSyncHistoryStore(tmp_path)))
    assert names == {"read_sync_history", "write_sync_history"}


def test_write_then_read_recovers_latest_cursor(tmp_path) -> None:
    tools = _tools(FileSystemSyncHistoryStore(tmp_path))
    write = tools["write_sync_history"]
    read = tools["read_sync_history"]

    write(
        {
            "source_system": "linear",
            "event_type": "linear_team",
            "team": "ENG",
            "new_cursor": "2026-06-01T00:00:00Z",
            "status": "success",
        }
    )
    write(
        {
            "source_system": "linear",
            "event_type": "linear_team",
            "team": "ENG",
            "new_cursor": "2026-06-02T00:00:00Z",
            "status": "success",
        }
    )

    out = read(source_system="linear", scope="linear_team", key="ENG")
    assert out["count"] == 2
    assert out["latest_cursor"] == "2026-06-02T00:00:00Z"


def test_write_derives_scope_for_jira_project(tmp_path) -> None:
    store = FileSystemSyncHistoryStore(tmp_path)
    tools = _tools(store)
    res = tools["write_sync_history"](
        {
            "source_system": "jira",
            "event_type": "jira_project",
            "project_key": "PROJ",
            "new_cursor": "c1",
        }
    )
    assert res["ok"] is True
    assert res["path"].endswith("jira-project-proj.jsonl")


def test_write_without_scope_fields_reports_error(tmp_path) -> None:
    tools = _tools(FileSystemSyncHistoryStore(tmp_path))
    res = tools["write_sync_history"]({"new_cursor": "c1"})
    assert res["error"] == "missing_scope"


def test_history_is_pot_scoped(tmp_path) -> None:
    store = FileSystemSyncHistoryStore(tmp_path)
    _tools(store, pot_id="pot-a")["write_sync_history"](
        {
            "source_system": "linear",
            "event_type": "linear_team",
            "team": "ENG",
            "new_cursor": "a",
        }
    )
    other = _tools(store, pot_id="pot-b")["read_sync_history"](
        source_system="linear", scope="linear_team", key="ENG"
    )
    assert other["count"] == 0


def test_read_empty_scope_has_null_cursor(tmp_path) -> None:
    tools = _tools(FileSystemSyncHistoryStore(tmp_path))
    out = tools["read_sync_history"](
        source_system="jira", scope="jira_project", key="NEW"
    )
    assert out["count"] == 0
    assert out["latest_cursor"] is None

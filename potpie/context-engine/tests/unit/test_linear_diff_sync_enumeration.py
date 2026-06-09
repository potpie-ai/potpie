"""The Linear list tools honour an explicit diff-sync ``updated_since`` cursor."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from adapters.outbound.connectors.linear.agent_tools import build_linear_tools

pytestmark = pytest.mark.unit


class _State:
    def __init__(self, pot_id: str | None) -> None:
        self.pot_id = pot_id


class _RecordingFetcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def get_issue(self, issue_id, *, pot_id=None):
        return {"id": issue_id}

    def list_issues(self, **kw):
        self.calls.append(("list_issues", kw))
        return []

    def list_projects(self, **kw):
        self.calls.append(("list_projects", kw))
        return []

    def get_project(self, project_id, *, pot_id=None):
        return {"id": project_id}

    def list_documents(self, **kw):
        self.calls.append(("list_documents", kw))
        return []

    def get_document(self, document_id, *, pot_id=None):
        return {"id": document_id}


def _tools(fetcher):
    return {
        t.name: getattr(t, "function", t)
        for t in build_linear_tools(fetcher)(_State("pot-1"))
    }


def test_updated_since_cursor_overrides_backfill_window(monkeypatch) -> None:
    # A wide window would otherwise dominate; the explicit cursor must win.
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_WINDOW_DAYS", "365")
    fetcher = _RecordingFetcher()
    tools = _tools(fetcher)

    cursor = "2026-06-01T00:00:00Z"
    tools["linear_list_issues"](team_id="ENG", updated_since=cursor, limit=50)
    tools["linear_list_projects"](team_id="ENG", updated_since=cursor, limit=50)
    tools["linear_list_documents"](team_id="ENG", updated_since=cursor, limit=50)

    by_name = {name: kw for name, kw in fetcher.calls}
    expected = datetime(2026, 6, 1, tzinfo=timezone.utc)
    for name in ("list_issues", "list_projects", "list_documents"):
        assert by_name[name]["updated_after"] == expected


def test_no_cursor_falls_back_to_backfill_window(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_WINDOW_DAYS", "30")
    fetcher = _RecordingFetcher()
    tools = _tools(fetcher)

    tools["linear_list_issues"](team_id="ENG")

    kw = dict(fetcher.calls)["list_issues"]
    # Window applied (not None) but it is the trailing window, ~30 days back.
    assert kw["updated_after"] is not None
    delta = datetime.now(timezone.utc) - kw["updated_after"]
    assert 29 <= delta.days <= 31


def test_garbage_cursor_falls_back_to_window(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_WINDOW_DAYS", "30")
    fetcher = _RecordingFetcher()
    tools = _tools(fetcher)

    tools["linear_list_issues"](team_id="ENG", updated_since="not-a-date")

    kw = dict(fetcher.calls)["list_issues"]
    assert kw["updated_after"] is not None  # did not crash; used window

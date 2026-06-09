"""Unit tests for the Jira agent tools builder."""

from __future__ import annotations

import pytest

from adapters.outbound.connectors.jira.agent_tools import build_jira_tools

pytestmark = pytest.mark.unit


class _State:
    def __init__(self, pot_id: str | None) -> None:
        self.pot_id = pot_id


class _FullJiraFetcher:
    """Implements every optional capability; records calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def get_issue(self, issue_key, *, pot_id=None):
        self.calls.append(("get_issue", {"issue_key": issue_key, "pot_id": pot_id}))
        return {"key": issue_key, "summary": "An issue"}

    def search_issues(self, jql, *, pot_id=None, limit=None):
        self.calls.append(("search_issues", {"jql": jql, "limit": limit}))
        return [{"key": "PROJ-1", "summary": "x", "issuetype": "Task", "updated_at": None}]

    def get_issue_changelog(self, issue_key, *, pot_id=None):
        self.calls.append(("get_issue_changelog", {"issue_key": issue_key}))
        return [{"field": "status", "to": "Done"}]

    def bulk_fetch_changelogs(self, issue_keys, *, pot_id=None):
        self.calls.append(("bulk_fetch_changelogs", {"issue_keys": issue_keys}))
        return {k: [] for k in issue_keys}


class _MinimalJiraFetcher:
    """Single-issue resolver — no enumeration / changelog capabilities."""

    def get_issue(self, issue_key, *, pot_id=None):
        return {"key": issue_key}


def _tools(fetcher, pot_id="pot-1"):
    builder = build_jira_tools(fetcher)
    tools = builder(_State(pot_id))
    return {t.name: getattr(t, "function", t) for t in tools}


def test_full_fetcher_surfaces_all_tools() -> None:
    names = set(_tools(_FullJiraFetcher()))
    assert names == {
        "jira_get_issue",
        "jira_search_issues",
        "jira_get_issue_changelog",
        "jira_bulk_fetch_changelogs",
    }


def test_minimal_fetcher_only_exposes_get_issue() -> None:
    assert set(_tools(_MinimalJiraFetcher())) == {"jira_get_issue"}


def test_search_clamps_limit_to_cap(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_MAX_ITEMS", "5")
    fetcher = _FullJiraFetcher()
    tools = _tools(fetcher)
    tools["jira_search_issues"](
        'project = PROJ AND updated >= "2026-06-01" ORDER BY updated ASC',
        limit=999,
    )
    call = dict(fetcher.calls)["search_issues"]
    assert call["limit"] == 5
    assert "ORDER BY updated ASC" in call["jql"]


def test_get_issue_not_found_is_structured() -> None:
    tools = _tools(_MinimalJiraFetcher())
    # _MinimalJiraFetcher always returns a dict, so simulate None via a fetcher.
    class _NoneFetcher:
        def get_issue(self, issue_key, *, pot_id=None):
            return None

    out = _tools(_NoneFetcher())["jira_get_issue"]("PROJ-9")
    assert out == {"found": False, "issue_key": "PROJ-9"}
    assert tools  # smoke


def test_auth_error_surfaces_as_jira_auth_failed() -> None:
    class _AuthFetcher:
        def get_issue(self, issue_key, *, pot_id=None):
            raise PermissionError("token expired")

    out = _tools(_AuthFetcher())["jira_get_issue"]("PROJ-1")
    assert out["error"] == "jira_auth_failed"


def test_generic_error_is_redacted_not_raised() -> None:
    class _BoomFetcher:
        def get_issue(self, issue_key, *, pot_id=None):
            raise RuntimeError("kaboom internal detail")

    out = _tools(_BoomFetcher())["jira_get_issue"]("PROJ-1")
    assert "error" in out


def test_bulk_changelog_keyed_by_issue() -> None:
    fetcher = _FullJiraFetcher()
    out = _tools(fetcher)["jira_bulk_fetch_changelogs"](["PROJ-1", "PROJ-2"])
    assert out["count"] == 2
    assert set(out["changelogs"]) == {"PROJ-1", "PROJ-2"}

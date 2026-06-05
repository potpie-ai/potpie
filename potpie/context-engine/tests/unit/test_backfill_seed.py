"""Backfill-seed mechanism: planner flag, bounded window, GitHub enumeration.

Covers the deterministic core of the agent-driven backfill:

- the declarative ``enables_planner`` signal the agent reads to turn the
  pydantic-deep todo/plan tools on for source-attach seeds only;
- the shared ``backfill_window`` window/cap knobs;
- ``PyGithubSourceControl.list_*`` honoring the window short-circuit, the
  item cap, and excluding PRs from the issue listing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from domain.backfill_window import (
    backfill_max_items,
    backfill_window_since,
    clamp_backfill_limit,
)
from domain.event_playbooks import find_playbook, playbooks_enable_planner


# --- planner signal --------------------------------------------------------


def test_backfill_playbooks_enable_planner() -> None:
    gh = find_playbook("github", "repository", "added")
    lin = find_playbook("linear", "linear_team", "added")
    assert gh.enables_planner is True
    assert lin.enables_planner is True
    assert playbooks_enable_planner([gh]) is True
    assert playbooks_enable_planner([lin]) is True


def test_live_event_playbooks_keep_planner_off() -> None:
    for src, et, ac in (
        ("github", "pull_request", "merged"),
        ("github", "issue", "opened"),
        ("manual", "raw_episode", "submit"),
        ("github", "unknown_kind", "whatever"),  # falls back to default
    ):
        pb = find_playbook(src, et, ac)
        assert pb.enables_planner is False, (src, et, ac)
    assert (
        playbooks_enable_planner([find_playbook("github", "pull_request", "merged")])
        is False
    )


def test_planner_on_if_any_playbook_in_batch_enables_it() -> None:
    mixed = [
        find_playbook("github", "pull_request", "merged"),  # off
        find_playbook("github", "repository", "added"),  # on
    ]
    assert playbooks_enable_planner(mixed) is True


# --- bounded window knobs --------------------------------------------------


def test_window_since_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_WINDOW_DAYS", "30")
    now = datetime(2026, 5, 18, tzinfo=timezone.utc)
    since = backfill_window_since(now)
    assert since == now - timedelta(days=30)


def test_window_disabled_when_non_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_WINDOW_DAYS", "0")
    assert backfill_window_since() is None
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_WINDOW_DAYS", "-5")
    assert backfill_window_since() is None


def test_clamp_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_MAX_ITEMS", "50")
    assert backfill_max_items() == 50
    assert clamp_backfill_limit(None) == 50  # default to ceiling
    assert clamp_backfill_limit(10) == 10
    assert clamp_backfill_limit(999) == 50  # clamped to ceiling
    assert clamp_backfill_limit(0) == 1  # floor
    assert clamp_backfill_limit("garbage") == 50  # type-safe fallback


def test_bad_env_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_MAX_ITEMS", "not-an-int")
    assert backfill_max_items() == 300


# --- GitHub enumeration (window short-circuit, cap, PR exclusion) ----------


class _FakeUser:
    def __init__(self, login: str) -> None:
        self.login = login


class _FakePull:
    def __init__(self, number: int, updated: datetime, merged: bool = False) -> None:
        self.number = number
        self.title = f"PR {number}"
        self.state = "closed" if merged else "open"
        self.merged_at = updated if merged else None
        self.created_at = updated
        self.updated_at = updated
        self.html_url = f"https://x/pr/{number}"
        self.user = _FakeUser("octocat")


class _FakeLabel:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeIssue:
    def __init__(
        self,
        number: int,
        updated: datetime,
        is_pr: bool = False,
        labels: list[str] | None = None,
    ) -> None:
        self.number = number
        self.title = f"Issue {number}"
        self.body = f"Body {number}"
        self.state = "open"
        self.created_at = updated
        self.updated_at = updated
        self.html_url = f"https://x/i/{number}"
        self.user = _FakeUser("octocat")
        self.comments = 0
        self.labels = [_FakeLabel(name) for name in (labels or [])]
        # PyGithub sets .pull_request on issues that are actually PRs.
        self.pull_request = object() if is_pr else None


class _FakeRepo:
    def __init__(self, pulls: list, issues: list) -> None:
        self._pulls = pulls
        self._issues = issues

    def get_pulls(self, **_kw):
        return list(self._pulls)

    def get_issues(self, **_kw):
        return list(self._issues)

    def get_issue(self, issue_number: int):
        for issue in self._issues:
            if issue.number == issue_number:
                return issue
        raise KeyError(issue_number)


class _FakeGithub:
    def __init__(self, repo: _FakeRepo) -> None:
        self._repo = repo

    def get_repo(self, _name: str) -> _FakeRepo:
        return self._repo


def _client(pulls=None, issues=None):
    from adapters.outbound.connectors.github.api_client import PyGithubSourceControl

    return PyGithubSourceControl(_FakeGithub(_FakeRepo(pulls or [], issues or [])))


def test_list_pull_requests_caps_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_MAX_ITEMS", "2")
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_WINDOW_DAYS", "0")  # no window
    now = datetime.now(timezone.utc)
    pulls = [_FakePull(n, now - timedelta(days=n)) for n in range(1, 6)]
    out = _client(pulls=pulls).list_pull_requests("o/r")
    assert [p["number"] for p in out] == [1, 2]
    assert out[0]["merged"] is False


def test_list_pull_requests_window_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_WINDOW_DAYS", "10")
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_MAX_ITEMS", "100")
    now = datetime.now(timezone.utc)
    # Newest-first: 3 inside the 10-day window, then one well outside it.
    pulls = [
        _FakePull(1, now - timedelta(days=1)),
        _FakePull(2, now - timedelta(days=3)),
        _FakePull(3, now - timedelta(days=9)),
        _FakePull(4, now - timedelta(days=40)),
        _FakePull(5, now - timedelta(days=80)),
    ]
    out = _client(pulls=pulls).list_pull_requests("o/r")
    assert [p["number"] for p in out] == [1, 2, 3]  # stops at #4


def test_list_issues_excludes_pull_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_WINDOW_DAYS", "0")
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_MAX_ITEMS", "100")
    now = datetime.now(timezone.utc)
    issues = [
        _FakeIssue(1, now),
        _FakeIssue(2, now, is_pr=True),  # a PR masquerading as an issue
        _FakeIssue(3, now),
    ]
    out = _client(issues=issues).list_issues("o/r")
    assert [i["number"] for i in out] == [1, 3]


def test_issue_reads_include_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_WINDOW_DAYS", "0")
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_MAX_ITEMS", "100")
    now = datetime.now(timezone.utc)
    client = _client(issues=[_FakeIssue(1, now, labels=["bug", "question"])])

    listed = client.list_issues("o/r")
    hydrated = client.get_issue("o/r", 1)

    assert listed[0]["labels"] == [{"name": "bug"}, {"name": "question"}]
    assert hydrated["labels"] == [{"name": "bug"}, {"name": "question"}]


def test_list_handles_naive_datetimes(monkeypatch: pytest.MonkeyPatch) -> None:
    """PyGithub historically returns naive UTC; the window math must not crash."""
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_WINDOW_DAYS", "10")
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_MAX_ITEMS", "100")
    naive_now = datetime.utcnow()
    pulls = [
        _FakePull(1, naive_now - timedelta(days=1)),
        _FakePull(2, naive_now - timedelta(days=40)),
    ]
    out = _client(pulls=pulls).list_pull_requests("o/r")
    assert [p["number"] for p in out] == [1]


# --- Linear: projects + documents playbook + capability-guarded tools ------


def test_linear_playbook_covers_projects_and_documents() -> None:
    pb = find_playbook("linear", "linear_team", "added")
    assert pb.enables_planner is True
    for hint in (
        "linear_list_projects",
        "linear_get_project",
        "linear_list_documents",
        "linear_get_document",
        "linear_list_issues",
    ):
        assert hint in pb.tool_hints
    blob = (pb.available_data + pb.extract + pb.skip).lower()
    assert "project" in blob and "document" in blob
    assert "linear:project:" in pb.extract
    assert "linear:document:" in pb.extract


class _State:
    def __init__(self, pot_id: str | None) -> None:
        self.pot_id = pot_id


class _FullLinearFetcher:
    """Implements every optional enumeration capability; records calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def get_issue(self, issue_id, *, pot_id=None):
        return {"id": issue_id}

    def list_issues(self, **kw):
        self.calls.append(("list_issues", kw))
        return [{"id": "i1", "identifier": "ENG-1", "updated_at": None}]

    def list_projects(self, **kw):
        self.calls.append(("list_projects", kw))
        return [{"id": "p1", "name": "Proj", "updated_at": None}]

    def get_project(self, project_id, *, pot_id=None):
        return {"id": project_id, "name": "Proj"}

    def list_documents(self, **kw):
        self.calls.append(("list_documents", kw))
        return [{"id": "d1", "title": "Spec", "updated_at": None}]

    def get_document(self, document_id, *, pot_id=None):
        return {"id": document_id, "title": "Spec"}


class _MinimalLinearFetcher:
    """Single-issue resolver — no enumeration capabilities at all."""

    def get_issue(self, issue_id, *, pot_id=None):
        return {"id": issue_id}


def _linear_tool_names(fetcher) -> list[str]:
    from adapters.outbound.connectors.linear.agent_tools import build_linear_tools

    builder = build_linear_tools(fetcher)
    tools = builder(_State("pot-1"))
    return [t.name for t in tools]


def test_full_fetcher_surfaces_all_linear_backfill_tools() -> None:
    names = _linear_tool_names(_FullLinearFetcher())
    for expected in (
        "linear_get_issue",
        "linear_list_issues",
        "linear_list_projects",
        "linear_get_project",
        "linear_list_documents",
        "linear_get_document",
    ):
        assert expected in names


def test_minimal_fetcher_only_exposes_get_issue() -> None:
    names = _linear_tool_names(_MinimalLinearFetcher())
    assert names == ["linear_get_issue"]
    for absent in (
        "linear_list_issues",
        "linear_list_projects",
        "linear_list_documents",
        "linear_get_document",
    ):
        assert absent not in names


def test_linear_list_tools_apply_window_and_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_MAX_ITEMS", "7")
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_WINDOW_DAYS", "30")
    from adapters.outbound.connectors.linear.agent_tools import build_linear_tools

    fetcher = _FullLinearFetcher()
    tools = {
        t.name: getattr(t, "function", t)
        for t in build_linear_tools(fetcher)(_State("pot-1"))
    }
    tools["linear_list_projects"]()
    tools["linear_list_documents"]()
    tools["linear_list_issues"]()

    by_name = {name: kw for name, kw in fetcher.calls}
    for name in ("list_projects", "list_documents", "list_issues"):
        assert by_name[name]["limit"] == 7  # clamped to the cap
        assert by_name[name]["updated_after"] is not None  # window applied

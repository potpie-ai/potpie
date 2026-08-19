"""Happy-path dispatch for the GitLab agent tools.

Companion to ``test_gitlab_agent_tools_repo_binding.py``, which pins the
C-5 security guard. Those binding tests exercise the guard's fail-closed
path, where it short-circuits *before* reaching the port — so the tool
bodies themselves never run there. These tests drive every tool body
through a fake implementing the full ``GitLabReadPort``, asserting it
resolves the port and forwards arguments to the matching method, and that
a port *missing* a method degrades to a safe error dict (the production
failure shape) instead of raising into the agent.
"""

from __future__ import annotations

import inspect

from potpie_context_engine.adapters.outbound.connectors.gitlab.agent_tools import (
    build_gitlab_tools,
)
from potpie_context_engine.adapters.outbound.connectors.gitlab.api_client import (
    GitLabRestSourceControl,
)


class _State:
    def __init__(self, pot_id: str | None = "pot-1") -> None:
        self.pot_id = pot_id


class _RecordingPort:
    """Implements the full GitLabReadPort surface; records forwarded calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def _rec(self, name: str, *args: object, **kwargs: object) -> None:
        self.calls.append((name, args, kwargs))

    def get_merge_request(self, project, iid, include_diff=False):
        self._rec("get_merge_request", project, iid, include_diff=include_diff)
        return {"iid": iid}

    def get_merge_request_commits(self, project, iid):
        self._rec("get_merge_request_commits", project, iid)
        return [{"sha": "abc"}]

    def get_merge_request_discussions(self, project, iid, limit=100):
        self._rec("get_merge_request_discussions", project, iid, limit=limit)
        return [{"id": 1}]

    def get_merge_request_notes(self, project, iid, limit=50):
        self._rec("get_merge_request_notes", project, iid, limit=limit)
        return [{"id": 2}]

    def get_merge_request_approvals(self, project, iid):
        self._rec("get_merge_request_approvals", project, iid)
        return {"available": True, "approved_by": ["ravi"]}

    def get_merge_request_state_events(self, project, iid):
        self._rec("get_merge_request_state_events", project, iid)
        return [{"state": "merged"}]

    def get_merge_request_closes_issues(self, project, iid):
        self._rec("get_merge_request_closes_issues", project, iid)
        return [{"iid": 12}]

    def get_issue(self, project, iid):
        self._rec("get_issue", project, iid)
        return {"iid": iid}

    def get_issue_notes(self, project, iid, limit=50):
        self._rec("get_issue_notes", project, iid, limit=limit)
        return [{"id": 3}]

    def get_issue_links(self, project, iid):
        self._rec("get_issue_links", project, iid)
        return {"related_merge_requests": [], "closed_by_merge_requests": []}

    def iter_merged_merge_requests(self, project):
        self._rec("iter_merged_merge_requests", project)
        return iter(())

    def list_merge_requests(self, project, *, state="all", limit=None):
        self._rec("list_merge_requests", project, state=state, limit=limit)
        return [{"iid": 1}, {"iid": 2}]

    def list_issues(self, project, *, state="all", limit=None):
        self._rec("list_issues", project, state=state, limit=limit)
        return [{"iid": 3}]


def _tools(port, *, allowed=frozenset({"acme/widgets"})):
    builder = build_gitlab_tools(
        lambda _p: port,
        allowed_projects_for_pot=lambda _pot: set(allowed),
    )
    return {t.name: t.function for t in builder(_State())}


def test_every_tool_forwards_to_its_port_method():
    port = _RecordingPort()
    tools = _tools(port)
    tools["gitlab_get_merge_request"]("acme/widgets", 7, True)
    tools["gitlab_get_merge_request_commits"]("acme/widgets", 7)
    tools["gitlab_get_merge_request_discussions"]("acme/widgets", 7, 25)
    tools["gitlab_get_merge_request_notes"]("acme/widgets", 7, 10)
    tools["gitlab_get_merge_request_approvals"]("acme/widgets", 7)
    tools["gitlab_get_merge_request_state_events"]("acme/widgets", 7)
    tools["gitlab_get_merge_request_closes_issues"]("acme/widgets", 7)
    tools["gitlab_get_issue"]("acme/widgets", 12)
    tools["gitlab_get_issue_notes"]("acme/widgets", 12, 5)
    tools["gitlab_get_issue_links"]("acme/widgets", 12)
    assert [c[0] for c in port.calls] == [
        "get_merge_request",
        "get_merge_request_commits",
        "get_merge_request_discussions",
        "get_merge_request_notes",
        "get_merge_request_approvals",
        "get_merge_request_state_events",
        "get_merge_request_closes_issues",
        "get_issue",
        "get_issue_notes",
        "get_issue_links",
    ]
    assert port.calls[0][2] == {"include_diff": True}
    assert port.calls[2][2] == {"limit": 25}
    assert port.calls[8][2] == {"limit": 5}


def test_string_iid_from_the_model_is_coerced():
    """The model routinely hands back a stringified number."""
    port = _RecordingPort()
    tools = _tools(port)
    tools["gitlab_get_merge_request"]("acme/widgets", "7")
    assert port.calls[0][1] == ("acme/widgets", 7)


def test_list_merge_requests_wraps_results_with_a_count():
    port = _RecordingPort()
    out = _tools(port)["gitlab_list_merge_requests"]("acme/widgets", "merged", 10)
    assert out == {
        "project": "acme/widgets",
        "count": 2,
        "merge_requests": [{"iid": 1}, {"iid": 2}],
    }
    assert port.calls[0] == (
        "list_merge_requests",
        ("acme/widgets",),
        {"state": "merged", "limit": 10},
    )


def test_list_issues_wraps_results_with_a_count():
    port = _RecordingPort()
    out = _tools(port)["gitlab_list_issues"]("acme/widgets")
    assert out == {"project": "acme/widgets", "count": 1, "issues": [{"iid": 3}]}


def test_port_missing_a_method_degrades_to_an_error_dict():
    class _Partial:
        pass

    out = _tools(_Partial())["gitlab_get_merge_request_approvals"]("acme/widgets", 1)
    assert "error" in out
    assert "unknown_project" not in out["error"]


def test_port_raising_degrades_to_an_error_dict():
    class _Boom:
        def get_issue(self, project, iid):
            raise RuntimeError("upstream 500 for glpat-ABCDEFGHIJKLMNOPQRSTUV")

    out = _tools(_Boom())["gitlab_get_issue"]("acme/widgets", 1)
    assert "error" in out
    # The redaction pass must run before the message reaches the agent.
    assert "glpat-" not in out["error"]


def test_every_tool_name_matches_a_read_port_method():
    """A tool the adapter cannot serve would fail only at agent runtime."""
    tools = _tools(_RecordingPort())
    adapter_methods = {
        name
        for name, _ in inspect.getmembers(
            GitLabRestSourceControl, predicate=inspect.isfunction
        )
        if not name.startswith("_")
    }
    for tool_name in tools:
        assert tool_name.startswith("gitlab_")
        assert tool_name[len("gitlab_") :] in adapter_methods, tool_name


def test_tool_schema_signature_is_preserved_through_the_guard():
    tools = _tools(_RecordingPort())
    sig = inspect.signature(tools["gitlab_get_merge_request"])
    assert list(sig.parameters) == ["project", "iid", "include_diff"]

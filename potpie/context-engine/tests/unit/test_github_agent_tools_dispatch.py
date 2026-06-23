"""Happy-path dispatch for the GitHub backfill tools.

Companion to ``test_github_agent_tools_repo_binding.py``, which pins the C-5
security guard. Those binding tests call the *list* tools only on the
fail-closed path, where the guard short-circuits *before* reaching the port
(``assert calls == []``) — so the tool bodies that call
``port.list_pull_requests`` / ``port.list_issues`` had **zero** execution.
That blind spot is exactly what let ``CodeProviderSourceControl`` ship without
those methods.

These tests drive each list tool's body through a fake implementing the full
``GitHubReadPort``, asserting it resolves the port and forwards args to the
matching method — and that a port *missing* the method degrades to a safe
error dict (the production failure shape) instead of raising into the agent.
"""

from __future__ import annotations

from potpie.context_engine.adapters.outbound.connectors.github.agent_tools import build_github_tools


class _State:
    def __init__(self, pot_id: str | None = "pot-1") -> None:
        self.pot_id = pot_id


class _RecordingPort:
    """Implements the full GitHubReadPort surface; records forwarded calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def _rec(self, name: str, *args: object, **kwargs: object) -> None:
        self.calls.append((name, args, kwargs))

    def get_pull_request(self, repo_name, pr_number, include_diff=False):
        self._rec("get_pull_request", repo_name, pr_number, include_diff=include_diff)
        return {"number": pr_number}

    def get_pull_request_commits(self, repo_name, pr_number):
        self._rec("get_pull_request_commits", repo_name, pr_number)
        return [{"sha": "abc"}]

    def get_pull_request_review_comments(self, repo_name, pr_number, limit=100):
        self._rec("get_pull_request_review_comments", repo_name, pr_number, limit=limit)
        return [{"id": 1}]

    def get_pull_request_issue_comments(self, repo_name, pr_number, limit=50):
        self._rec("get_pull_request_issue_comments", repo_name, pr_number, limit=limit)
        return [{"id": 2}]

    def get_issue(self, repo_name, issue_number):
        self._rec("get_issue", repo_name, issue_number)
        return {"number": issue_number}

    def iter_closed_pulls(self, repo_name):
        self._rec("iter_closed_pulls", repo_name)
        return iter(())

    def list_pull_requests(self, repo_name, *, state="all", limit=None):
        self._rec("list_pull_requests", repo_name, state=state, limit=limit)
        return [{"number": 1}, {"number": 2}]

    def list_issues(self, repo_name, *, state="all", limit=None):
        self._rec("list_issues", repo_name, state=state, limit=limit)
        return [{"number": 3}]


def _tools(port, *, allowed=frozenset({"acme/widgets"})):
    builder = build_github_tools(
        lambda _r: port,
        allowed_repos_for_pot=lambda _p: set(allowed),
    )
    return {t.name: t.function for t in builder(_State())}


def test_list_pull_requests_dispatches_to_port():
    port = _RecordingPort()
    out = _tools(port)["github_list_pull_requests"](
        "acme/widgets", state="open", limit=5
    )
    assert port.calls == [
        ("list_pull_requests", ("acme/widgets",), {"state": "open", "limit": 5})
    ]
    assert out == {
        "repo_name": "acme/widgets",
        "count": 2,
        "pull_requests": [{"number": 1}, {"number": 2}],
    }


def test_list_issues_dispatches_to_port():
    port = _RecordingPort()
    out = _tools(port)["github_list_issues"]("acme/widgets", state="closed", limit=3)
    assert port.calls == [
        ("list_issues", ("acme/widgets",), {"state": "closed", "limit": 3})
    ]
    assert out == {
        "repo_name": "acme/widgets",
        "count": 1,
        "issues": [{"number": 3}],
    }


def test_list_pull_requests_missing_method_degrades_to_error():
    """The exact production failure: a port lacking list_pull_requests.

    The tool must catch the AttributeError and return a safe error dict rather
    than propagate it into the agent loop (this is the ``{"error": "...has no
    attribute 'list_pull_requests'"}`` the user originally saw).
    """

    class _PartialPort:  # implements nothing on the port
        pass

    out = _tools(_PartialPort())["github_list_pull_requests"]("acme/widgets")
    assert "error" in out
    assert "pull_requests" not in out

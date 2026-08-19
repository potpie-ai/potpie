"""GitLab REST read client: encoding, mapping, windowing, and failure modes.

The transport is injected, so every case here pins the exact request the
client issues and the exact shape it hands back — the vocabulary the
playbooks and graph mutations depend on.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote

import pytest

from potpie_context_engine.adapters.outbound.connectors.gitlab.api_client import (
    GitLabApiError,
    GitLabRestSourceControl,
)


class _Response:
    def __init__(self, status_code: int, payload=None, *, text: str | None = None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        if self._payload is _INVALID:
            raise ValueError("not json")
        return self._payload


_INVALID = object()


class _FakeHttp:
    """Records requests and replays queued responses by URL suffix."""

    def __init__(self, routes: dict[str, object], default=None):
        self.routes = routes
        self.default = default
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        for suffix, response in self.routes.items():
            if url.endswith(suffix) or suffix in url:
                if callable(response):
                    return response(url, kwargs)
                return response
        if self.default is not None:
            return self.default
        return _Response(404, {"message": "404 Not Found"})

    def post(self, url, **kwargs):
        return self.get(url, **kwargs)

    def close(self):
        pass


def _client(routes, default=None, **kwargs):
    return GitLabRestSourceControl(
        "https://gitlab.corp.example",
        "glpat-secret-token-value-1234567890",
        http=_FakeHttp(routes, default),
        **kwargs,
    )


# ----------------------------------------------------------------------
# Project path encoding
# ----------------------------------------------------------------------
def test_nested_group_path_is_percent_encoded():
    http = _FakeHttp({"/merge_requests/5": _Response(200, {"iid": 5})})
    client = GitLabRestSourceControl("https://gitlab.corp.example", "t", http=http)
    client.get_merge_request("acme/platform/widgets", 5)
    url = http.calls[0][0]
    assert "acme%2Fplatform%2Fwidgets" in url
    assert unquote(url).endswith("/projects/acme/platform/widgets/merge_requests/5")


def test_numeric_project_id_is_passed_through():
    http = _FakeHttp({"/merge_requests/1": _Response(200, {"iid": 1})})
    client = GitLabRestSourceControl("https://gitlab.corp.example", "t", http=http)
    client.get_merge_request("42", 1)
    assert "/projects/42/merge_requests/1" in http.calls[0][0]


def test_api_base_honours_self_managed_host():
    client = _client({})
    assert client.instance_url == "https://gitlab.corp.example"
    assert client.instance_host == "gitlab.corp.example"


def test_private_token_header_is_used_not_bearer():
    http = _FakeHttp({"/issues/1": _Response(200, {"iid": 1})})
    client = GitLabRestSourceControl(
        "https://gitlab.corp.example", "glpat-x", http=http
    )
    client.get_issue("acme/api", 1)
    headers = http.calls[0][1]["headers"]
    assert headers["PRIVATE-TOKEN"] == "glpat-x"
    assert "Authorization" not in headers


# ----------------------------------------------------------------------
# Merge request mapping
# ----------------------------------------------------------------------
_MR = {
    "iid": 7,
    "title": "Fix login timeout",
    "description": "Why: the session cache expired early.",
    "state": "merged",
    "created_at": "2026-01-02T03:04:05Z",
    "updated_at": "2026-01-03T03:04:05Z",
    "merged_at": "2026-01-03T03:04:05Z",
    "source_branch": "fix/login-timeout",
    "target_branch": "main",
    "web_url": "https://gitlab.corp.example/acme/api/-/merge_requests/7",
    "author": {"username": "dana"},
    "labels": ["bug", "backend"],
    "milestone": {"title": "v2"},
    "draft": False,
    "detailed_merge_status": "mergeable",
    "reviewers": [{"username": "ravi"}],
    "assignees": [{"username": "dana"}],
    "merged_by": {"username": "ravi"},
}


def test_merge_request_maps_onto_github_vocabulary():
    client = _client({"/merge_requests/7": _Response(200, _MR)})
    mr = client.get_merge_request("acme/api", 7)
    assert mr["number"] == 7 and mr["iid"] == 7
    assert mr["title"] == "Fix login timeout"
    assert mr["body"] == "Why: the session cache expired early."
    assert mr["head_branch"] == "fix/login-timeout"
    assert mr["base_branch"] == "main"
    assert mr["url"].endswith("/merge_requests/7")
    assert mr["author"] == "dana"
    assert mr["merged"] is True
    # GitLab returns labels as bare strings; the port normalizes to objects.
    assert mr["labels"] == [{"name": "bug"}, {"name": "backend"}]
    assert mr["milestone"] == {"title": "v2"}
    assert mr["reviewers"] == ["ravi"]


def test_merge_request_without_diff_does_not_fetch_diffs():
    http = _FakeHttp({"/merge_requests/7": _Response(200, _MR)})
    client = GitLabRestSourceControl("https://gitlab.corp.example", "t", http=http)
    client.get_merge_request("acme/api", 7)
    assert len(http.calls) == 1


def test_include_diff_maps_file_status():
    diffs = [
        {"new_path": "a.py", "old_path": "a.py", "diff": "@@ -1 +1 @@\n-x\n+y\n"},
        {"new_path": "b.py", "old_path": "b.py", "new_file": True, "diff": "+new"},
        {"new_path": "c.py", "old_path": "c.py", "deleted_file": True, "diff": "-gone"},
        {"new_path": "e.py", "old_path": "d.py", "renamed_file": True, "diff": ""},
    ]
    client = _client(
        {
            "/merge_requests/7/diffs": _Response(200, diffs),
            "/merge_requests/7": _Response(200, _MR),
        }
    )
    mr = client.get_merge_request("acme/api", 7, include_diff=True)
    assert [f["filename"] for f in mr["files"]] == ["a.py", "b.py", "c.py", "e.py"]
    assert [f["status"] for f in mr["files"]] == [
        "modified",
        "added",
        "removed",
        "renamed",
    ]
    assert mr["files"][3]["previous_filename"] == "d.py"


# ----------------------------------------------------------------------
# Review history
# ----------------------------------------------------------------------
_DISCUSSIONS = [
    {
        "id": "disc-1",
        "notes": [
            {
                "id": 11,
                "type": "DiffNote",
                "body": "this drops the retry",
                "author": {"username": "ravi"},
                "system": False,
                "resolvable": True,
                "resolved": True,
                "resolved_by": {"username": "dana"},
                "created_at": "2026-01-02T10:00:00Z",
                "position": {"new_path": "auth/session.py", "new_line": 42},
            }
        ],
    },
    {
        "id": "disc-2",
        "notes": [
            {
                "id": 12,
                "system": True,
                "body": "approved this merge request",
                "author": {"username": "ravi"},
            }
        ],
    },
]


def test_discussions_expose_file_and_line_for_diff_notes():
    client = _client({"/discussions": _Response(200, _DISCUSSIONS)})
    notes = client.get_merge_request_discussions("acme/api", 7)
    assert len(notes) == 1  # the system note is not inline review
    note = notes[0]
    assert note["path"] == "auth/session.py"
    assert note["line"] == 42
    assert note["discussion_id"] == "disc-1"
    assert note["resolved"] is True
    assert note["resolved_by"] == "dana"
    assert note["user"] == {"login": "ravi"}


def test_discussions_respect_limit():
    many = [
        {"id": f"d{i}", "notes": [{"id": i, "body": "x", "position": {}}]}
        for i in range(10)
    ]
    client = _client({"/discussions": _Response(200, many)})
    assert len(client.get_merge_request_discussions("acme/api", 7, limit=3)) == 3


def test_notes_keep_system_entries_as_the_review_trail():
    payload = [
        {
            "id": 1,
            "body": "looks good",
            "author": {"username": "ravi"},
            "system": False,
        },
        {
            "id": 2,
            "body": "approved this merge request",
            "author": {"username": "ravi"},
            "system": True,
        },
    ]
    client = _client({"/notes": _Response(200, payload)})
    notes = client.get_merge_request_notes("acme/api", 7)
    assert [n["system"] for n in notes] == [False, True]
    assert notes[1]["body"] == "approved this merge request"


def test_notes_request_ascending_order():
    http = _FakeHttp({"/notes": _Response(200, [])})
    client = GitLabRestSourceControl("https://gitlab.corp.example", "t", http=http)
    client.get_merge_request_notes("acme/api", 7)
    assert http.calls[0][1]["params"]["sort"] == "asc"


def test_approvals_maps_approver_usernames():
    payload = {
        "approved": True,
        "approvals_required": 2,
        "approvals_left": 0,
        "approved_by": [{"user": {"username": "ravi"}}, {"user": {"username": "kim"}}],
    }
    client = _client({"/approvals": _Response(200, payload)})
    out = client.get_merge_request_approvals("acme/api", 7)
    assert out == {
        "available": True,
        "approved": True,
        "approvals_required": 2,
        "approvals_left": 0,
        "approved_by": ["ravi", "kim"],
    }


@pytest.mark.parametrize("status", [403, 404])
def test_approvals_unavailable_is_reported_not_raised(status):
    """Not every CE instance exposes /approvals — degrade, don't fail."""
    client = _client({"/approvals": _Response(status, {"message": "no"})})
    out = client.get_merge_request_approvals("acme/api", 7)
    assert out["available"] is False
    assert out["approved_by"] == []


def test_state_events_and_closes_issues_degrade_to_empty():
    client = _client({}, default=_Response(404, {"message": "gone"}))
    assert client.get_merge_request_state_events("acme/api", 7) == []
    assert client.get_merge_request_closes_issues("acme/api", 7) == []


def test_review_history_falls_back_to_rest_when_graphql_raises():
    class _BrokenGraphQL:
        def merge_request_review_history(self, project, iid):
            raise RuntimeError("schema mismatch")

    client = _client(
        {
            "/approvals": _Response(200, {"approved": False, "approved_by": []}),
            "/discussions": _Response(200, _DISCUSSIONS),
            "/notes": _Response(200, [{"id": 1, "body": "hi", "system": False}]),
        },
        graphql=_BrokenGraphQL(),
    )
    out = client.get_merge_request_review_history("acme/api", 7)
    assert out["transport"] == "rest"
    assert out["approvals"]["available"] is True
    assert len(out["discussions"]) == 1


def test_review_history_uses_graphql_when_it_answers():
    class _GoodGraphQL:
        def merge_request_review_history(self, project, iid):
            return {"approvals": {"available": True}, "discussions": [], "notes": []}

    client = _client({}, graphql=_GoodGraphQL())
    out = client.get_merge_request_review_history("acme/api", 7)
    assert out["transport"] == "graphql"


# ----------------------------------------------------------------------
# Issues
# ----------------------------------------------------------------------
def test_issue_maps_task_tracking_fields():
    payload = {
        "iid": 12,
        "title": "Login times out",
        "description": "repro: wait 5m",
        "state": "opened",
        "created_at": "2026-02-01T00:00:00Z",
        "web_url": "https://gitlab.corp.example/acme/api/-/issues/12",
        "author": {"username": "sam"},
        "labels": ["bug"],
        "milestone": {"title": "v2"},
        "assignees": [{"username": "dana"}],
        "due_date": "2026-03-01",
        "weight": 3,
        "issue_type": "issue",
        "time_stats": {"time_estimate": 3600},
        "user_notes_count": 4,
    }
    client = _client({"/issues/12": _Response(200, payload)})
    issue = client.get_issue("acme/api", 12)
    assert issue["number"] == 12
    assert issue["body"] == "repro: wait 5m"
    assert issue["labels"] == [{"name": "bug"}]
    assert issue["assignees"] == ["dana"]
    assert issue["due_date"] == "2026-03-01"
    assert issue["weight"] == 3
    assert issue["comments"] == 4


def test_issue_links_merge_related_and_closing_merge_requests():
    client = _client(
        {
            "/related_merge_requests": _Response(
                200, [{"iid": 5, "title": "wip", "state": "opened", "web_url": "u1"}]
            ),
            "/closed_by": _Response(
                200, [{"iid": 6, "title": "fix", "state": "merged", "web_url": "u2"}]
            ),
        }
    )
    links = client.get_issue_links("acme/api", 12)
    assert [m["iid"] for m in links["related_merge_requests"]] == [5]
    assert [m["iid"] for m in links["closed_by_merge_requests"]] == [6]


# ----------------------------------------------------------------------
# Enumeration bounds
# ----------------------------------------------------------------------
def _mr_page(count, *, start=1, updated="2026-01-01T00:00:00Z"):
    return [
        {
            "iid": start + i,
            "title": f"mr {start + i}",
            "state": "merged",
            "merged_at": updated,
            "updated_at": updated,
            "web_url": f"https://gitlab.corp.example/acme/api/-/merge_requests/{start + i}",
            "author": {"username": "dana"},
        }
        for i in range(count)
    ]


def test_list_merge_requests_pushes_window_to_the_server(monkeypatch):
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_WINDOW_DAYS", "30")
    http = _FakeHttp({"/merge_requests": _Response(200, _mr_page(2))})
    client = GitLabRestSourceControl("https://gitlab.corp.example", "t", http=http)
    client.list_merge_requests("acme/api")
    params = http.calls[0][1]["params"]
    assert "updated_after" in params
    cutoff = datetime.fromisoformat(params["updated_after"])
    expected = datetime.now(timezone.utc) - timedelta(days=30)
    assert abs((cutoff - expected).total_seconds()) < 120
    assert params["order_by"] == "updated_at" and params["sort"] == "desc"


def test_list_merge_requests_omits_window_when_disabled(monkeypatch):
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_WINDOW_DAYS", "0")
    http = _FakeHttp({"/merge_requests": _Response(200, _mr_page(1))})
    client = GitLabRestSourceControl("https://gitlab.corp.example", "t", http=http)
    client.list_merge_requests("acme/api")
    assert "updated_after" not in http.calls[0][1]["params"]


def test_list_merge_requests_clamps_to_the_item_cap(monkeypatch):
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_MAX_ITEMS", "3")
    http = _FakeHttp({"/merge_requests": _Response(200, _mr_page(100))})
    client = GitLabRestSourceControl("https://gitlab.corp.example", "t", http=http)
    out = client.list_merge_requests("acme/api", limit=50)
    assert len(out) == 3
    # A caller-supplied limit above the cap must not widen the page request.
    assert http.calls[0][1]["params"]["per_page"] == "3"


def test_list_merge_requests_stops_on_a_short_page(monkeypatch):
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_MAX_ITEMS", "300")
    http = _FakeHttp({"/merge_requests": _Response(200, _mr_page(4))})
    client = GitLabRestSourceControl("https://gitlab.corp.example", "t", http=http)
    out = client.list_merge_requests("acme/api")
    assert len(out) == 4
    assert len(http.calls) == 1  # short page ends the walk


def test_list_merge_requests_pages_until_the_cap(monkeypatch):
    """GitLab's ``page`` is an offset in units of ``per_page``.

    The server is modelled faithfully here: a walk that shrank per_page on
    page 2 would re-serve rows it already yielded and never reach the tail,
    so the iids must come back strictly increasing with no repeats.
    """
    monkeypatch.setenv("CONTEXT_ENGINE_BACKFILL_MAX_ITEMS", "150")
    total = 400
    seen_per_page: set[int] = set()

    def _serve(_url, kwargs):
        page = int(kwargs["params"]["page"])
        per_page = int(kwargs["params"]["per_page"])
        seen_per_page.add(per_page)
        offset = (page - 1) * per_page
        remaining = max(0, total - offset)
        return _Response(200, _mr_page(min(per_page, remaining), start=offset + 1))

    http = _FakeHttp({"/merge_requests": _serve})
    client = GitLabRestSourceControl("https://gitlab.corp.example", "t", http=http)
    out = client.list_merge_requests("acme/api")
    assert [m["iid"] for m in out] == list(range(1, 151))
    assert seen_per_page == {100}  # constant across the walk
    assert len(http.calls) == 2


@pytest.mark.parametrize(
    "requested,expected",
    [
        ("all", "all"),
        ("open", "opened"),
        ("opened", "opened"),
        ("closed", "closed"),
        ("merged", "merged"),
        ("nonsense", "all"),
    ],
)
def test_merge_request_state_vocabulary_is_translated(requested, expected):
    http = _FakeHttp({"/merge_requests": _Response(200, [])})
    client = GitLabRestSourceControl("https://gitlab.corp.example", "t", http=http)
    client.list_merge_requests("acme/api", state=requested)
    assert http.calls[0][1]["params"]["state"] == expected


def test_list_issues_returns_compact_refs():
    payload = [
        {
            "iid": 3,
            "title": "flaky test",
            "state": "opened",
            "updated_at": "2026-01-01T00:00:00Z",
            "web_url": "u",
            "author": {"username": "sam"},
            "labels": ["bug"],
            "user_notes_count": 2,
        }
    ]
    client = _client({"/issues": _Response(200, payload)})
    out = client.list_issues("acme/api")
    assert out == [
        {
            "number": 3,
            "iid": 3,
            "title": "flaky test",
            "state": "opened",
            "created_at": None,
            "updated_at": "2026-01-01T00:00:00Z",
            "url": "u",
            "author": "sam",
            "labels": [{"name": "bug"}],
            "milestone": None,
            "comments": 2,
        }
    ]


def test_iter_merged_merge_requests_only_asks_for_merged():
    http = _FakeHttp({"/merge_requests": _Response(200, _mr_page(2))})
    client = GitLabRestSourceControl("https://gitlab.corp.example", "t", http=http)
    out = list(client.iter_merged_merge_requests("acme/api"))
    assert http.calls[0][1]["params"]["state"] == "merged"
    assert [m["iid"] for m in out] == [1, 2]


# ----------------------------------------------------------------------
# Failure modes
# ----------------------------------------------------------------------
def test_401_raises_permission_error():
    client = _client({"/issues/1": _Response(401, {"message": "unauthorized"})})
    with pytest.raises(PermissionError):
        client.get_issue("acme/api", 1)


def test_403_on_a_required_endpoint_raises_permission_error():
    client = _client({"/merge_requests/1": _Response(403, {"message": "forbidden"})})
    with pytest.raises(PermissionError):
        client.get_merge_request("acme/api", 1)


def test_429_raises_api_error():
    client = _client({"/issues/1": _Response(429, {"message": "slow down"})})
    with pytest.raises(GitLabApiError, match="rate limit"):
        client.get_issue("acme/api", 1)


def test_non_json_body_raises_api_error():
    client = _client({"/issues/1": _Response(200, _INVALID, text="<html>")})
    with pytest.raises(GitLabApiError, match="non-JSON"):
        client.get_issue("acme/api", 1)


def test_error_text_never_carries_the_token():
    """A raised message must not become a token leak into the agent context."""
    from potpie_context_engine.domain.error_redaction import safe_error

    client = _client({"/issues/1": _Response(500, {"message": "boom"})})
    try:
        client.get_issue("acme/api", 1)
    except GitLabApiError as exc:
        assert "glpat-" not in safe_error(exc)
    else:  # pragma: no cover - the call above must raise
        pytest.fail("expected GitLabApiError")

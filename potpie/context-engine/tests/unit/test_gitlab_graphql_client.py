"""GitLab GraphQL fast path: mapping parity with REST, and safe degradation.

The GraphQL query is an optimization, never a requirement — anything that
does not come back in the expected shape must leave the caller on the REST
path rather than surfacing a half-built review history.
"""

from __future__ import annotations

import json

import pytest

from potpie_context_engine.adapters.outbound.connectors.gitlab.graphql_client import (
    GitLabGraphQLClient,
    graphql_enabled,
)


class _Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


class _FakeHttp:
    def __init__(self, response):
        self.response = response
        self.calls: list[tuple[str, dict]] = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response

    def get(self, url, **kwargs):  # pragma: no cover - unused
        raise AssertionError("GraphQL must POST")

    def close(self):
        pass


_MR_NODE = {
    "iid": "7",
    "title": "Fix login timeout",
    "state": "merged",
    "mergedAt": "2026-01-03T00:00:00Z",
    "approved": True,
    "approvedBy": {"nodes": [{"username": "ravi"}]},
    "discussions": {
        "nodes": [
            {
                "id": "disc-1",
                "resolved": True,
                "resolvable": True,
                "notes": {
                    "nodes": [
                        {
                            "id": "note-1",
                            "body": "this drops the retry",
                            "system": False,
                            "createdAt": "2026-01-02T10:00:00Z",
                            "resolvable": True,
                            "resolved": True,
                            "author": {"username": "ravi"},
                            "position": {
                                "newPath": "auth/session.py",
                                "oldPath": "auth/session.py",
                                "newLine": 42,
                                "oldLine": 40,
                            },
                        },
                        {
                            "id": "note-2",
                            "body": "approved this merge request",
                            "system": True,
                            "createdAt": "2026-01-03T00:00:00Z",
                            "author": {"username": "ravi"},
                            "position": None,
                        },
                    ]
                },
            }
        ]
    },
}


def _client(payload, status=200):
    http = _FakeHttp(_Response(status, payload))
    return GitLabGraphQLClient(
        "https://gitlab.corp.example", "glpat-x", http=http
    ), http


def test_endpoint_and_bearer_auth():
    client, http = _client({"data": {"project": {"mergeRequest": _MR_NODE}}})
    client.merge_request_review_history("acme/api", 7)
    url, kwargs = http.calls[0]
    assert url == "https://gitlab.corp.example/api/graphql"
    assert kwargs["headers"]["Authorization"] == "Bearer glpat-x"


def test_iid_is_sent_as_a_string():
    """GitLab's GraphQL schema types iid as String; an int is a hard error."""
    client, http = _client({"data": {"project": {"mergeRequest": _MR_NODE}}})
    client.merge_request_review_history("acme/api", 7)
    variables = http.calls[0][1]["json"]["variables"]
    assert variables == {"fullPath": "acme/api", "iid": "7"}


def test_nested_group_path_is_passed_whole():
    client, http = _client({"data": {"project": {"mergeRequest": _MR_NODE}}})
    client.merge_request_review_history("/acme/platform/api/", 7)
    assert http.calls[0][1]["json"]["variables"]["fullPath"] == "acme/platform/api"


def test_discussions_map_to_the_rest_note_shape():
    client, _ = _client({"data": {"project": {"mergeRequest": _MR_NODE}}})
    out = client.merge_request_review_history("acme/api", 7)
    assert len(out["discussions"]) == 1  # system note excluded from inline review
    note = out["discussions"][0]
    assert note["path"] == "auth/session.py"
    assert note["line"] == 42
    assert note["discussion_id"] == "disc-1"
    assert note["resolved"] is True
    assert note["user"] == {"login": "ravi"}


def test_approvals_map_to_the_rest_approval_shape():
    client, _ = _client({"data": {"project": {"mergeRequest": _MR_NODE}}})
    out = client.merge_request_review_history("acme/api", 7)
    assert out["approvals"]["available"] is True
    assert out["approvals"]["approved"] is True
    assert out["approvals"]["approved_by"] == ["ravi"]


def test_system_notes_are_returned_as_the_review_trail():
    client, _ = _client({"data": {"project": {"mergeRequest": _MR_NODE}}})
    out = client.merge_request_review_history("acme/api", 7)
    assert [n["system"] for n in out["notes"]] == [False, True]
    assert out["notes"][1]["body"] == "approved this merge request"


@pytest.mark.parametrize(
    "payload",
    [
        {"data": {"project": None}},
        {"data": {"project": {"mergeRequest": None}}},
        {"data": None},
        {},
    ],
)
def test_missing_nodes_return_none_so_the_caller_falls_back(payload):
    client, _ = _client(payload)
    assert client.merge_request_review_history("acme/api", 7) is None


def test_graphql_errors_raise_so_the_caller_falls_back():
    client, _ = _client({"errors": [{"message": "Field 'approved' doesn't exist"}]})
    with pytest.raises(RuntimeError, match="errors"):
        client.merge_request_review_history("acme/api", 7)


def test_non_200_raises():
    client, _ = _client({}, status=502)
    with pytest.raises(RuntimeError, match="502"):
        client.merge_request_review_history("acme/api", 7)


def test_disabled_by_env_returns_none_without_a_request(monkeypatch):
    monkeypatch.setenv("CONTEXT_ENGINE_GITLAB_GRAPHQL", "0")
    client, http = _client({"data": {"project": {"mergeRequest": _MR_NODE}}})
    assert client.merge_request_review_history("acme/api", 7) is None
    assert http.calls == []


@pytest.mark.parametrize("value", ["", "1", "true", "yes", "anything"])
def test_enabled_by_default_and_for_truthy_values(monkeypatch, value):
    monkeypatch.setenv("CONTEXT_ENGINE_GITLAB_GRAPHQL", value)
    assert graphql_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
def test_disabled_for_falsey_values(monkeypatch, value):
    monkeypatch.setenv("CONTEXT_ENGINE_GITLAB_GRAPHQL", value)
    assert graphql_enabled() is False

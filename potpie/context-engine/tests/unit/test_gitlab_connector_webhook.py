"""GitLab webhook normalization: authorization, event selection, routing fields.

GitLab authenticates a delivery with a shared secret echoed verbatim in
``X-Gitlab-Token`` rather than an HMAC over the body, so the checks here
pin both that the comparison happens and that the fail-closed posture
matches the GitHub connector's.
"""

from __future__ import annotations

import json

import pytest

from potpie_context_engine.adapters.outbound.connectors.gitlab.connector import (
    GitLabConnector,
)
from potpie_context_engine.domain.source_connector import ConnectorScope

SECRET = "s3cr3t-token"


class _FakePort:
    def __init__(self, merged=()):
        self._merged = list(merged)

    def iter_merged_merge_requests(self, project):
        return iter(self._merged)


def _connector(**kwargs):
    kwargs.setdefault("source_for_project", lambda _p: _FakePort())
    kwargs.setdefault("webhook_secret", SECRET)
    kwargs.setdefault("instance_host", "gitlab.corp.example")
    return GitLabConnector(**kwargs)


def _mr_payload(action="merge", *, iid=7, path="acme/platform/api", state="merged"):
    return json.dumps(
        {
            "object_kind": "merge_request",
            "user": {"username": "dana"},
            "project": {
                "path_with_namespace": path,
                "web_url": f"https://gitlab.corp.example/{path}",
            },
            "object_attributes": {"iid": iid, "action": action, "state": state},
        }
    ).encode()


def _issue_payload(action="open", *, iid=12, path="acme/api"):
    return json.dumps(
        {
            "object_kind": "issue",
            "user": {"username": "sam"},
            "project": {
                "path_with_namespace": path,
                "web_url": f"https://gitlab.corp.example/{path}",
            },
            "object_attributes": {"iid": iid, "action": action, "state": "opened"},
        }
    ).encode()


def _headers(event="Merge Request Hook", token=SECRET, uuid="deliv-1"):
    out = {"X-Gitlab-Event": event}
    if token is not None:
        out["X-Gitlab-Token"] = token
    if uuid is not None:
        out["X-Gitlab-Event-UUID"] = uuid
    return out


# ----------------------------------------------------------------------
# Authorization
# ----------------------------------------------------------------------
def test_matching_secret_token_is_accepted():
    event = _connector().normalize_webhook(_mr_payload(), _headers())
    assert event is not None


def test_wrong_secret_token_is_rejected():
    with pytest.raises(PermissionError, match="mismatch"):
        _connector().normalize_webhook(_mr_payload(), _headers(token="wrong"))


def test_missing_secret_token_is_rejected():
    with pytest.raises(PermissionError, match="mismatch"):
        _connector().normalize_webhook(_mr_payload(), _headers(token=None))


def test_unconfigured_secret_fails_closed():
    connector = _connector(webhook_secret=None)
    with pytest.raises(PermissionError, match="GITLAB_WEBHOOK_SECRET"):
        connector.normalize_webhook(_mr_payload(), _headers(token=None))


def test_unconfigured_secret_passes_only_with_explicit_dev_optin():
    connector = _connector(webhook_secret=None, allow_unsigned=True)
    event = connector.normalize_webhook(_mr_payload(), _headers(token=None))
    assert event is not None


def test_non_ascii_secret_token_compares_without_crashing():
    """An operator may pick a non-ASCII secret; that must 401, not 500."""
    connector = _connector(webhook_secret="ব্যক্তিগত-টোকেন")
    assert (
        connector.normalize_webhook(_mr_payload(), _headers(token="ব্যক্তিগত-টোকেন"))
        is not None
    )
    with pytest.raises(PermissionError):
        connector.normalize_webhook(_mr_payload(), _headers(token="wrong"))


def test_authorization_runs_before_payload_parsing():
    """A bad token must not get as far as decoding attacker-supplied JSON."""
    with pytest.raises(PermissionError):
        _connector().normalize_webhook(b"{not json", _headers(token="wrong"))


def test_header_lookup_is_case_insensitive():
    headers = {"x-gitlab-event": "Merge Request Hook", "x-gitlab-token": SECRET}
    assert _connector().normalize_webhook(_mr_payload(), headers) is not None


# ----------------------------------------------------------------------
# Event selection
# ----------------------------------------------------------------------
def test_merged_merge_request_becomes_a_merged_event():
    event = _connector().normalize_webhook(_mr_payload(), _headers())
    assert event.source_system == "gitlab"
    assert event.event_type == "merge_request"
    assert event.action == "merged"
    assert event.source_id == "mr_7_merged"
    assert event.payload["iid"] == 7
    assert event.payload["is_live_bridge"] is True


def test_state_merged_without_merge_action_still_counts():
    payload = _mr_payload(action="update", state="merged")
    event = _connector().normalize_webhook(payload, _headers())
    assert event is not None and event.action == "merged"


def test_open_merge_request_is_ignored():
    payload = _mr_payload(action="open", state="opened")
    assert _connector().normalize_webhook(payload, _headers()) is None


def test_opened_issue_becomes_an_opened_event():
    event = _connector().normalize_webhook(
        _issue_payload(), _headers(event="Issue Hook")
    )
    assert event.event_type == "issue"
    assert event.action == "opened"
    assert event.source_id == "issue_12_opened"


def test_issue_update_is_ignored():
    payload = _issue_payload(action="update")
    assert _connector().normalize_webhook(payload, _headers(event="Issue Hook")) is None


def test_unrelated_hook_kind_is_ignored():
    assert (
        _connector().normalize_webhook(_mr_payload(), _headers(event="Push Hook"))
        is None
    )


def test_malformed_json_is_ignored_not_raised():
    assert _connector().normalize_webhook(b"{not json", _headers()) is None


def test_payload_without_project_path_is_ignored():
    body = json.dumps(
        {
            "object_kind": "merge_request",
            "object_attributes": {"iid": 1, "action": "merge"},
        }
    ).encode()
    assert _connector().normalize_webhook(body, _headers()) is None


# ----------------------------------------------------------------------
# Routing fields
# ----------------------------------------------------------------------
def test_nested_group_path_is_preserved_verbatim():
    event = _connector().normalize_webhook(_mr_payload(), _headers())
    assert event.repo_name == "acme/platform/api"
    assert event.payload["project_path"] == "acme/platform/api"


def test_provider_host_comes_from_the_project_url_not_a_default():
    body = json.dumps(
        {
            "object_kind": "merge_request",
            "project": {
                "path_with_namespace": "acme/api",
                "web_url": "https://git.internal.example/acme/api",
            },
            "object_attributes": {"iid": 1, "action": "merge"},
        }
    ).encode()
    event = _connector().normalize_webhook(body, _headers())
    assert event.provider_host == "git.internal.example"


def test_provider_host_falls_back_to_the_configured_instance():
    body = json.dumps(
        {
            "object_kind": "merge_request",
            "project": {"path_with_namespace": "acme/api"},
            "object_attributes": {"iid": 1, "action": "merge"},
        }
    ).encode()
    event = _connector().normalize_webhook(body, _headers())
    assert event.provider_host == "gitlab.corp.example"


def test_delivery_uuid_is_carried_for_deduplication():
    event = _connector().normalize_webhook(_mr_payload(), _headers(uuid="uuid-42"))
    assert event.source_event_id == "uuid-42"


def test_sender_login_is_captured():
    event = _connector().normalize_webhook(_mr_payload(), _headers())
    assert event.payload["sender_login"] == "dana"


# ----------------------------------------------------------------------
# Capabilities and artifact listing
# ----------------------------------------------------------------------
def test_capabilities_cover_merge_requests_and_issues():
    caps = _connector().capabilities()
    kinds = {c.source_kind for c in caps}
    assert {"merge_request", "issue"} <= kinds
    assert all(c.provider == "gitlab" for c in caps)
    assert any(c.webhook_capable for c in caps)


def test_list_artifacts_emits_merged_merge_request_refs():
    merged = [{"iid": 3}, {"iid": 4}]
    connector = _connector(source_for_project=lambda _p: _FakePort(merged))
    refs = list(
        connector.list_artifacts(
            ConnectorScope(pot_id="pot-1", scope={"project_path": "acme/api"})
        )
    )
    assert [r.ref for r in refs] == ["gitlab:mr:acme/api:3", "gitlab:mr:acme/api:4"]
    assert all(r.source_type == "merge_request" for r in refs)
    assert all(r.source_system == "gitlab" for r in refs)


def test_list_artifacts_accepts_the_shared_repo_name_key():
    connector = _connector(source_for_project=lambda _p: _FakePort([{"iid": 1}]))
    refs = list(
        connector.list_artifacts(
            ConnectorScope(pot_id="pot-1", scope={"repo_name": "acme/api"})
        )
    )
    assert [r.ref for r in refs] == ["gitlab:mr:acme/api:1"]


def test_list_artifacts_without_a_project_is_empty():
    assert list(_connector().list_artifacts(ConnectorScope(pot_id="p", scope={}))) == []

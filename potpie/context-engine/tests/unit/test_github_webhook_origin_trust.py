"""GitHub webhook ingress captures author_association for origin trust."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from potpie_context_engine.adapters.outbound.connectors.github.connector import (
    GitHubConnector,
)

pytestmark = pytest.mark.unit


class _NoGithub:
    def iter_closed_pulls(self, repo_name: str):
        return []


def test_normalize_webhook_captures_author_association_and_sender() -> None:
    secret = "whsec"
    body = {
        "action": "closed",
        "pull_request": {
            "number": 9,
            "merged": True,
            "author_association": "FIRST_TIME_CONTRIBUTOR",
            "user": {"login": "attacker"},
        },
        "repository": {"full_name": "acme/widgets"},
        "sender": {"login": "attacker", "type": "User"},
    }
    raw = json.dumps(body).encode("utf-8")
    signature = "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    connector = GitHubConnector(
        source_for_repo=lambda _repo: _NoGithub(),
        webhook_secret=secret,
    )
    event = connector.normalize_webhook(
        raw,
        {
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": signature,
            "X-GitHub-Delivery": "deliv-1",
        },
    )
    assert event is not None
    assert event.payload["author_association"] == "FIRST_TIME_CONTRIBUTOR"
    assert event.payload["sender_login"] == "attacker"
    assert event.payload["sender_type"] == "User"

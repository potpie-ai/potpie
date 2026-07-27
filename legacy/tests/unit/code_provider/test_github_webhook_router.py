"""Unit tests for the GitHub webhook PR review endpoint."""

import hashlib
import hmac
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Must be set before any app import that touches app.core.database (it creates
# the SQLAlchemy engine at module level and needs a non-None URL).
os.environ.setdefault("POSTGRES_SERVER", "postgresql://test:test@localhost:5432/test_db")

from app.modules.code_provider.github.github_webhook_router import router, _verify_signature

pytestmark = pytest.mark.unit

app = FastAPI()
app.include_router(router)
client = TestClient(app)


# ── signature verification ────────────────────────────────────────────────────

class TestVerifySignature:
    """Tests for HMAC-SHA256 webhook signature verification."""

    def test_valid_signature(self):
        """Returns True when signature matches the payload and secret."""
        secret = "mysecret"
        payload = b'{"action": "opened"}'
        sig = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        assert _verify_signature(payload, sig, secret) is True

    def test_invalid_signature(self):
        """Returns False when the signature digest does not match."""
        assert _verify_signature(b"payload", "sha256=wrong", "secret") is False

    def test_missing_sha256_prefix(self):
        """Returns False when the signature is missing the sha256= prefix."""
        assert _verify_signature(b"payload", "badsig", "secret") is False


# ── webhook endpoint ──────────────────────────────────────────────────────────

PR_PAYLOAD = {
    "action": "opened",
    "pull_request": {
        "number": 42,
        "title": "Add new feature",
        "head": {"ref": "feat/new-feature"},
    },
    "repository": {"full_name": "acme/my-repo"},
}


class TestGithubWebhookEndpoint:
    """Tests for the POST /github/webhook endpoint."""

    def _post(self, payload=None, event="pull_request", secret=None):
        """Helper to POST a webhook payload with optional signature."""
        body = json.dumps(payload or PR_PAYLOAD).encode()
        headers = {"X-GitHub-Event": event}
        if secret:
            sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
            headers["X-Hub-Signature-256"] = sig
        return client.post("/github/webhook", content=body, headers=headers)

    def test_non_pr_event_is_ignored(self):
        """Non-pull_request events return status ignored."""
        resp = self._post(event="push")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"

    def test_pr_action_not_in_allowlist_is_ignored(self):
        """PR actions outside the allowlist (e.g. closed) return status ignored."""
        payload = {**PR_PAYLOAD, "action": "closed"}
        resp = self._post(payload=payload)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"

    def test_missing_signature_rejected_when_secret_set(self):
        """Returns 400 when GITHUB_WEBHOOK_SECRET is set but no signature header is sent."""
        with patch.dict("os.environ", {"GITHUB_WEBHOOK_SECRET": "mysecret"}):
            body = json.dumps(PR_PAYLOAD).encode()
            resp = client.post(
                "/github/webhook",
                content=body,
                headers={"X-GitHub-Event": "pull_request"},
            )
        assert resp.status_code == 400

    def test_wrong_signature_rejected(self):
        """Returns 401 when the provided signature does not match the payload."""
        with patch.dict("os.environ", {"GITHUB_WEBHOOK_SECRET": "mysecret"}):
            body = json.dumps(PR_PAYLOAD).encode()
            resp = client.post(
                "/github/webhook",
                content=body,
                headers={
                    "X-GitHub-Event": "pull_request",
                    "X-Hub-Signature-256": "sha256=badsignature",
                },
            )
        assert resp.status_code == 401

    def test_valid_pr_accepted_and_background_task_queued(self):
        """Valid PR payload returns 200 accepted and queues the review background task."""
        with patch(
            "app.modules.code_provider.github.github_webhook_router._run_pr_review"
        ) as mock_task:
            resp = self._post()

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["pr"] == 42
        assert data["repo"] == "acme/my-repo"
        mock_task.assert_called_once()

    @pytest.mark.parametrize("action", ["opened", "synchronize", "reopened"])
    def test_all_trigger_actions_accepted(self, action):
        """All three trigger actions (opened, synchronize, reopened) are accepted."""
        payload = {**PR_PAYLOAD, "action": action}
        resp = self._post(payload=payload)
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"

    def test_valid_signature_accepted(self):
        """Request with correct HMAC signature is accepted when secret is configured."""
        secret = "topsecret"
        with patch.dict("os.environ", {"GITHUB_WEBHOOK_SECRET": secret}):
            resp = self._post(secret=secret)
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"


# ── background task ───────────────────────────────────────────────────────────

class TestRunPrReview:
    """Tests for the _run_pr_review background task."""

    @pytest.mark.asyncio
    async def test_skips_when_no_project_found(self):
        """Silently skips posting a review when no matching Potpie project exists."""
        from app.modules.code_provider.github.github_webhook_router import _run_pr_review

        mock_project_service = MagicMock()
        mock_project_service.get_global_project_from_db = AsyncMock(return_value=None)

        with patch("app.modules.code_provider.github.github_webhook_router.SessionLocal") as mock_db, \
             patch("app.modules.code_provider.github.github_webhook_router.ProjectService",
                   return_value=mock_project_service):
            mock_db.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_db.return_value.__exit__ = MagicMock(return_value=False)

            await _run_pr_review("acme/unknown-repo", 1, "main", "Test PR")

        mock_project_service.get_global_project_from_db.assert_called_once_with(
            repo_name="acme/unknown-repo",
            branch_name="main",
        )

    @pytest.mark.asyncio
    async def test_runs_agent_and_posts_comment_when_project_exists(self):
        """Runs BlastRadiusAgent and posts review comment when a matching project is found."""
        from app.modules.code_provider.github.github_webhook_router import _run_pr_review

        mock_project = MagicMock()
        mock_project.id = "proj-123"
        mock_project.repo_name = "acme/my-repo"
        mock_project.user_id = "user-abc"

        mock_project_service = MagicMock()
        mock_project_service.get_global_project_from_db = AsyncMock(return_value=mock_project)

        mock_result = MagicMock()
        mock_result.response = "This PR looks risky — auth middleware is affected."

        mock_agents_service = MagicMock()
        mock_agents_service.execute = AsyncMock(return_value=mock_result)

        with patch("app.modules.code_provider.github.github_webhook_router.SessionLocal"), \
             patch("app.modules.code_provider.github.github_webhook_router.ProjectService",
                   return_value=mock_project_service), \
             patch("app.modules.code_provider.github.github_webhook_router.ProviderService"), \
             patch("app.modules.code_provider.github.github_webhook_router.ToolService"), \
             patch("app.modules.code_provider.github.github_webhook_router.PromptService"), \
             patch("app.modules.code_provider.github.github_webhook_router.AgentsService",
                   return_value=mock_agents_service), \
             patch("app.modules.code_provider.github.github_webhook_router._post_pr_comment") as mock_post:

            await _run_pr_review("acme/my-repo", 42, "feat/new-feature", "Add new feature")

        mock_agents_service.execute.assert_called_once()
        mock_post.assert_called_once_with(
            "acme/my-repo", 42, "This PR looks risky — auth middleware is affected."
        )

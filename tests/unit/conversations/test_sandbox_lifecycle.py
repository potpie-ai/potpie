"""Unit tests for the conversation_service ↔ ProjectSandbox seam (Phase 4).

Covers ``_ensure_project_sandbox_safe`` — the single helper called from
``create_conversation`` (fire-and-forget) and ``store_message``
(synchronous with timeout). The full ``store_message`` and
``create_conversation`` paths are heavy mocks elsewhere; these tests
target the helper directly so the policy decisions (skip local-mode,
swallow ensure failures, pass repo metadata correctly) are pinned in
isolation.

The end-to-end "create_conversation kicks off ensure / store_message
runs ensure with a timeout" wiring is verified via direct method
calls rather than a full streaming run, since the latter requires
spinning up too many collaborators for what's a thin glue layer.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.modules.conversations.conversation.conversation_service import (
    ConversationService,
)


pytestmark = pytest.mark.unit


@pytest.fixture
def fake_project_sandbox():
    """A MagicMock with the ProjectSandbox surface ConversationService touches.

    Calls to ``ensure`` are recorded with their kwargs so tests can
    assert the wiring; ``ensure`` returns a placeholder handle (the
    real call returns a WorkspaceHandle, but the conversation service
    discards it and only cares about success/failure).
    """
    mock = MagicMock(name="ProjectSandbox")
    mock.ensure = AsyncMock(return_value=MagicMock(name="WorkspaceHandle"))
    return mock


@pytest.fixture
def service(fake_project_sandbox):
    """Build a ConversationService with all collaborators mocked.

    Patches ``get_project_sandbox`` so the constructor wires up our
    fake instead of fetching the process-wide one (which would trip
    up the real sandbox bootstrap if env wasn't set).
    """
    with patch(
        "app.modules.intelligence.tools.sandbox.project_sandbox.get_project_sandbox",
        return_value=fake_project_sandbox,
    ):
        svc = ConversationService(
            db=MagicMock(),
            user_id="u1",
            user_email="u1@example.com",
            conversation_store=MagicMock(),
            message_store=MagicMock(),
            project_service=MagicMock(),
            history_manager=MagicMock(),
            provider_service=MagicMock(),
            tools_service=MagicMock(),
            promt_service=MagicMock(),
            agent_service=MagicMock(),
            custom_agent_service=MagicMock(),
            media_service=MagicMock(),
            session_service=MagicMock(),
            redis_manager=MagicMock(),
        )
    return svc


def _project_dict(**overrides):
    """Build a project metadata dict matching the shape ProjectService
    returns. Defaults to a remote (clonable) project."""
    base = {
        "project_name": "owner/repo",
        "branch_name": "main",
        "commit_id": "deadbeef",
        "repo_path": None,  # None ⇒ remote repo, not a local upload
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# _ensure_project_sandbox_safe — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_calls_project_sandbox_with_correct_ref(
    service, fake_project_sandbox
):
    """Happy path: project metadata resolved → ProjectSandbox.ensure called
    with a ProjectRef built from the project row. base_ref prefers
    commit_id over branch_name (matches what parsing pinned)."""
    with patch.object(
        service, "_fetch_project_for_provision",
        return_value=_project_dict(),
    ):
        result = await service._ensure_project_sandbox_safe("p1", "u1")

    assert result is True
    fake_project_sandbox.ensure.assert_awaited_once()
    kwargs = fake_project_sandbox.ensure.await_args.kwargs
    assert kwargs["user_id"] == "u1"
    assert kwargs["project_id"] == "p1"
    repo_ref = kwargs["repo"]
    assert repo_ref.repo_name == "owner/repo"
    # commit_id wins because the parse pinned it; branch_name is fallback.
    assert repo_ref.base_ref == "deadbeef"


@pytest.mark.asyncio
async def test_ensure_falls_back_to_branch_when_no_commit_id(
    service, fake_project_sandbox
):
    with patch.object(
        service, "_fetch_project_for_provision",
        return_value=_project_dict(commit_id=None),
    ):
        await service._ensure_project_sandbox_safe("p1", "u1")
    repo_ref = fake_project_sandbox.ensure.await_args.kwargs["repo"]
    assert repo_ref.base_ref == "main"


# ---------------------------------------------------------------------------
# Skip cases — local uploads, missing metadata, no ref
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_skips_local_upload_projects(
    service, fake_project_sandbox
):
    """Projects with a host-side ``repo_path`` are local uploads — no
    remote URL to clone, no point in materializing a sandbox. The IDE
    tunnel covers reads for those."""
    with patch.object(
        service, "_fetch_project_for_provision",
        return_value=_project_dict(repo_path="/host/path/to/upload"),
    ):
        result = await service._ensure_project_sandbox_safe("p1", "u1")

    assert result is False
    fake_project_sandbox.ensure.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_skips_when_project_not_found(
    service, fake_project_sandbox
):
    with patch.object(
        service, "_fetch_project_for_provision", return_value=None
    ):
        result = await service._ensure_project_sandbox_safe("missing", "u1")
    assert result is False
    fake_project_sandbox.ensure.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_skips_when_no_ref_resolvable(
    service, fake_project_sandbox
):
    """A project row with neither commit_id nor branch_name shouldn't
    crash — we log and skip. Happens for partial-write projects that
    failed mid-parse."""
    with patch.object(
        service, "_fetch_project_for_provision",
        return_value=_project_dict(commit_id=None, branch_name=None),
    ):
        result = await service._ensure_project_sandbox_safe("p1", "u1")
    assert result is False
    fake_project_sandbox.ensure.assert_not_called()


# ---------------------------------------------------------------------------
# Failure tolerance — ensure errors and timeouts must not propagate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_swallows_underlying_failures(
    service, fake_project_sandbox
):
    """A flaky Daytona / network error during ensure must not crash
    the calling conversation create or message handler. The agent
    tools each call ensure() again on first run and pay the cost
    there if the conversation actually needs the sandbox."""
    fake_project_sandbox.ensure.side_effect = RuntimeError("Daytona blip")
    with patch.object(
        service, "_fetch_project_for_provision",
        return_value=_project_dict(),
    ):
        result = await service._ensure_project_sandbox_safe("p1", "u1")
    assert result is False  # logged, swallowed


@pytest.mark.asyncio
async def test_ensure_returns_false_on_timeout(
    service, fake_project_sandbox
):
    """When the call overruns its timeout (slow Daytona create),
    we report False so observability sees the deadline missed, but
    don't propagate — the agent tool's own ensure() will pick up."""
    import asyncio

    async def _slow(*_args, **_kwargs):
        await asyncio.sleep(5.0)

    fake_project_sandbox.ensure.side_effect = _slow
    with patch.object(
        service, "_fetch_project_for_provision",
        return_value=_project_dict(),
    ):
        result = await service._ensure_project_sandbox_safe(
            "p1", "u1", timeout_s=0.05
        )
    assert result is False

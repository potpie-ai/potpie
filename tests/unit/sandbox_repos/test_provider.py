"""Tests for ``RepoManagerWorkspaceProvider``.

The provider implements the sandbox ``WorkspaceProvider`` protocol but
delegates clone + worktree work to ``RepoManager``. We verify the right
RepoManager methods get called for each ``WorkspaceMode``, that the in-memory
key index returns the same ``Workspace`` for repeat calls in a conversation,
and that ``delete_workspace`` cleans up via ``cleanup_unique_worktree``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.modules.sandbox_repos import RepoManagerWorkspaceProvider
from sandbox.domain.errors import RepoAuthFailed, RepoCacheUnavailable
from sandbox.domain.models import (
    RepoIdentity,
    WorkspaceMode,
    WorkspaceRequest,
)


def _request(
    *,
    mode: WorkspaceMode = WorkspaceMode.EDIT,
    conversation_id: str | None = "conv-abc",
    task_id: str | None = None,
    branch_name: str | None = None,
    create_branch: bool = True,
    auth_token: str | None = "tok",
) -> WorkspaceRequest:
    return WorkspaceRequest(
        user_id="u1",
        project_id="proj-1",
        repo=RepoIdentity(repo_name="owner/repo"),
        base_ref="main",
        mode=mode,
        conversation_id=conversation_id,
        task_id=task_id,
        branch_name=branch_name,
        create_branch=create_branch,
        auth_token=auth_token,
    )


@pytest.fixture
def fake_repo_manager(tmp_path: Path) -> MagicMock:
    rm = MagicMock()
    # Default returns: every call hands back a real path under tmp_path so
    # ``Workspace.location.local_path`` resolves to something concrete.
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    rm.create_worktree_with_new_branch.return_value = worktree
    rm.create_worktree.return_value = worktree
    rm.ensure_bare_repo.return_value = tmp_path / ".bare"
    rm.cleanup_unique_worktree.return_value = True
    return rm


@pytest.mark.asyncio
async def test_edit_mode_uses_create_worktree_with_new_branch(fake_repo_manager: MagicMock) -> None:
    provider = RepoManagerWorkspaceProvider(fake_repo_manager)
    request = _request()

    workspace = await provider.get_or_create_workspace(request)

    fake_repo_manager.create_worktree_with_new_branch.assert_called_once()
    kwargs = fake_repo_manager.create_worktree_with_new_branch.call_args.kwargs
    assert kwargs["repo_name"] == "owner/repo"
    assert kwargs["base_ref"] == "main"
    assert kwargs["new_branch_name"] == "agent/edits-conv-abc"
    assert kwargs["unique_id"] == "conv-abc"
    assert kwargs["user_id"] == "u1"
    assert kwargs["exists_ok"] is True
    assert workspace.metadata["branch"] == "agent/edits-conv-abc"


@pytest.mark.asyncio
async def test_task_mode_uses_task_branch_prefix(fake_repo_manager: MagicMock) -> None:
    provider = RepoManagerWorkspaceProvider(fake_repo_manager)
    request = _request(
        mode=WorkspaceMode.TASK, conversation_id=None, task_id="task-42"
    )

    workspace = await provider.get_or_create_workspace(request)

    kwargs = fake_repo_manager.create_worktree_with_new_branch.call_args.kwargs
    assert kwargs["new_branch_name"] == "agent/task-task-42"
    assert kwargs["unique_id"] == "task-42"
    assert workspace.metadata["branch"] == "agent/task-task-42"


@pytest.mark.asyncio
async def test_explicit_branch_name_overrides_default(fake_repo_manager: MagicMock) -> None:
    provider = RepoManagerWorkspaceProvider(fake_repo_manager)
    request = _request(branch_name="agent/custom-branch")

    workspace = await provider.get_or_create_workspace(request)

    kwargs = fake_repo_manager.create_worktree_with_new_branch.call_args.kwargs
    assert kwargs["new_branch_name"] == "agent/custom-branch"
    assert workspace.metadata["branch"] == "agent/custom-branch"


@pytest.mark.asyncio
async def test_analysis_mode_uses_create_worktree_at_base_ref(fake_repo_manager: MagicMock) -> None:
    provider = RepoManagerWorkspaceProvider(fake_repo_manager)
    request = _request(mode=WorkspaceMode.ANALYSIS, create_branch=False)

    workspace = await provider.get_or_create_workspace(request)

    fake_repo_manager.create_worktree_with_new_branch.assert_not_called()
    fake_repo_manager.ensure_bare_repo.assert_called_once()
    fake_repo_manager.create_worktree.assert_called_once()
    kwargs = fake_repo_manager.create_worktree.call_args.kwargs
    assert kwargs["ref"] == "main"
    assert kwargs["exists_ok"] is True
    assert workspace.metadata["branch"] == "main"


@pytest.mark.asyncio
async def test_repeat_call_returns_same_workspace(fake_repo_manager: MagicMock) -> None:
    provider = RepoManagerWorkspaceProvider(fake_repo_manager)
    request = _request()

    first = await provider.get_or_create_workspace(request)
    second = await provider.get_or_create_workspace(request)

    assert first is second
    # RepoManager hit only on the first call — second hit the in-memory index.
    assert fake_repo_manager.create_worktree_with_new_branch.call_count == 1


@pytest.mark.asyncio
async def test_delete_workspace_cleans_up_via_repo_manager(fake_repo_manager: MagicMock) -> None:
    provider = RepoManagerWorkspaceProvider(fake_repo_manager)
    request = _request()
    workspace = await provider.get_or_create_workspace(request)

    await provider.delete_workspace(workspace)

    fake_repo_manager.cleanup_unique_worktree.assert_called_once_with(
        "owner/repo", "u1", "conv-abc"
    )
    # Index drops the entry so a subsequent call would re-create.
    assert await provider.get_workspace(workspace.id) is None


@pytest.mark.asyncio
async def test_delete_analysis_workspace_does_not_cleanup(fake_repo_manager: MagicMock) -> None:
    provider = RepoManagerWorkspaceProvider(fake_repo_manager)
    request = _request(mode=WorkspaceMode.ANALYSIS, create_branch=False)
    workspace = await provider.get_or_create_workspace(request)

    await provider.delete_workspace(workspace)

    # Analysis worktrees are shared and aren't conversation-scoped — they age
    # out via RepoManager's volume eviction, not eager cleanup.
    fake_repo_manager.cleanup_unique_worktree.assert_not_called()


@pytest.mark.asyncio
async def test_runtime_error_with_auth_message_raises_repo_auth_failed(
    fake_repo_manager: MagicMock,
) -> None:
    fake_repo_manager.create_worktree_with_new_branch.side_effect = RuntimeError(
        "Git authentication failed for 'owner/repo'"
    )
    provider = RepoManagerWorkspaceProvider(fake_repo_manager)

    with pytest.raises(RepoAuthFailed):
        await provider.get_or_create_workspace(_request())


@pytest.mark.asyncio
async def test_other_runtime_error_raises_repo_cache_unavailable(
    fake_repo_manager: MagicMock,
) -> None:
    fake_repo_manager.create_worktree_with_new_branch.side_effect = RuntimeError(
        "Git worktree add failed: bad ref"
    )
    provider = RepoManagerWorkspaceProvider(fake_repo_manager)

    with pytest.raises(RepoCacheUnavailable):
        await provider.get_or_create_workspace(_request())


@pytest.mark.asyncio
async def test_mount_for_runtime_returns_local_path(fake_repo_manager: MagicMock) -> None:
    provider = RepoManagerWorkspaceProvider(fake_repo_manager)
    workspace = await provider.get_or_create_workspace(_request())

    mount = await provider.mount_for_runtime(workspace, writable=True)

    assert mount.target == "/work"
    assert mount.writable is True
    assert mount.source == str(Path(workspace.location.local_path).resolve())  # type: ignore[arg-type]

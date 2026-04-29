"""Repo-aware bridge between the standalone sandbox module and RepoManager.

The sandbox library under ``app/src/sandbox/`` is intentionally standalone —
it knows nothing about RepoManager. This module sits on top of both: it
implements ``sandbox.domain.ports.workspaces.WorkspaceProvider`` and delegates
clone + worktree work to ``app.modules.repo_manager.RepoManager`` so parsing
and agent tooling share the same on-disk cache (one eviction policy, no
duplicate clones).
"""

from app.modules.sandbox_repos.provider import RepoManagerWorkspaceProvider

__all__ = ["RepoManagerWorkspaceProvider"]

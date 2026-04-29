"""Process-wide accessor for the shared :class:`SandboxClient`.

We hold one client per process so the underlying provider/store/locks are
constructed once. Tools call :func:`get_sandbox_client` and then
:func:`resolve_workspace` to materialise a workspace for the current run.

For tests, :func:`set_sandbox_client` lets a fixture inject a stub client.
"""

from __future__ import annotations

import os
from typing import Any

from sandbox import RepoCache, SandboxClient, WorkspaceHandle, WorkspaceMode
from sandbox.bootstrap.container import build_sandbox_container
from sandbox.bootstrap.settings import settings_from_env

from app.modules.intelligence.tools.sandbox.context import (
    get_auth_token,
    get_conversation_id,
    get_user_id,
)


_client: SandboxClient | None = None


def get_sandbox_client() -> SandboxClient:
    """Return the lazily-initialised process-wide client.

    Local mode has two wirings:

    * **Legacy (default)** — swap in :class:`RepoManagerWorkspaceProvider`
      so clones and worktrees flow through ``app.modules.repo_manager``.
      Parsing's ``clone_or_copy_repository`` already uses the same
      RepoManager, so both paths share one on-disk cache and one
      eviction policy. Worktree layout is RepoManager's
      (``<branch>`` for shared, ``<user>_<unique>_<branch>`` for
      conversation-scoped).

    * **Canonical** (``SANDBOX_USE_CANONICAL_LOCAL=true``, P1) —
      ``LocalGitWorkspaceProvider`` becomes the workspace provider and
      ``LocalRepoCacheProvider`` is the cache port. Worktree layout is
      ``<user>_<scope>_<branch>``. Existing legacy worktrees on disk
      are NOT migrated — they remain accessible via RepoManager but
      become orphaned from the canonical adapter's view. New
      conversations get fresh worktrees in the canonical layout.

    Daytona uses the stock provider in both cases — the bare repo
    lives inside the managed sandbox.
    """
    global _client
    if _client is None:
        settings = settings_from_env()
        if settings.provider == "local" and not settings.use_canonical_local:
            from app.modules.repo_manager import RepoManager
            from app.modules.sandbox_repos import RepoManagerWorkspaceProvider

            repo_manager = RepoManager(repos_base_path=settings.repos_base_path)
            workspace_provider = RepoManagerWorkspaceProvider(repo_manager)
            container = build_sandbox_container(
                settings, workspace_provider=workspace_provider
            )
        else:
            # Canonical local OR Daytona — let bootstrap pick everything.
            container = build_sandbox_container(settings)
        _client = SandboxClient.from_container(container)
    return _client


def set_sandbox_client(client: SandboxClient | None) -> None:
    """Override the process-wide client (test seam, embedded use)."""
    global _client
    _client = client


def _resolve_auth_token(project_user_id: str | None) -> str | None:
    """Best-effort token lookup.

    Order:
      1. The auth_token contextvar (set by the execution flow when a per-run
         token is available — e.g. user OAuth).
      2. ``GH_TOKEN`` / ``GITHUB_TOKEN`` env vars (CI / dev path).

    The proper GitHub-App → OAuth → env-token chain currently lives in
    ``app/modules/repo_manager/sync_helper.py``. Plan phase 9 moves it into
    the sandbox local adapter; until then we accept a degraded auth path so
    sandbox tools work for public repos and CI tokens out of the box.
    """
    token = get_auth_token()
    if token:
        return token
    return os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")


async def provision_repo_cache(
    *,
    user_id: str,
    repo_name: str,
    base_ref: str,
    auth_token: str | None = None,
    repo_url: str | None = None,
) -> RepoCache:
    """Materialise the bare repo for this project and persist a `RepoCache`.

    Called from parsing's READY hook so subsequent agent runs hit a warm
    cache instead of paying for a re-clone on first workspace request.
    Best-effort token resolution: uses the explicit token if provided,
    otherwise falls back to the local adapter's resolver chain (env
    vars; production wires in the GitHub-App / OAuth resolver).

    Idempotent on `(provider_host, repo_name)` — repeat calls just
    fetch the requested ref into the existing bare. Safe to call from
    multiple workers concurrently; the service's per-key lock serialises.
    """
    client = get_sandbox_client()
    token = auth_token or _resolve_auth_token(user_id)
    return await client.ensure_repo_cache(
        user_id=user_id,
        repo=repo_name,
        base_ref=base_ref,
        repo_url=repo_url,
        auth_token=token,
    )


async def resolve_workspace(
    *,
    user_id: str,
    project_id: str,
    repo_name: str,
    branch: str,
    base_ref: str | None = None,
    create_branch: bool = True,
    repo_url: str | None = None,
    mode: WorkspaceMode = WorkspaceMode.EDIT,
) -> WorkspaceHandle:
    """Materialise (or look up) the workspace for this run.

    Idempotent on ``(user_id, project_id, repo, branch, mode, conversation)``
    inside the underlying client; repeated tool calls in the same run hit the
    same workspace. The conversation id comes from the contextvar so the LLM
    can't fabricate one.

    For EDIT/TASK modes we derive a per-conversation branch name from the
    contextvar (``agent/edits-<convid>`` / ``agent/task-<taskid>``). The
    ``branch`` argument is interpreted as the BASE branch the workspace
    should fork off — this matches what tool callers actually pass (the
    project's stored base branch). Without this derivation, the canonical
    local adapter would refuse to create a second worktree on the shared
    base branch (git rejects multiple worktrees on the same ref).
    """
    client = get_sandbox_client()
    auth_token = _resolve_auth_token(user_id)
    conversation_id = get_conversation_id()
    base = base_ref or branch
    workspace_branch: str
    if mode is WorkspaceMode.ANALYSIS:
        # Analysis mode reads from the base branch directly.
        workspace_branch = base
    elif conversation_id:
        workspace_branch = f"agent/edits-{conversation_id}"
    else:
        # No conversation context — fall back to the user-provided branch.
        # The agent harness should always set the contextvar; this path
        # exists only to keep the API forgiving for ad-hoc invocations.
        workspace_branch = branch
    return await client.get_workspace(
        user_id=user_id,
        project_id=project_id,
        repo=repo_name,
        repo_url=repo_url,
        branch=workspace_branch,
        base_ref=base,
        create_branch=create_branch,
        auth_token=auth_token,
        mode=mode,
        conversation_id=conversation_id,
    )


def context_user_id_required() -> str:
    """Return ``user_id`` from contextvar; raise if unset."""
    user_id = get_user_id()
    if not user_id:
        raise RuntimeError(
            "SandboxRunContext.user_id is not set. The agent execution flow must "
            "call sandbox.context.set_run_context(...) before invoking tools."
        )
    return user_id


# A small process-wide cache keyed on project_id. Project rows are immutable
# enough for our purposes (repo_name and base branch only change on re-parse),
# and both the workspace resolver and the announcement banner read them — we
# don't want to hit Postgres on every sandbox tool call.
_project_summary_cache: dict[str, dict[str, str]] = {}


def lookup_project_summary(project_id: str) -> dict[str, str]:
    """Return ``{repo_name, base_branch, repo_url, user_id}`` for a project.

    Synchronous DB read; cached for the life of the process. Raises if the
    project row isn't found. Used by both the workspace resolver and the
    setup-banner emitted before the first sandbox tool call.
    """
    cached = _project_summary_cache.get(project_id)
    if cached is not None:
        return cached

    from app.core.database import SessionLocal
    from app.modules.projects.projects_service import ProjectService

    with SessionLocal() as db:
        service = ProjectService(db)
        details = service.get_project_from_db_by_id_sync(project_id)  # type: ignore[arg-type]
    if not details or "project_name" not in details:
        raise ValueError(f"Cannot find repo details for project_id: {project_id}")

    summary = {
        "repo_name": details["project_name"],
        "base_branch": details.get("branch_name") or "main",
        "repo_url": details.get("repo_path") or details.get("repo_url") or "",
        "user_id": details.get("user_id") or "",
    }
    _project_summary_cache[project_id] = summary
    return summary


def clear_project_summary_cache() -> None:
    """Drop the cached project summaries (test seam / re-parse hook)."""
    _project_summary_cache.clear()


def workspace_setup_banner(project_id: str, conversation_id: str | None) -> str:
    """Build the human-readable "Setting up workspace…" line.

    Returns "" if the project lookup fails — the announcement is best-effort
    and must never break the tool flow.
    """
    try:
        summary = lookup_project_summary(project_id)
    except Exception:
        return ""
    repo = summary["repo_name"]
    base = summary["base_branch"]
    if conversation_id:
        derived = f"agent/edits-{conversation_id[:8].replace('/', '-')}"
        return f"Setting up workspace for {repo} on {derived} (from {base})…"
    return f"Setting up workspace for {repo} (from {base})…"



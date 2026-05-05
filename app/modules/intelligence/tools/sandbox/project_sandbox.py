"""Project-scoped sandbox lifecycle facade.

One sandbox per ``(user_id, project_id)``, kept alive across parses and
conversations. Callers (parsing pipeline, conversation_service, REST
handlers) go through this facade rather than reaching into
:class:`SandboxClient` directly so the lifecycle policy lives in one
place:

* ``ensure(...)`` is idempotent and self-healing. If the backing
  workspace is gone (Daytona sandbox archived/deleted, local worktree
  scrubbed) it tears down the stale record and re-creates. Long TTL is
  the provider's job (see ``DaytonaWorkspaceProvider.auto_stop_minutes``);
  ProjectSandbox just picks the right mode and hands back a handle.
* ``health_check(...)`` is the cheap probe conversation_service runs
  on every message. ``False`` ⇒ caller routes back through ``ensure``.
* ``parse(...)`` is the in-sandbox parser wrapper — phase 3 will plug
  this into the parsing pipeline.

The facade is mode-aware: parsing and read-only agent tools want
ANALYSIS workspaces (read-only, idempotent on base_ref), while
conversation-edit tools forking per-conversation branches still go
through :func:`resolve_workspace` in ``client.py`` for the EDIT-mode
key. Both share the same underlying Daytona sandbox once Phase 5
collapses the workspace surface; for now they coexist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sandbox import SandboxClient, WorkspaceHandle, WorkspaceMode

from app.modules.intelligence.tools.sandbox.client import (
    _resolve_auth_token,
    get_sandbox_client,
)
from app.modules.utils.logger import setup_logger

if TYPE_CHECKING:
    from sandbox.api.parser_wire import ParseArtifacts

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class ProjectRef:
    """Minimal repo-identity bundle the facade needs to materialize a workspace.

    Surfaces only the fields that vary per call — auth resolution,
    user_id and project_id are passed separately so callers don't
    have to keep them in sync with the auth context.
    """

    repo_name: str
    base_ref: str
    repo_url: str | None = None


class ProjectSandbox:
    """Owns the ANALYSIS-mode workspace for one project across its lifetime.

    Stateless — every call goes through the underlying client. The
    "project sandbox" is a logical concept; physically it's one
    workspace per ``(user, project, base_ref)`` keyed in ANALYSIS mode,
    sharing a single Daytona sandbox per ``(user, project)`` at the
    provider layer.

    Pass ``client`` for tests / embedded use; production callers can
    use the module-level :func:`get_project_sandbox` accessor which
    re-uses the process-wide :class:`SandboxClient`.
    """

    def __init__(self, client: SandboxClient | None = None) -> None:
        self._client = client or get_sandbox_client()

    async def ensure(
        self,
        *,
        user_id: str,
        project_id: str,
        repo: ProjectRef,
        auth_token: str | None = None,
    ) -> WorkspaceHandle:
        """Get-or-create the long-lived ANALYSIS workspace for this project.

        Returns a handle that's verified-alive at call time. Specifically:

        1. Resolve auth (caller-supplied wins; otherwise the local
           adapter's chain — env, GitHub App, user OAuth).
        2. Acquire the ANALYSIS workspace (idempotent on
           ``(user, project, repo, ANALYSIS, base_ref)``).
        3. Probe the backing storage. If dead, destroy the stale store
           record and re-acquire. The second acquire goes through the
           Daytona provider's label-based recovery — it adopts a
           live sandbox for ``(user, project)`` if one exists, otherwise
           mints a fresh one.

        The two-pass shape is deliberate: the recovery path costs an
        extra round-trip but keeps the happy path zero-overhead. The
        cheap ``is_alive`` probe (one filesystem stat / one Daytona
        SDK lookup) makes this affordable on every entry point.
        """
        token = auth_token or _resolve_auth_token(user_id, repo.repo_name)
        handle = await self._acquire(
            user_id=user_id,
            project_id=project_id,
            repo=repo,
            auth_token=token,
        )
        if await self._client.is_alive(handle):
            return handle

        logger.warning(
            "project sandbox %s/%s reported dead on ensure() — recovering",
            user_id,
            project_id,
        )
        # Drop the stale workspace from the store so the second acquire
        # routes through create rather than reusing the dead key.
        try:
            await self._client.destroy_workspace(handle)
        except Exception as exc:  # noqa: BLE001
            # The destroy can fail if the underlying sandbox is already
            # gone — which is exactly the case we're recovering from.
            # Log and proceed; the next acquire either adopts a live
            # sandbox via labels or mints a new one.
            logger.warning(
                "destroy_workspace during ensure recovery failed (%s); "
                "continuing — re-acquire will route through create",
                exc,
            )
        return await self._acquire(
            user_id=user_id,
            project_id=project_id,
            repo=repo,
            auth_token=token,
        )

    async def health_check(self, handle: WorkspaceHandle) -> bool:
        """Cheap liveness probe — returns ``True`` iff the handle is usable.

        Doesn't raise. Callers (conversation_service.store_message,
        REST middleware) treat ``False`` as "call ensure() before
        proceeding"; that keeps the policy out of every entry point.
        """
        return await self._client.is_alive(handle)

    async def parse(
        self,
        handle: WorkspaceHandle,
        *,
        repo_subdir: str | None = None,
        timeout_s: int = 600,
    ) -> "ParseArtifacts":
        """Run the in-sandbox parser and return the reconstructed graph.

        Thin wrapper on :meth:`SandboxClient.parse_repo` — exists so
        the parsing pipeline imports ProjectSandbox once and never has
        to know about the underlying client.
        """
        return await self._client.parse_repo(
            handle, repo_subdir=repo_subdir, timeout_s=timeout_s
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    async def _acquire(
        self,
        *,
        user_id: str,
        project_id: str,
        repo: ProjectRef,
        auth_token: str | None,
    ) -> WorkspaceHandle:
        """Single-call acquire helper used by both ensure and recovery.

        ANALYSIS-mode locks the workspace key to ``(..., base_ref)``
        so reparses at a new commit get a fresh workspace while the
        underlying Daytona sandbox stays the same. ``branch`` is set to
        ``base_ref`` so the worktree checks out the parsing target.
        """
        return await self._client.acquire_session(
            user_id=user_id,
            project_id=project_id,
            repo=repo.repo_name,
            repo_url=repo.repo_url,
            branch=repo.base_ref,
            base_ref=repo.base_ref,
            auth_token=auth_token,
            mode=WorkspaceMode.ANALYSIS,
        )


_facade: ProjectSandbox | None = None


def get_project_sandbox() -> ProjectSandbox:
    """Process-wide accessor. One facade per process, sharing the singleton client."""
    global _facade
    if _facade is None:
        _facade = ProjectSandbox()
    return _facade


def set_project_sandbox(facade: ProjectSandbox | None) -> None:
    """Override for tests / embedded use. Pass ``None`` to reset."""
    global _facade
    _facade = facade


__all__ = [
    "ProjectRef",
    "ProjectSandbox",
    "get_project_sandbox",
    "set_project_sandbox",
]

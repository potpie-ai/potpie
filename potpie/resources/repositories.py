"""Repositories resource for PotpieRuntime library.

Codegen / workflow worktree creation routed through the sandbox client.

Historically this resource wrapped :class:`RepoManager` and built worktrees by
hand under ``.repos/<owner>/<repo>/worktrees/codegen_<id>/``. The sandbox
client now owns that lifecycle: a single ``acquire_session`` call gives back
a :class:`WorkspaceHandle` that points at a per-``unique_id`` worktree forked
off the requested ``ref``. We just expose the handle's ``local_path`` to
preserve the public ``str``-returning contract.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from potpie.exceptions import PotpieError
from potpie.resources.base import BaseResource

if TYPE_CHECKING:
    from potpie.config import RuntimeConfig
    from potpie.core.database import DatabaseManager
    from potpie.core.neo4j import Neo4jManager

logger = logging.getLogger(__name__)


class RepositoriesResource(BaseResource):
    """Codegen worktree creation, sandbox-backed.

    The public surface (``create_worktree(repo_name, ref, user_id, unique_id)``)
    is unchanged; the internals went from RepoManager + raw ``git worktree``
    invocation to :func:`SandboxClient.acquire_session` so codegen and the
    agent harness share one cache and one set of locks.
    """

    def __init__(
        self,
        config: "RuntimeConfig",
        db_manager: "DatabaseManager",
        neo4j_manager: "Neo4jManager",
    ):
        super().__init__(config, db_manager, neo4j_manager)

    async def create_worktree(
        self,
        repo_name: str,
        ref: str,
        user_id: str,
        unique_id: str,
        *,
        exists_ok: bool = False,
    ) -> str:
        """Acquire (or look up) an isolated worktree for ``repo_name@ref``.

        Idempotent on ``(user_id, unique_id, repo_name, ref)`` — repeated
        calls return the same worktree until it's released. ``exists_ok`` is
        accepted for backwards compatibility but is now redundant: the
        sandbox client always returns the existing handle when the same
        key is presented.
        """
        if not repo_name or not ref:
            raise PotpieError("repo_name and ref are required")
        del exists_ok  # The sandbox client is idempotent; flag is a no-op.

        # ``project_id`` for the sandbox key — codegen runs aren't
        # tied to a Potpie project row, so we synthesise a stable id from
        # the unique_id passed in by the caller. Two codegen runs with the
        # same unique_id (e.g. retries) deliberately share the same
        # workspace, matching the legacy RepoManager behaviour.
        from app.modules.intelligence.tools.sandbox.client import (
            get_sandbox_client,
        )
        from sandbox import WorkspaceMode

        client = get_sandbox_client()
        try:
            handle = await client.acquire_session(
                user_id=user_id,
                project_id=f"codegen:{unique_id}",
                repo=repo_name,
                branch=ref,
                base_ref=ref,
                create_branch=False,
                mode=WorkspaceMode.EDIT,
                conversation_id=unique_id,
            )
        except Exception as exc:
            raise PotpieError(
                f"Failed to acquire codegen worktree for {repo_name}@{ref}: {exc}"
            ) from exc

        local = handle.local_path
        if local is None:
            raise PotpieError(
                f"Sandbox returned a workspace with no local path for "
                f"{repo_name}@{ref}; codegen requires a host-fs backend."
            )
        logger.info(
            "[RepositoriesResource] Sandbox worktree ready at %s for %s@%s",
            local,
            repo_name,
            ref,
        )
        return str(local)

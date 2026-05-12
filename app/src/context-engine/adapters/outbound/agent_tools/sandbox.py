"""Sandbox agent tools: read-only worktree access via ``SandboxClient``.

The agent gets a single workspace per batch run (lazily acquired on the first
sandbox tool call) so subsequent reads / list_dir / search calls don't pay the
acquire latency. The workspace is hibernated via a cleanup callback registered
on the run state so it tears down even when the agent crashes mid-loop.

The host wires this in via::

    agent.add_extra_tools([
        build_sandbox_tools(client_factory, session_config_for_pot),
    ])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class SandboxSessionConfig:
    """Identity for the workspace the agent should read from.

    Fields mirror :meth:`SandboxClient.acquire_session`. Pass ``mode=READ`` for
    agent ingestion; the sandbox library defaults to ``EDIT``.
    """

    user_id: str
    project_id: str
    repo_name: str
    branch: str
    base_ref: str | None = None
    auth_token: str | None = None
    repo_url: str | None = None
    conversation_id: str | None = None
    task_id: str | None = None


SessionConfigResolver = Callable[[str, str | None], SandboxSessionConfig | None]
"""``(pot_id, primary_repo_name) -> SandboxSessionConfig | None``.

Return ``None`` to disable sandbox tools for a pot (e.g. when the pot has no
attached repo). The host implementation typically looks up tenant + project +
default branch + auth token from its own DB.
"""


def build_sandbox_tools(
    *,
    client_factory: Callable[[], Awaitable[Any]],
    session_config_for_pot: SessionConfigResolver,
    max_read_bytes: int = 256_000,
    max_search_hits: int = 200,
) -> Callable[[Any], list[Any]]:
    """Return a per-batch tool builder that exposes sandbox read operations.

    Args:
        client_factory: Async factory that returns a ready ``SandboxClient``.
            Called at most once per batch run (on first sandbox tool call).
        session_config_for_pot: Resolves ``(pot_id, repo_name)`` to a
            :class:`SandboxSessionConfig`. Returning ``None`` disables sandbox
            tools for that batch.
        max_read_bytes: Hard cap on bytes returned by ``sandbox_read_file``.
        max_search_hits: Hard cap on hits returned by ``sandbox_search``.
    """

    def _builder(state: Any) -> list[Any]:
        try:
            from pydantic_ai import Tool  # type: ignore[import-not-found]
        except Exception:
            try:
                from pydantic_deep import Tool  # type: ignore[import-not-found, no-redef]
            except Exception:
                logger.warning(
                    "pydantic-ai/pydantic-deep Tool not importable; skipping sandbox tools"
                )
                return []

        cfg = session_config_for_pot(state.pot_id, state.repo_name)
        if cfg is None:
            logger.info(
                "sandbox tools disabled for pot %s (no session config)", state.pot_id
            )
            return []

        cache: dict[str, Any] = {"client": None, "handle": None}

        async def _ensure_handle():
            if cache["handle"] is not None:
                return cache["client"], cache["handle"]
            client = await client_factory()
            handle = await client.acquire_session(
                user_id=cfg.user_id,
                project_id=cfg.project_id,
                repo=cfg.repo_name,
                branch=cfg.branch,
                base_ref=cfg.base_ref,
                auth_token=cfg.auth_token,
                conversation_id=cfg.conversation_id,
                task_id=cfg.task_id,
                repo_url=cfg.repo_url,
            )
            cache["client"] = client
            cache["handle"] = handle
            return client, handle

        async def _release() -> None:
            client, handle = cache.get("client"), cache.get("handle")
            if client is None or handle is None:
                return
            try:
                await client.release_session(handle, destroy_runtime=False)
            except Exception:
                logger.exception(
                    "sandbox release_session failed for pot %s", state.pot_id
                )

        state.cleanup_callbacks.append(_release)

        async def sandbox_read_file(path: str) -> dict[str, Any]:
            """Read a UTF-8 file from the pot's sandbox worktree (capped to ~256KB)."""
            client, handle = await _ensure_handle()
            try:
                data = await client.read_file(
                    handle, path, max_bytes=max_read_bytes
                )
            except Exception as exc:
                return {"error": str(exc), "path": path}
            try:
                text = data.decode("utf-8")
                encoding = "utf-8"
            except UnicodeDecodeError:
                import base64

                text = base64.b64encode(data).decode("ascii")
                encoding = "base64"
            return {
                "path": path,
                "encoding": encoding,
                "size_bytes": len(data),
                "truncated": len(data) >= max_read_bytes,
                "content": text,
            }

        async def sandbox_list_dir(path: str = ".") -> dict[str, Any]:
            """List one directory level in the pot's sandbox worktree."""
            client, handle = await _ensure_handle()
            try:
                entries = await client.list_dir(handle, path)
            except Exception as exc:
                return {"error": str(exc), "path": path}
            return {
                "path": path,
                "entries": [
                    {
                        "name": e.name,
                        "is_dir": bool(e.is_dir),
                        "size": e.size,
                    }
                    for e in entries
                ],
            }

        async def sandbox_search(
            pattern: str,
            glob: str | None = None,
            case_sensitive: bool = False,
        ) -> dict[str, Any]:
            """Ripgrep across the pot's worktree (capped to ~200 hits)."""
            client, handle = await _ensure_handle()
            try:
                hits = await client.search(
                    handle,
                    pattern,
                    glob=glob,
                    case=case_sensitive,
                    max_hits=max_search_hits,
                )
            except Exception as exc:
                return {"error": str(exc), "pattern": pattern}
            return {
                "pattern": pattern,
                "hit_count": len(hits),
                "truncated": len(hits) >= max_search_hits,
                "hits": [
                    {
                        "path": getattr(h, "path", None),
                        "line": getattr(h, "line_number", None),
                        "snippet": getattr(h, "line", None),
                    }
                    for h in hits
                ],
            }

        return [
            Tool(
                sandbox_read_file,
                name="sandbox_read_file",
                description=(
                    "Read a file from the pot's sandbox worktree by repo-relative path. "
                    "Returns up to ~256KB; binary files come back base64-encoded."
                ),
            ),
            Tool(
                sandbox_list_dir,
                name="sandbox_list_dir",
                description=(
                    "List one directory level (no recursion) in the pot's sandbox "
                    "worktree by repo-relative path. Default is the repo root."
                ),
            ),
            Tool(
                sandbox_search,
                name="sandbox_search",
                description=(
                    "Ripgrep across the pot's sandbox worktree. Optional glob filter "
                    "(e.g. '*.py') and case_sensitive flag. Returns up to ~200 hits."
                ),
            ),
        ]

    return _builder

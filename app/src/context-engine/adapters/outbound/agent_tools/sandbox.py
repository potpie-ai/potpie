"""Sandbox agent tools — pot-scoped, multi-repo worktree access.

The reconciliation agent walks every repo attached to a pot via a single
:class:`PotSandboxFacade` per batch run. The facade holds one
``SandboxClient`` workspace per ``(pot_id, repo)`` and serialises
``sandbox_checkout`` calls per repo so two events can't race on the same
worktree.

The host wires this in via::

    agent.add_extra_tools([
        build_sandbox_tools(
            client_factory=...,
            pot_resolver=...,
        ),
    ])

A legacy single-repo resolver (``session_config_for_pot``) is still
accepted — it's adapted into a one-repo :class:`PotSandboxConfig` so the
agent surface stays uniform.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from adapters.outbound.agent_tools._path_safety import is_safe_relpath
from domain.error_redaction import safe_error

logger = logging.getLogger(__name__)


def _is_not_found_error(exc: BaseException) -> bool:
    """True for a backend "path does not exist" error.

    A missing file/dir is a normal answer to an agent probe, not an
    infrastructure fault — callers use this to return a structured
    ``not_found`` result and log quietly instead of emitting an
    ERROR-level traceback that looks like the sandbox is broken.

    Detected backend-agnostically (this adapter talks to the abstract
    ``SandboxClient``, never the SDK): by class name — Daytona raises
    ``DaytonaNotFoundError`` — and by message for the exec backends
    (Docker/local raise ``SandboxOpError("file not found: …")``).
    """
    if "notfound" in type(exc).__name__.lower():
        return True
    message = str(exc).lower()
    return "not found" in message or "no such file" in message


@dataclass(slots=True, frozen=True)
class SandboxSessionConfig:
    """Legacy single-repo session identity.

    Kept for callers that haven't migrated to :class:`PotSandboxConfig`.
    Internally adapted to a one-repo :class:`PotSandboxConfig` so the tool
    surface is the same regardless of which resolver shape the host passes.
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
"""``(pot_id, primary_repo_name) -> SandboxSessionConfig | None`` (legacy)."""


@dataclass(slots=True, frozen=True)
class RepoAttachment:
    """One repo cloned into the pot's sandbox.

    ``auth_kind`` records which branch of the token chain produced
    ``auth_token`` (``"app"`` for a GitHub-App installation token,
    ``"user_oauth"`` for a user's OAuth, ``"env"`` for the env-var
    fallback, ``"none"`` when no token was found). The agent never reads
    it, but it surfaces in telemetry so operators can see which auth path
    the sandbox actually used.
    """

    owner: str
    repo: str
    default_branch: str
    repo_url: str | None = None
    auth_token: str | None = None
    auth_kind: str | None = None

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"


@dataclass(slots=True, frozen=True)
class PotSandboxConfig:
    """Identity for the pot-scoped sandbox the agent should read from."""

    user_id: str
    pot_id: str
    provider_host: str
    repos: list[RepoAttachment] = field(default_factory=list)


PotSandboxResolver = Callable[[str], PotSandboxConfig | None]
"""``(pot_id) -> PotSandboxConfig | None``.

Return ``None`` to disable sandbox tools for a pot (e.g. when the pot has
no attached repos). The host implementation typically loads every row from
``context_graph_pot_repositories`` plus a per-repo auth token.
"""


def _adapt_legacy_resolver(
    legacy: SessionConfigResolver,
) -> PotSandboxResolver:
    """Wrap a single-repo resolver as a one-repo :class:`PotSandboxConfig`."""

    def _adapter(pot_id: str) -> PotSandboxConfig | None:
        cfg = legacy(pot_id, None)
        if cfg is None:
            return None
        # ``repo_name`` is ``owner/repo``; split for the attachment.
        parts = cfg.repo_name.split("/", 1)
        if len(parts) != 2:
            return None
        return PotSandboxConfig(
            user_id=cfg.user_id,
            pot_id=cfg.project_id,
            provider_host="github.com",
            repos=[
                RepoAttachment(
                    owner=parts[0],
                    repo=parts[1],
                    default_branch=cfg.branch or "main",
                    repo_url=cfg.repo_url,
                    auth_token=cfg.auth_token,
                )
            ],
        )

    return _adapter


class _AmbiguousRepoError(LookupError):
    """Raised when an agent omits ``repo=`` on a multi-repo pot sandbox."""


class _UnknownRepoError(LookupError):
    """Raised when the agent passes a ``repo=`` not attached to the pot."""


class PotSandboxFacade:
    """Pot-scoped, multi-repo wrapper over :class:`SandboxClient`.

    One facade per agent batch run. Acquires workspaces lazily on first
    access per repo, hibernates all of them on ``release_all``. Per-repo
    asyncio locks serialise ``sandbox_checkout`` so two events can't race
    on the same worktree.
    """

    def __init__(self, *, client: Any, cfg: PotSandboxConfig) -> None:
        self._client = client
        self._cfg = cfg
        self._workspaces: dict[str, Any] = {}
        self._workspace_locks: dict[str, asyncio.Lock] = {}
        self._repo_locks: dict[str, asyncio.Lock] = {}
        self._by_full_name: dict[str, RepoAttachment] = {
            r.full_name: r for r in cfg.repos
        }

    @property
    def cfg(self) -> PotSandboxConfig:
        return self._cfg

    def list_repos(self) -> list[RepoAttachment]:
        return list(self._cfg.repos)

    def resolve_repo(self, repo: str | None) -> RepoAttachment:
        """Pick the attachment for ``repo``, or the sole one if ``repo`` is None.

        Raises:
            _AmbiguousRepoError: ``repo`` is None and the pot has >1 repo.
            _UnknownRepoError: ``repo`` is not attached to the pot.
        """
        if repo is None:
            if len(self._cfg.repos) == 1:
                return self._cfg.repos[0]
            if not self._cfg.repos:
                raise _UnknownRepoError("pot has no repos attached")
            raise _AmbiguousRepoError(
                "multiple repos attached; pass repo='owner/name'"
            )
        match = self._by_full_name.get(repo)
        if match is None:
            raise _UnknownRepoError(repo)
        return match

    async def acquire(self, repo: str | None) -> tuple[RepoAttachment, Any]:
        """Resolve the attachment + (cached) :class:`WorkspaceHandle`."""
        attachment = self.resolve_repo(repo)
        full = attachment.full_name
        lock = self._workspace_locks.setdefault(full, asyncio.Lock())
        async with lock:
            handle = self._workspaces.get(full)
            if handle is not None:
                return attachment, handle
            from adapters.outbound.agent_tools._sandbox_metrics import record
            import time

            started = time.monotonic()
            handle = await self._client.acquire_session(
                user_id=self._cfg.user_id,
                project_id=self._cfg.pot_id,
                repo=full,
                branch=attachment.default_branch,
                base_ref=attachment.default_branch,
                auth_token=attachment.auth_token,
                repo_url=attachment.repo_url,
            )
            elapsed_ms = int((time.monotonic() - started) * 1000)
            record(
                "pot_sandbox.attach",
                {
                    "pot_id": self._cfg.pot_id,
                    "repo": full,
                    "auth_kind": attachment.auth_kind or "unknown",
                },
            )
            record(
                "pot_sandbox.cold_start_ms",
                {"pot_id": self._cfg.pot_id, "repo": full},
                value=elapsed_ms,
            )
            self._workspaces[full] = handle
            return attachment, handle

    def repo_lock(self, repo: str) -> asyncio.Lock:
        """Per-repo lock for ops that mutate worktree state (checkout etc.)."""
        return self._repo_locks.setdefault(repo, asyncio.Lock())

    async def release_all(self) -> None:
        for full, handle in list(self._workspaces.items()):
            try:
                await self._client.release_session(
                    handle, destroy_runtime=False
                )
            except Exception:
                logger.exception(
                    "release_session failed for pot=%s repo=%s",
                    self._cfg.pot_id,
                    full,
                )
        self._workspaces.clear()


def _coerce_resolver(
    *,
    pot_resolver: PotSandboxResolver | None,
    session_config_for_pot: SessionConfigResolver | None,
) -> PotSandboxResolver | None:
    if pot_resolver is not None:
        return pot_resolver
    if session_config_for_pot is not None:
        return _adapt_legacy_resolver(session_config_for_pot)
    return None


def build_sandbox_tools(
    *,
    client_factory: Callable[[], Awaitable[Any]],
    pot_resolver: PotSandboxResolver | None = None,
    session_config_for_pot: SessionConfigResolver | None = None,
    max_read_bytes: int = 256_000,
    max_search_hits: int = 200,
) -> Callable[[Any], list[Any]]:
    """Return a per-batch tool builder for pot-scoped sandbox access.

    Args:
        client_factory: Async factory that returns a ready ``SandboxClient``.
            Called at most once per batch run (on first sandbox tool call).
        pot_resolver: ``(pot_id) -> PotSandboxConfig | None``. Multi-repo,
            preferred. Returning ``None`` disables sandbox tools for the batch.
        session_config_for_pot: Legacy single-repo resolver. Used iff
            ``pot_resolver`` is omitted.
        max_read_bytes: Hard cap on bytes returned by ``sandbox_read_file``.
        max_search_hits: Hard cap on hits returned by ``sandbox_search``.

    Raises:
        ValueError: neither resolver was provided.
    """
    resolver = _coerce_resolver(
        pot_resolver=pot_resolver,
        session_config_for_pot=session_config_for_pot,
    )
    if resolver is None:
        raise ValueError(
            "build_sandbox_tools requires pot_resolver or "
            "session_config_for_pot"
        )

    def _builder(state: Any) -> list[Any]:
        try:
            from pydantic_ai import Tool  # type: ignore[import-not-found]
        except Exception:
            try:
                from pydantic_deep import Tool  # type: ignore[import-not-found, no-redef]
            except Exception:
                logger.warning(
                    "pydantic-ai/pydantic-deep Tool not importable; "
                    "skipping sandbox tools"
                )
                return []

        cfg = resolver(state.pot_id)
        if cfg is None or not cfg.repos:
            logger.info(
                "sandbox tools disabled for pot %s (no pot sandbox config)",
                state.pot_id,
            )
            return []

        facade_box: dict[str, Any] = {"facade": None, "client": None}

        async def _ensure_facade() -> PotSandboxFacade:
            if facade_box["facade"] is not None:
                return facade_box["facade"]
            client = await client_factory()
            facade = PotSandboxFacade(client=client, cfg=cfg)
            facade_box["client"] = client
            facade_box["facade"] = facade
            return facade

        async def _release() -> None:
            facade = facade_box.get("facade")
            if facade is None:
                return
            await facade.release_all()

        state.cleanup_callbacks.append(_release)

        def _ambiguous_error(exc: _AmbiguousRepoError) -> dict[str, Any]:
            return {
                "error": "ambiguous_repo",
                "message": safe_error(exc),
                "available": [r.full_name for r in cfg.repos],
            }

        def _unknown_error(exc: _UnknownRepoError) -> dict[str, Any]:
            return {
                "error": "unknown_repo",
                "message": safe_error(exc),
                "available": [r.full_name for r in cfg.repos],
            }

        def _sandbox_unavailable_error(
            exc: Exception, **extra: Any
        ) -> dict[str, Any]:
            # Infrastructure failure while attaching to the pot's sandbox
            # (Daytona/SDK timeouts, connection resets, snapshot pulls, …).
            # Contained at the tool boundary so one transient blip doesn't
            # kill the whole batch — the agent can retry or fall back.
            return {
                "error": safe_error(exc),
                "kind": "sandbox_unavailable",
                "transient": True,
                **extra,
            }

        from adapters.outbound.agent_tools._sandbox_metrics import record as _metric

        def _tool_call(name: str, ok: bool) -> None:
            _metric(
                "pot_sandbox.tool_call",
                {"name": name, "ok": ok, "pot_id": cfg.pot_id},
            )

        async def sandbox_list_repos() -> dict[str, Any]:
            """List every repo attached to this pot's sandbox."""
            _tool_call("sandbox_list_repos", True)
            return {
                "pot_id": cfg.pot_id,
                "repos": [
                    {
                        "repo": r.full_name,
                        "default_branch": r.default_branch,
                    }
                    for r in cfg.repos
                ],
            }

        async def sandbox_read_file(
            path: str, repo: str | None = None
        ) -> dict[str, Any]:
            """Read a UTF-8 file from a pot repo's worktree (capped to ~256KB)."""
            if not is_safe_relpath(path):
                _tool_call("sandbox_read_file", False)
                return {
                    "error": "invalid path",
                    "kind": "invalid_argument",
                    "path": path,
                }
            try:
                facade = await _ensure_facade()
                attachment, handle = await facade.acquire(repo)
            except _AmbiguousRepoError as exc:
                return _ambiguous_error(exc)
            except _UnknownRepoError as exc:
                return _unknown_error(exc)
            except Exception as exc:
                logger.exception(
                    "sandbox_read_file: acquire failed for pot=%s repo=%s",
                    cfg.pot_id,
                    repo,
                )
                _tool_call("sandbox_read_file", False)
                return _sandbox_unavailable_error(
                    exc, path=path, repo=repo
                )
            try:
                data = await facade_box["client"].read_file(
                    handle, path, max_bytes=max_read_bytes
                )
            except Exception as exc:
                _tool_call("sandbox_read_file", False)
                if _is_not_found_error(exc):
                    logger.debug(
                        "sandbox_read_file: not found pot=%s repo=%s path=%s",
                        cfg.pot_id,
                        attachment.full_name,
                        path,
                    )
                    return {
                        "error": "file not found",
                        "kind": "not_found",
                        "path": path,
                        "repo": attachment.full_name,
                    }
                logger.exception(
                    "sandbox_read_file: read failed for pot=%s repo=%s path=%s",
                    cfg.pot_id,
                    attachment.full_name,
                    path,
                )
                return {
                    "error": safe_error(exc),
                    "path": path,
                    "repo": attachment.full_name,
                }
            try:
                text = data.decode("utf-8")
                encoding = "utf-8"
            except UnicodeDecodeError:
                import base64

                text = base64.b64encode(data).decode("ascii")
                encoding = "base64"
            return {
                "repo": attachment.full_name,
                "path": path,
                "encoding": encoding,
                "size_bytes": len(data),
                "truncated": len(data) >= max_read_bytes,
                "content": text,
            }

        async def sandbox_list_dir(
            path: str = ".", repo: str | None = None
        ) -> dict[str, Any]:
            """List one directory level in a pot repo's worktree."""
            if not is_safe_relpath(path):
                _tool_call("sandbox_list_dir", False)
                return {
                    "error": "invalid path",
                    "kind": "invalid_argument",
                    "path": path,
                }
            try:
                facade = await _ensure_facade()
                attachment, handle = await facade.acquire(repo)
            except _AmbiguousRepoError as exc:
                return _ambiguous_error(exc)
            except _UnknownRepoError as exc:
                return _unknown_error(exc)
            except Exception as exc:
                logger.exception(
                    "sandbox_list_dir: acquire failed for pot=%s repo=%s",
                    cfg.pot_id,
                    repo,
                )
                _tool_call("sandbox_list_dir", False)
                return _sandbox_unavailable_error(
                    exc, path=path, repo=repo
                )
            try:
                entries = await facade_box["client"].list_dir(handle, path)
            except Exception as exc:
                _tool_call("sandbox_list_dir", False)
                if _is_not_found_error(exc):
                    logger.debug(
                        "sandbox_list_dir: not found pot=%s repo=%s path=%s",
                        cfg.pot_id,
                        attachment.full_name,
                        path,
                    )
                    return {
                        "error": "directory not found",
                        "kind": "not_found",
                        "path": path,
                        "repo": attachment.full_name,
                    }
                logger.exception(
                    "sandbox_list_dir: list failed for pot=%s repo=%s path=%s",
                    cfg.pot_id,
                    attachment.full_name,
                    path,
                )
                return {
                    "error": safe_error(exc),
                    "path": path,
                    "repo": attachment.full_name,
                }
            return {
                "repo": attachment.full_name,
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
            repo: str | None = None,
        ) -> dict[str, Any]:
            """Ripgrep across a pot repo's worktree (capped to ~200 hits)."""
            try:
                facade = await _ensure_facade()
                attachment, handle = await facade.acquire(repo)
            except _AmbiguousRepoError as exc:
                return _ambiguous_error(exc)
            except _UnknownRepoError as exc:
                return _unknown_error(exc)
            except Exception as exc:
                logger.exception(
                    "sandbox_search: acquire failed for pot=%s repo=%s",
                    cfg.pot_id,
                    repo,
                )
                _tool_call("sandbox_search", False)
                return _sandbox_unavailable_error(
                    exc, pattern=pattern, repo=repo
                )
            try:
                hits = await facade_box["client"].search(
                    handle,
                    pattern,
                    glob=glob,
                    case=case_sensitive,
                    max_hits=max_search_hits,
                )
            except Exception as exc:
                logger.exception(
                    "sandbox_search: search failed for pot=%s repo=%s pattern=%s",
                    cfg.pot_id,
                    attachment.full_name,
                    pattern,
                )
                _tool_call("sandbox_search", False)
                return {
                    "error": safe_error(exc),
                    "pattern": pattern,
                    "repo": attachment.full_name,
                }
            return {
                "repo": attachment.full_name,
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

        def _session_payload(attachment: Any, result: Any) -> dict[str, Any]:
            return {
                "repo": attachment.full_name,
                "session_id": result.session_id,
                "running": bool(result.running),
                "exit_code": result.exit_code,
                "output": result.output,
                "truncated": result.truncated,
            }

        async def sandbox_exec(
            command: str,
            repo: str | None = None,
            tty: bool = False,
            yield_time_ms: int = 10_000,
            max_output_bytes: int | None = None,
        ) -> dict[str, Any]:
            """Start a streamable command in a pot repo's worktree (read-only)."""
            try:
                facade = await _ensure_facade()
                attachment, handle = await facade.acquire(repo)
            except _AmbiguousRepoError as exc:
                return _ambiguous_error(exc)
            except _UnknownRepoError as exc:
                return _unknown_error(exc)
            except Exception as exc:
                logger.exception(
                    "sandbox_exec: acquire failed for pot=%s repo=%s",
                    cfg.pot_id,
                    repo,
                )
                _tool_call("sandbox_exec", False)
                return _sandbox_unavailable_error(exc, repo=repo)
            from sandbox.domain.models import CommandKind

            try:
                result = await facade_box["client"].exec_start(
                    handle,
                    [command],
                    shell=True,
                    tty=tty,
                    yield_time_ms=yield_time_ms,
                    max_output_bytes=max_output_bytes,
                    command_kind=CommandKind.READ,
                )
            except Exception as exc:
                _tool_call("sandbox_exec", False)
                if _is_not_found_error(exc):
                    return {
                        "error": safe_error(exc),
                        "kind": "not_found",
                        "repo": attachment.full_name,
                    }
                logger.exception(
                    "sandbox_exec: start failed for pot=%s repo=%s",
                    cfg.pot_id,
                    attachment.full_name,
                )
                return {
                    "error": safe_error(exc),
                    "repo": attachment.full_name,
                    "command": command,
                }
            facade_box.setdefault("exec_sessions", {})[result.session_id] = (
                attachment.full_name
            )
            _tool_call("sandbox_exec", True)
            return _session_payload(attachment, result)

        async def sandbox_write_stdin(
            session_id: str,
            chars: str = "",
            kill: bool = False,
            yield_time_ms: int = 10_000,
            max_output_bytes: int | None = None,
        ) -> dict[str, Any]:
            """Write stdin / poll / kill a sandbox_exec session by id."""
            repo = facade_box.get("exec_sessions", {}).get(session_id)
            if repo is None:
                return {
                    "error": "unknown or expired session_id",
                    "kind": "not_found",
                    "session_id": session_id,
                }
            try:
                facade = await _ensure_facade()
                attachment, handle = await facade.acquire(repo)
            except Exception as exc:
                logger.exception(
                    "sandbox_write_stdin: acquire failed for pot=%s repo=%s",
                    cfg.pot_id,
                    repo,
                )
                _tool_call("sandbox_write_stdin", False)
                return _sandbox_unavailable_error(exc, session_id=session_id)
            client = facade_box["client"]
            try:
                if kill:
                    await client.exec_kill(handle, session_id)
                    facade_box.get("exec_sessions", {}).pop(session_id, None)
                    _tool_call("sandbox_write_stdin", True)
                    return {
                        "repo": attachment.full_name,
                        "session_id": session_id,
                        "running": False,
                        "killed": True,
                    }
                result = await client.exec_write_stdin(
                    handle,
                    session_id,
                    chars,
                    yield_time_ms=yield_time_ms,
                    max_output_bytes=max_output_bytes,
                )
            except Exception as exc:
                _tool_call("sandbox_write_stdin", False)
                if _is_not_found_error(exc):
                    facade_box.get("exec_sessions", {}).pop(session_id, None)
                    return {
                        "error": safe_error(exc),
                        "kind": "not_found",
                        "session_id": session_id,
                    }
                logger.exception(
                    "sandbox_write_stdin: failed for pot=%s session=%s",
                    cfg.pot_id,
                    session_id,
                )
                return {"error": safe_error(exc), "session_id": session_id}
            if not result.running:
                facade_box.get("exec_sessions", {}).pop(session_id, None)
            _tool_call("sandbox_write_stdin", True)
            return _session_payload(attachment, result)

        # Phase 4 git-history tools are appended here in the same builder.
        from adapters.outbound.agent_tools._sandbox_git_tools import (
            build_git_history_tools,
        )

        base = [
            Tool(
                sandbox_list_repos,
                name="sandbox_list_repos",
                description=(
                    "List every repo attached to this pot's sandbox. Returns "
                    "[{repo, default_branch}]. Call this first when more than "
                    "one repo may be attached."
                ),
            ),
            Tool(
                sandbox_read_file,
                name="sandbox_read_file",
                description=(
                    "Read a file from a pot repo's sandbox worktree by "
                    "repo-relative path. Pass repo='owner/name' when multiple "
                    "repos are attached. Returns up to ~256KB; binary files "
                    "come back base64-encoded."
                ),
            ),
            Tool(
                sandbox_list_dir,
                name="sandbox_list_dir",
                description=(
                    "List one directory level (no recursion) in a pot repo's "
                    "sandbox worktree by repo-relative path. Default is the "
                    "repo root. Pass repo='owner/name' when multiple repos "
                    "are attached."
                ),
            ),
            Tool(
                sandbox_search,
                name="sandbox_search",
                description=(
                    "Ripgrep across a pot repo's sandbox worktree. Optional "
                    "glob filter (e.g. '*.py') and case_sensitive flag. "
                    "Returns up to ~200 hits. Pass repo='owner/name' when "
                    "multiple repos are attached."
                ),
            ),
            Tool(
                sandbox_exec,
                name="sandbox_exec",
                description=(
                    "Start a streamable / long-running command in a pot repo's "
                    "worktree and read its progress. Returns after a short "
                    "window even if still running, handing back a session_id. "
                    "Use for builds, test runs, or interactive probes; set "
                    "tty=true for programs that need a terminal. Returns "
                    "{session_id, output, running, exit_code, truncated}. If "
                    "running is true, drive it with sandbox_write_stdin. Pass "
                    "repo='owner/name' when multiple repos are attached. "
                    "For any text/code search use ripgrep ('rg -n PATTERN', "
                    "'rg -t py PATTERN', 'rg -l PATTERN | head') — never "
                    "'grep -r' or 'find ... -exec grep'; rg is preinstalled, "
                    "faster, and honors .gitignore ('fd', 'jq', 'tree', 'git', "
                    "'python3', 'node' are also available). Prefer sandbox_search "
                    "for simple searches; reach here only for pipes / flags it "
                    "doesn't expose."
                ),
            ),
            Tool(
                sandbox_write_stdin,
                name="sandbox_write_stdin",
                description=(
                    "Drive a running sandbox_exec session by session_id: write "
                    "to its stdin (chars; add a trailing newline to submit a "
                    "line), poll for more output (empty chars), or stop it "
                    "(kill=true). Returns the same shape as sandbox_exec. Poll "
                    "until running is false to capture the exit_code."
                ),
            ),
        ]
        git_tools = build_git_history_tools(
            Tool=Tool,
            ensure_facade=_ensure_facade,
            facade_box=facade_box,
            ambiguous_error=_ambiguous_error,
            unknown_error=_unknown_error,
            sandbox_unavailable_error=_sandbox_unavailable_error,
        )
        return base + git_tools

    return _builder

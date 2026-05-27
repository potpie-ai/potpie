"""Git-history tools the reconciliation agent uses to walk a pot's repos.

Composed into the sandbox tool set by ``build_sandbox_tools``. Every tool
takes an optional ``repo='owner/name'`` argument (required when the pot has
multiple repos attached) and returns a structured payload so the agent can
reason about failure modes without parsing free-text error strings.

Conventions:

* All read-only tools run ``client.exec`` with the default ``CommandKind.READ``.
* ``sandbox_checkout`` is the one mutator — it takes the facade's per-repo
  asyncio lock so two events can't race on the same worktree.
* Byte caps apply uniformly: 4 MB for diffs, 1 MB for show, 200 KB for
  log/blame. Truncation is signalled with ``truncated: true`` and the agent
  is expected to narrow the query.
"""

from __future__ import annotations

import logging

from observability import get_logger
from typing import Any, Awaitable, Callable

from adapters.outbound.agent_tools._path_safety import (
    is_safe_date_expr,
    is_safe_git_ref,
    is_safe_relpath,
)
from domain.error_redaction import safe_error

logger = get_logger(__name__)

# Hardening flags prepended to every git invocation: kill the ``ext::`` /
# ``file::`` / fd transports and any per-invocation config override so a
# crafted ref/remote can't escape the sandbox (security review H-4).
_GIT_HARDENING = [
    "-c",
    "protocol.ext.allow=never",
    "-c",
    "protocol.file.allow=never",
    "-c",
    "protocol.fd.allow=never",
]


def _invalid_arg_error(field: str, value: Any) -> dict[str, Any]:
    """Structured rejection for an unsafe agent-supplied git arg/path."""
    return {
        "error": f"invalid {field}",
        "kind": "invalid_argument",
        "field": field,
        "value": value,
    }


_LOG_BYTES = 200_000
_BLAME_BYTES = 200_000
_SHOW_BYTES = 1_000_000
_DIFF_BYTES = 4_000_000

_DEFAULT_LOG_LIMIT = 50
_MAX_LOG_LIMIT = 500


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _err_text(result: Any) -> str:
    return _decode(getattr(result, "stderr", b"")) or _decode(
        getattr(result, "stdout", b"")
    )


def _classify_git_error(stderr: str) -> str:
    s = stderr.lower()
    if (
        "unknown revision" in s
        or "bad revision" in s
        or "not a valid object" in s
        or "couldn't find remote ref" in s
        or "couldn't find remote branch" in s
        or "did not match any file" in s
        or "ambiguous argument" in s
        or "pathspec" in s and "did not match" in s
    ):
        return "unknown_ref"
    if "would be overwritten" in s or "would clobber" in s or "conflict" in s:
        return "conflict"
    if (
        "could not resolve host" in s
        or "fatal: unable to access" in s
        or "remote: invalid" in s
    ):
        return "network"
    if "permission denied" in s or "authentication failed" in s:
        return "auth"
    return "git_error"


def build_git_history_tools(
    *,
    Tool: Any,
    ensure_facade: Callable[[], Awaitable[Any]],
    facade_box: dict[str, Any],
    ambiguous_error: Callable[..., dict[str, Any]],
    unknown_error: Callable[..., dict[str, Any]],
    sandbox_unavailable_error: Callable[..., dict[str, Any]],
) -> list[Any]:
    """Return ``[Tool(...)]`` for ``sandbox_checkout`` + ``sandbox_git_*``.

    Wired alongside ``sandbox_read_file`` / ``sandbox_list_dir`` /
    ``sandbox_search`` by ``build_sandbox_tools``. Imports the facade error
    types lazily to avoid circular module references.

    ``sandbox_unavailable_error`` is the shared infra-failure formatter; the
    git tools call it when ``facade.acquire`` fails so a transient Daytona
    blip surfaces as a structured tool result instead of killing the batch.
    """
    from adapters.outbound.agent_tools._sandbox_metrics import record as _metric
    from adapters.outbound.agent_tools.sandbox import (
        _AmbiguousRepoError,
        _UnknownRepoError,
    )

    async def _run_git(
        repo: str | None,
        cmd: list[str],
        *,
        write: bool = False,
        max_output_bytes: int | None = None,
    ) -> tuple[Any, Any, dict[str, Any] | None]:
        """Resolve workspace, run command. Returns (attachment, result, error_payload)."""
        try:
            facade = await ensure_facade()
            attachment, handle = await facade.acquire(repo)
        except _AmbiguousRepoError as exc:
            return None, None, ambiguous_error(exc)
        except _UnknownRepoError as exc:
            return None, None, unknown_error(exc)
        except Exception as exc:
            logger.exception(
                "git tool: sandbox acquire failed for repo=%s", repo
            , repo=repo)
            return None, None, sandbox_unavailable_error(exc, repo=repo)

        from sandbox.domain.models import CommandKind  # type: ignore[import-not-found]

        if cmd and cmd[0] == "git":
            cmd = ["git", *_GIT_HARDENING, *cmd[1:]]
        try:
            result = await facade_box["client"].exec(
                handle,
                cmd,
                command_kind=CommandKind.WRITE if write else CommandKind.READ,
                max_output_bytes=max_output_bytes,
            )
        except Exception as exc:
            return attachment, None, {
                "error": safe_error(exc),
                "kind": "exec_failed",
                "repo": attachment.full_name,
            }
        return attachment, result, None

    async def sandbox_checkout(
        ref: str, repo: str | None = None, force: bool = False
    ) -> dict[str, Any]:
        """Detach HEAD onto ``ref`` for ``repo``.

        Fetches ``ref`` from ``origin`` first so freshly-pushed refs land.
        Holds the per-repo asyncio lock so two batches can't race the same
        worktree. Errors return ``{"error": ..., "kind": "unknown_ref" |
        "conflict" | "network" | "auth" | "git_error"}``.
        """
        if not is_safe_git_ref(ref):
            return _invalid_arg_error("ref", ref)
        try:
            facade = await ensure_facade()
            attachment = facade.resolve_repo(repo)
        except _AmbiguousRepoError as exc:
            return ambiguous_error(exc)
        except _UnknownRepoError as exc:
            return unknown_error(exc)
        except Exception as exc:
            logger.exception(
                "sandbox_checkout: facade init failed for repo=%s", repo
            , repo=repo)
            return sandbox_unavailable_error(exc, ref=ref, repo=repo)

        async with facade.repo_lock(attachment.full_name):
            from sandbox.domain.models import CommandKind  # type: ignore[import-not-found]

            try:
                _, handle = await facade.acquire(attachment.full_name)
            except Exception as exc:
                logger.exception(
                    "sandbox_checkout: acquire failed for repo=%s",
                    attachment.full_name,
                 repo=attachment.full_name)
                return sandbox_unavailable_error(
                    exc, ref=ref, repo=attachment.full_name
                )
            client = facade_box["client"]
            fetch = await client.exec(
                handle,
                ["git", *_GIT_HARDENING, "fetch", "origin", ref],
                command_kind=CommandKind.WRITE,
            )
            if fetch.exit_code != 0:
                stderr = _err_text(fetch)
                return {
                    "error": stderr.strip() or "fetch failed",
                    "kind": _classify_git_error(stderr),
                    "ref": ref,
                    "repo": attachment.full_name,
                }
            checkout_cmd = ["git", *_GIT_HARDENING, "checkout", "--detach"]
            if force:
                checkout_cmd.append("--force")
            # Trailing ``--`` forces ``ref`` to be parsed as a revision,
            # never an option or pathspec.
            checkout_cmd.extend([ref, "--"])
            co = await client.exec(
                handle, checkout_cmd, command_kind=CommandKind.WRITE
            )
            if co.exit_code != 0:
                stderr = _err_text(co)
                return {
                    "error": stderr.strip() or "checkout failed",
                    "kind": _classify_git_error(stderr),
                    "ref": ref,
                    "repo": attachment.full_name,
                }
            head = await client.exec(
                handle,
                ["git", "rev-parse", "HEAD"],
                command_kind=CommandKind.READ,
            )
            head_sha = _decode(head.stdout).strip() if head.exit_code == 0 else None
            _metric(
                "pot_sandbox.checkout",
                {"repo": attachment.full_name, "ok": True},
            )
            return {
                "repo": attachment.full_name,
                "ref": ref,
                "head_sha": head_sha,
            }

    async def sandbox_git_log(
        repo: str | None = None,
        path: str | None = None,
        since: str | None = None,
        limit: int = _DEFAULT_LOG_LIMIT,
    ) -> dict[str, Any]:
        """``git log`` for ``repo``; returns parsed commits.

        ``path`` narrows the log to a file/dir. ``since`` is any git-parseable
        date expression (``"2026-01-01"``, ``"2 weeks ago"``). ``limit`` is
        capped at 500 entries.
        """
        if since is not None and not is_safe_date_expr(since):
            return _invalid_arg_error("since", since)
        if path is not None and not is_safe_relpath(path):
            return _invalid_arg_error("path", path)
        capped = max(1, min(int(limit), _MAX_LOG_LIMIT))
        sep = "\x1f"  # ASCII unit separator — safe inside commit messages
        cmd = [
            "git",
            "log",
            f"--max-count={capped}",
            f"--pretty=format:%H{sep}%an{sep}%ae{sep}%aI{sep}%s",
        ]
        if since:
            cmd.append(f"--since={since}")
        if path:
            cmd.extend(["--", path])

        attachment, result, err = await _run_git(
            repo, cmd, max_output_bytes=_LOG_BYTES
        )
        if err is not None:
            return err
        if result.exit_code != 0:
            stderr = _err_text(result)
            return {
                "error": stderr.strip() or "git log failed",
                "kind": _classify_git_error(stderr),
                "repo": attachment.full_name,
            }
        commits: list[dict[str, str]] = []
        for line in _decode(result.stdout).splitlines():
            parts = line.split(sep)
            if len(parts) != 5:
                continue
            commits.append(
                {
                    "commit": parts[0],
                    "author": parts[1],
                    "email": parts[2],
                    "iso_date": parts[3],
                    "subject": parts[4],
                }
            )
        return {
            "repo": attachment.full_name,
            "commits": commits,
            "count": len(commits),
            "truncated": len(result.stdout) >= _LOG_BYTES,
        }

    async def sandbox_git_show(
        ref: str, repo: str | None = None, path: str | None = None
    ) -> dict[str, Any]:
        """``git show <ref>`` or ``git show <ref> -- <path>``.

        Returns either a commit (with diff) or a single file's contents at
        the given ref. Capped at ~1 MB; truncation is flagged.
        """
        if not is_safe_git_ref(ref):
            return _invalid_arg_error("ref", ref)
        if path is not None and not is_safe_relpath(path):
            return _invalid_arg_error("path", path)
        cmd = ["git", "show", "--no-color", ref, "--"]
        if path:
            cmd.append(path)
        attachment, result, err = await _run_git(
            repo, cmd, max_output_bytes=_SHOW_BYTES
        )
        if err is not None:
            return err
        if result.exit_code != 0:
            stderr = _err_text(result)
            return {
                "error": stderr.strip() or "git show failed",
                "kind": _classify_git_error(stderr),
                "ref": ref,
                "repo": attachment.full_name,
            }
        body = _decode(result.stdout)
        return {
            "repo": attachment.full_name,
            "ref": ref,
            "path": path,
            "content": body,
            "truncated": len(result.stdout) >= _SHOW_BYTES,
        }

    async def sandbox_git_blame(
        path: str,
        line_start: int | None = None,
        line_end: int | None = None,
        repo: str | None = None,
    ) -> dict[str, Any]:
        """``git blame --line-porcelain`` for ``path``; returns per-line records.

        Optional ``line_start``/``line_end`` narrow the range (inclusive).
        Output is parsed into ``[{commit, author, line, text}]``.
        """
        if not is_safe_relpath(path):
            return _invalid_arg_error("path", path)
        cmd = ["git", "blame", "--line-porcelain"]
        if line_start is not None:
            end = line_end if line_end is not None else line_start
            cmd.extend(["-L", f"{int(line_start)},{int(end)}"])
        cmd.extend(["--", path])
        attachment, result, err = await _run_git(
            repo, cmd, max_output_bytes=_BLAME_BYTES
        )
        if err is not None:
            return err
        if result.exit_code != 0:
            stderr = _err_text(result)
            return {
                "error": stderr.strip() or "git blame failed",
                "kind": _classify_git_error(stderr),
                "repo": attachment.full_name,
                "path": path,
            }
        records: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        current_line: int | None = None
        for line in _decode(result.stdout).splitlines():
            if not line:
                continue
            if line.startswith("\t"):
                if current is not None:
                    records.append(
                        {
                            "commit": current.get("commit"),
                            "author": current.get("author"),
                            "line": current_line,
                            "text": line[1:],
                        }
                    )
                    current = None
                    current_line = None
                continue
            parts = line.split(" ", 1)
            head = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            if current is None:
                sub = line.split(" ")
                current = {"commit": sub[0]}
                if len(sub) >= 3:
                    try:
                        current_line = int(sub[2])
                    except ValueError:
                        current_line = None
                continue
            if head == "author":
                current["author"] = rest
        return {
            "repo": attachment.full_name,
            "path": path,
            "lines": records,
            "count": len(records),
            "truncated": len(result.stdout) >= _BLAME_BYTES,
        }

    async def sandbox_git_diff(
        base: str,
        head: str = "HEAD",
        paths: list[str] | None = None,
        repo: str | None = None,
    ) -> dict[str, Any]:
        """``git diff <base>..<head>`` (optionally narrowed by paths).

        Capped at ~4 MB. ``truncated: true`` means the agent should narrow
        the path list or use ``sandbox_git_log`` to walk commits one at a
        time.
        """
        if not is_safe_git_ref(base):
            return _invalid_arg_error("base", base)
        if not is_safe_git_ref(head):
            return _invalid_arg_error("head", head)
        if paths:
            for p in paths:
                if not is_safe_relpath(p):
                    return _invalid_arg_error("paths", p)
        cmd = ["git", "diff", "--no-color", f"{base}..{head}", "--"]
        if paths:
            cmd.extend(paths)
        attachment, result, err = await _run_git(
            repo, cmd, max_output_bytes=_DIFF_BYTES
        )
        if err is not None:
            return err
        if result.exit_code != 0:
            stderr = _err_text(result)
            return {
                "error": stderr.strip() or "git diff failed",
                "kind": _classify_git_error(stderr),
                "base": base,
                "head": head,
                "repo": attachment.full_name,
            }
        return {
            "repo": attachment.full_name,
            "base": base,
            "head": head,
            "diff": _decode(result.stdout),
            "truncated": len(result.stdout) >= _DIFF_BYTES,
        }

    return [
        Tool(
            sandbox_checkout,
            name="sandbox_checkout",
            description=(
                "Detach HEAD onto ``ref`` (branch/tag/SHA) on a repo's "
                "worktree, fetching from origin first. Pass repo='owner/name' "
                "when multiple repos are attached. Returns {repo, ref, "
                "head_sha} or {error, kind: 'unknown_ref'|'conflict'|"
                "'network'|'auth'|'git_error'}. Pass force=True only when "
                "you need to overwrite local state."
            ),
        ),
        Tool(
            sandbox_git_log,
            name="sandbox_git_log",
            description=(
                "Walk a repo's commit history. Optional path narrows to a "
                "file/dir; since takes any git-parseable date "
                "('2 weeks ago', '2026-01-01'); limit capped at 500. "
                "Returns parsed [{commit, author, email, iso_date, subject}]."
            ),
        ),
        Tool(
            sandbox_git_show,
            name="sandbox_git_show",
            description=(
                "Show a commit (with diff) or a single file's content at a "
                "ref. Pass path to narrow to one file. Capped at ~1MB; "
                "truncation flagged."
            ),
        ),
        Tool(
            sandbox_git_blame,
            name="sandbox_git_blame",
            description=(
                "Per-line blame for a file. Optional line_start/line_end "
                "narrows the range (inclusive). Returns [{commit, author, "
                "line, text}]."
            ),
        ),
        Tool(
            sandbox_git_diff,
            name="sandbox_git_diff",
            description=(
                "Unified diff between two refs (base..head). Optional paths "
                "narrow the diff. Capped at ~4MB; if truncated, narrow paths "
                "or walk commits with sandbox_git_log."
            ),
        ),
    ]

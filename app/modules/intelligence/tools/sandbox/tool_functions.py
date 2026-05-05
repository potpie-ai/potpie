"""Async dispatchers for the four consolidated ``sandbox_*`` agent tools.

Shape: each top-level function takes ``project_id`` plus a ``command`` arg
(except ``sandbox_shell`` and ``sandbox_search`` which are single-purpose),
resolves the workspace, and returns ``{success: bool, ...}``.

Workspace resolution flow:

    project_id → ProjectService → repo_name + branch + user_id  (DB)
    user_id    ← contextvar (set by execution flow)
    auth       ← contextvar token, then GH_TOKEN/GITHUB_TOKEN fallback
        ↓
    SandboxClient.get_workspace(... mode=EDIT)  →  WorkspaceHandle
        ↓
    SandboxClient.<method>(handle, ...)
"""

from __future__ import annotations

import shlex
from typing import Any, Dict, List, Optional

from sandbox import CommandKind, ExecResult, WorkspaceMode

from app.modules.intelligence.tools.sandbox.client import (
    context_user_id_required,
    get_sandbox_client,
    lookup_project_summary,
    resolve_workspace,
)
from app.modules.intelligence.tools.sandbox.context import get_user_id
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


# ----------------------------------------------------------------------
# Project lookup + workspace resolution
# ----------------------------------------------------------------------
def _project_details(project_id: str) -> Dict[str, str]:
    """Resolve repo_name / branch / user_id for a project, with auth check.

    Reads through the shared :func:`lookup_project_summary` cache so the
    setup-banner emitter and the workspace resolver hit Postgres at most once
    per project per process.
    """
    summary = lookup_project_summary(project_id)
    user_id = get_user_id()
    if user_id and summary.get("user_id") and summary["user_id"] != user_id:
        raise ValueError(
            f"Project {project_id} does not belong to the current user."
        )
    # Return the legacy field names callers expect.
    return {
        "project_name": summary["repo_name"],
        "branch_name": summary["base_branch"],
        "repo_path": summary["repo_url"],
        "user_id": summary["user_id"],
    }


async def _resolve(project_id: str, *, mode: WorkspaceMode = WorkspaceMode.EDIT) -> Any:
    user_id = context_user_id_required()
    details = _project_details(project_id)
    repo_name = details["project_name"]
    branch = details.get("branch_name") or "main"
    repo_url = details.get("repo_path") or None
    return await resolve_workspace(
        user_id=user_id,
        project_id=project_id,
        repo_name=repo_name,
        repo_url=repo_url,
        branch=branch,
        base_ref=branch,
        create_branch=mode is not WorkspaceMode.ANALYSIS,
        mode=mode,
    )


def _err(msg: str, **extra: Any) -> Dict[str, Any]:
    logger.warning(msg)
    payload: Dict[str, Any] = {"success": False, "error": msg}
    payload.update(extra)
    return payload


def _wrap_exec_result(handle: Any, result: ExecResult) -> Dict[str, Any]:
    # Daytona's process.exec collapses stdout+stderr into one stream and
    # leaves ExecResult.stderr empty. Surface a non-empty stderr by
    # falling back to stdout on failed runs so the LLM sees the actual
    # diagnostic instead of an empty string. Successful runs keep stderr
    # empty so consumers can still distinguish "command warned to
    # stderr" on backends that split the streams (local, Docker).
    stdout_text = result.stdout.decode("utf-8", errors="replace")
    stderr_text = result.stderr.decode("utf-8", errors="replace")
    if not stderr_text and result.exit_code != 0:
        stderr_text = stdout_text
    return {
        "success": result.exit_code == 0,
        "branch": handle.branch,
        "exit_code": result.exit_code,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "timed_out": result.timed_out,
        "truncated": result.truncated,
    }


# ----------------------------------------------------------------------
# sandbox_text_editor
# ----------------------------------------------------------------------
async def _exec_text_editor(
    client: Any,
    handle: Any,
    *,
    command: str,
    path: str,
    view_range: Optional[List[int]] = None,
    file_text: Optional[str] = None,
    old_str: Optional[str] = None,
    new_str: Optional[str] = None,
    insert_line: Optional[int] = None,
) -> Dict[str, Any]:
    """Handle-bound text-editor implementation.

    Used by both the legacy ``project_id`` tool wrapper and the explicit
    ``create_sandbox_tools(client, handle)`` factory. Errors are wrapped
    by the public-facing tool function — this helper raises through.
    """
    if command == "view":
        return await _view(client, handle, path, view_range)
    if command == "create":
        if file_text is None:
            return _err("create requires file_text")
        return await _create(client, handle, path, file_text)
    if command == "str_replace":
        if old_str is None or new_str is None:
            return _err("str_replace requires old_str and new_str")
        return await _str_replace(client, handle, path, old_str, new_str)
    if command == "insert":
        if insert_line is None or new_str is None:
            return _err("insert requires insert_line and new_str")
        return await _insert(client, handle, path, insert_line, new_str)
    return _err(
        f"unknown command: {command!r}. "
        "Expected: view, create, str_replace, insert."
    )


async def sandbox_text_editor_tool(
    project_id: str,
    command: str,
    path: str,
    view_range: Optional[List[int]] = None,
    file_text: Optional[str] = None,
    old_str: Optional[str] = None,
    new_str: Optional[str] = None,
    insert_line: Optional[int] = None,
) -> Dict[str, Any]:
    """Dispatch view / create / str_replace / insert (legacy form)."""
    try:
        handle = await _resolve(project_id)
        client = get_sandbox_client()
        return await _exec_text_editor(
            client,
            handle,
            command=command,
            path=path,
            view_range=view_range,
            file_text=file_text,
            old_str=old_str,
            new_str=new_str,
            insert_line=insert_line,
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"sandbox_text_editor failed: {exc}")


async def _view(
    client: Any, handle: Any, path: str, view_range: Optional[List[int]]
) -> Dict[str, Any]:
    """View a file (with optional line slice) or list a directory."""
    is_dir = await _is_dir(client, handle, path)
    if is_dir:
        entries = await client.list_dir(handle, path)
        return {
            "success": True,
            "kind": "directory",
            "path": path,
            "branch": handle.branch,
            "entries": [
                {"name": e.name, "is_dir": e.is_dir, "size": e.size}
                for e in entries
            ],
        }
    data = await client.read_file(handle, path)
    text = data.decode("utf-8", errors="replace")
    if view_range:
        if len(view_range) != 2:
            return _err("view_range must be [start_line, end_line] (1-indexed)")
        start, end = view_range
        if start < 1 or end < start:
            return _err(
                f"view_range invalid: start={start}, end={end} (1-indexed, end>=start)"
            )
        # splitlines(True) keeps the trailing newline of each line so we
        # can reconstruct the slice exactly.
        lines = text.splitlines(keepends=True)
        sliced = "".join(lines[start - 1 : end])
        return {
            "success": True,
            "kind": "file",
            "path": path,
            "branch": handle.branch,
            "content": sliced,
            "view_range": [start, end],
            "total_lines": len(lines),
            "bytes": len(data),
        }
    return {
        "success": True,
        "kind": "file",
        "path": path,
        "branch": handle.branch,
        "content": text,
        "total_lines": text.count("\n") + (0 if text.endswith("\n") or not text else 1),
        "bytes": len(data),
    }


async def _is_dir(client: Any, handle: Any, path: str) -> bool:
    """Probe whether ``path`` is a directory inside the worktree."""
    if handle.local_path is not None:
        from pathlib import Path as _P

        return (_P(handle.local_path) / (path if path != "." else ".")).is_dir()
    # exec backend: use stat
    probe = await client.exec(
        handle,
        ["sh", "-c", f"if [ -d {shlex.quote(path)} ]; then echo dir; fi"],
        command_kind=CommandKind.READ,
    )
    return probe.exit_code == 0 and probe.stdout.strip() == b"dir"


async def _create(
    client: Any, handle: Any, path: str, file_text: str
) -> Dict[str, Any]:
    """Write a NEW file. Fails if the path already exists."""
    # Probe for existence — local fast path or exec.
    exists = False
    if handle.local_path is not None:
        from pathlib import Path as _P

        exists = (_P(handle.local_path) / path).exists()
    else:
        probe = await client.exec(
            handle,
            ["sh", "-c", f"[ -e {shlex.quote(path)} ] && echo y"],
            command_kind=CommandKind.READ,
        )
        exists = probe.exit_code == 0 and probe.stdout.strip() == b"y"
    if exists:
        return _err(
            f"create: {path!r} already exists. Use str_replace for edits, "
            "or sandbox_shell to remove the file first."
        )
    await client.write_file(handle, path, file_text)
    return {
        "success": True,
        "command": "create",
        "path": path,
        "branch": handle.branch,
        "bytes": len(file_text.encode("utf-8")),
        "lines": file_text.count("\n") + (0 if file_text.endswith("\n") or not file_text else 1),
    }


async def _str_replace(
    client: Any, handle: Any, path: str, old_str: str, new_str: str
) -> Dict[str, Any]:
    """Replace ``old_str`` with ``new_str`` once. Fails on 0 or >1 matches."""
    current_bytes = await client.read_file(handle, path)
    current = current_bytes.decode("utf-8", errors="replace")
    occurrences = current.count(old_str)
    if occurrences == 0:
        return _err(
            f"old_str not found in {path}. Re-view the file and include enough "
            "surrounding context to make the match unique."
        )
    if occurrences > 1:
        return _err(
            f"old_str matches {occurrences} times in {path}. Include more "
            "surrounding context so the match is unique.",
            occurrences=occurrences,
        )
    updated = current.replace(old_str, new_str, 1)
    await client.write_file(handle, path, updated)
    return {
        "success": True,
        "command": "str_replace",
        "path": path,
        "branch": handle.branch,
        "lines_added": len(new_str.splitlines()),
        "lines_removed": len(old_str.splitlines()),
    }


async def _insert(
    client: Any, handle: Any, path: str, insert_line: int, new_str: str
) -> Dict[str, Any]:
    """Insert ``new_str`` after line ``insert_line`` (1-indexed; 0 = top)."""
    if insert_line < 0:
        return _err(f"insert_line must be >= 0; got {insert_line}")
    current_bytes = await client.read_file(handle, path)
    current = current_bytes.decode("utf-8", errors="replace")
    lines = current.splitlines(keepends=True)
    if insert_line > len(lines):
        return _err(
            f"insert_line={insert_line} is past EOF (file has {len(lines)} lines)."
        )
    # Ensure the inserted block ends with a newline so subsequent line
    # numbering stays sane.
    block = new_str if new_str.endswith("\n") else new_str + "\n"
    new_text = "".join(lines[:insert_line]) + block + "".join(lines[insert_line:])
    await client.write_file(handle, path, new_text)
    return {
        "success": True,
        "command": "insert",
        "path": path,
        "branch": handle.branch,
        "insert_line": insert_line,
        "lines_added": block.count("\n"),
    }


# ----------------------------------------------------------------------
# sandbox_shell
# ----------------------------------------------------------------------
async def _exec_shell(
    client: Any,
    handle: Any,
    *,
    command: str,
    timeout_s: Optional[int] = 120,
    max_output_bytes: Optional[int] = 80_000,
) -> Dict[str, Any]:
    """Handle-bound shell-exec implementation."""
    result = await client.exec(
        handle,
        ["sh", "-c", command],
        command_kind=CommandKind.WRITE,
        timeout_s=timeout_s,
        max_output_bytes=max_output_bytes,
        shell=False,  # already wrapped in sh -c above
    )
    return _wrap_exec_result(handle, result)


async def sandbox_shell_tool(
    project_id: str,
    command: str,
    timeout_s: Optional[int] = 120,
    max_output_bytes: Optional[int] = 80_000,
) -> Dict[str, Any]:
    """Run a single shell command inside the sandbox (legacy form)."""
    try:
        handle = await _resolve(project_id)
        client = get_sandbox_client()
        return await _exec_shell(
            client,
            handle,
            command=command,
            timeout_s=timeout_s,
            max_output_bytes=max_output_bytes,
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"sandbox_shell failed: {exc}")


# ----------------------------------------------------------------------
# sandbox_search
# ----------------------------------------------------------------------
async def _exec_search(
    client: Any,
    handle: Any,
    *,
    pattern: str,
    glob: Optional[str] = None,
    case: bool = False,
    path: Optional[str] = None,
    max_hits: Optional[int] = 200,
) -> Dict[str, Any]:
    """Handle-bound ripgrep implementation."""
    hits = await client.search(
        handle,
        pattern,
        glob=glob,
        case=case,
        path=path,
        max_hits=max_hits,
    )
    return {
        "success": True,
        "branch": handle.branch,
        "pattern": pattern,
        "hits": [
            {"path": h.path, "line": h.line, "snippet": h.snippet} for h in hits
        ],
        "truncated": max_hits is not None and len(hits) >= max_hits,
    }


async def sandbox_search_tool(
    project_id: str,
    pattern: str,
    glob: Optional[str] = None,
    case: bool = False,
    path: Optional[str] = None,
    max_hits: Optional[int] = 200,
) -> Dict[str, Any]:
    """Ripgrep across the worktree (legacy form)."""
    try:
        handle = await _resolve(project_id, mode=WorkspaceMode.ANALYSIS)
        client = get_sandbox_client()
        return await _exec_search(
            client,
            handle,
            pattern=pattern,
            glob=glob,
            case=case,
            path=path,
            max_hits=max_hits,
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"sandbox_search failed: {exc}")


# ----------------------------------------------------------------------
# sandbox_git
# ----------------------------------------------------------------------
async def _exec_git(
    client: Any,
    handle: Any,
    *,
    command: str,
    base_ref: Optional[str] = None,
    paths: Optional[List[str]] = None,
    limit: Optional[int] = 20,
    message: Optional[str] = None,
    set_upstream: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """Handle-bound git implementation."""
    if command == "status":
        status = await client.status(handle)
        return {
            "success": True,
            "command": "status",
            "branch": status.branch,
            "is_clean": status.is_clean,
            "staged": list(status.staged),
            "unstaged": list(status.unstaged),
            "untracked": list(status.untracked),
        }
    if command == "diff":
        diff = await client.diff(handle, base_ref=base_ref, paths=paths)
        return {
            "success": True,
            "command": "diff",
            "branch": handle.branch,
            "base_ref": base_ref,
            "diff": diff,
        }
    if command == "log":
        return await _git_log(client, handle, limit or 20)
    if command == "commit":
        if not message:
            return _err("commit requires a message")
        sha = await client.commit(handle, message, paths=paths)
        return {
            "success": True,
            "command": "commit",
            "branch": handle.branch,
            "commit": sha,
        }
    if command == "push":
        await client.push(handle, set_upstream=set_upstream, force=force)
        return {
            "success": True,
            "command": "push",
            "branch": handle.branch,
        }
    return _err(
        f"unknown command: {command!r}. "
        "Expected: status, diff, log, commit, push."
    )


async def sandbox_git_tool(
    project_id: str,
    command: str,
    base_ref: Optional[str] = None,
    paths: Optional[List[str]] = None,
    limit: Optional[int] = 20,
    message: Optional[str] = None,
    set_upstream: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """Dispatch status / diff / log / commit / push (legacy form)."""
    try:
        handle = await _resolve(project_id)
        client = get_sandbox_client()
        return await _exec_git(
            client,
            handle,
            command=command,
            base_ref=base_ref,
            paths=paths,
            limit=limit,
            message=message,
            set_upstream=set_upstream,
            force=force,
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"sandbox_git failed: {exc}")


# ----------------------------------------------------------------------
# sandbox_pr (handle-bound only — there's no legacy contextvar form)
# ----------------------------------------------------------------------
async def _exec_pull_request(
    client: Any,
    handle: Any,
    *,
    title: str,
    body: str,
    base_branch: str,
    head_branch: Optional[str] = None,
    reviewers: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    repo_name: Optional[str] = None,
    repo_url: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Handle-bound PR-creation tool.

    The repo identity normally lives on the bound workspace's request,
    but ``WorkspaceHandle`` doesn't carry the full request — callers can
    pass ``repo_name``/``repo_url`` explicitly when constructing the
    factory if the harness has them. This helper delegates to
    ``SandboxClient.create_pull_request`` which enforces capability.
    """
    if not repo_name:
        return _err(
            "sandbox_pr: repo_name not bound on the toolset; pass it via "
            "create_sandbox_tools(... pr_repo_name=...)."
        )
    try:
        pr = await client.create_pull_request(
            handle,
            repo=repo_name,
            title=title,
            body=body,
            base_branch=base_branch,
            head_branch=head_branch,
            repo_url=repo_url,
            reviewers=reviewers,
            labels=labels,
            auth_token=auth_token,
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"sandbox_pr failed: {exc}")
    return {
        "success": True,
        "pr_id": pr.id,
        "url": pr.url,
        "title": pr.title,
        "head_branch": pr.head_branch,
        "base_branch": pr.base_branch,
    }


async def sandbox_pr_comment_tool(
    project_id: str,
    pr_number: int,
    body: str,
    path: Optional[str] = None,
    line: Optional[int] = None,
    commit_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Post a comment on an existing PR through the sandbox path.

    Supports both top-level (no ``path``/``line``) and inline
    (``path`` + ``line``) comments. Unlike ``sandbox_pr``, this does
    NOT require a writable workspace — review-flow agents can comment
    without a worktree. Attribution flows through the same
    ``GitPlatformProvider`` factory chain ``sandbox_pr`` uses, so the
    PR comment lands under the same identity (the Potpie bot when the
    GitHub App is installed).
    """
    try:
        details = _project_details(project_id)
        client = get_sandbox_client()
        result = await client.comment_on_pull_request(
            repo=details["project_name"],
            pr_number=pr_number,
            body=body,
            path=path,
            line=line,
            commit_id=commit_id,
            repo_url=details.get("repo_path") or None,
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"sandbox_pr_comment failed: {exc}")
    return {
        "success": True,
        "comment_id": result.id,
        "url": result.url,
    }


async def sandbox_pr_tool(
    project_id: str,
    title: str,
    body: str,
    base_branch: Optional[str] = None,
    head_branch: Optional[str] = None,
    reviewers: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Open a PR from the agent's worktree branch into ``base_branch``.

    Legacy contextvar form: derives ``repo_name`` from
    ``project_id`` via the cached project summary, defaults
    ``base_branch`` to the project's stored base branch when omitted,
    and dispatches through :class:`SandboxClient.create_pull_request`.
    The platform-side step needs ``head_branch`` already pushed —
    callers should run ``sandbox_git push`` first.
    """
    try:
        details = _project_details(project_id)
        summary = lookup_project_summary(project_id)
        handle = await _resolve(project_id)
        client = get_sandbox_client()
        return await _exec_pull_request(
            client,
            handle,
            title=title,
            body=body,
            base_branch=base_branch or summary["base_branch"],
            head_branch=head_branch,
            reviewers=reviewers,
            labels=labels,
            repo_name=details["project_name"],
            repo_url=details.get("repo_path") or None,
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"sandbox_pr failed: {exc}")


async def _git_log(client: Any, handle: Any, limit: int) -> Dict[str, Any]:
    """`git log` with a stable pretty format the LLM can parse."""
    fmt = "%H%x09%an%x09%ae%x09%aI%x09%s"
    result = await client.exec(
        handle,
        [
            "git",
            "log",
            f"--pretty=format:{fmt}",
            f"-{max(1, limit)}",
        ],
        command_kind=CommandKind.READ,
    )
    if result.exit_code != 0:
        return _err(
            f"git log failed: {result.stderr.decode('utf-8', errors='replace')}"
        )
    commits = []
    for line in result.stdout.decode("utf-8", errors="replace").splitlines():
        parts = line.split("\t", 4)
        if len(parts) == 5:
            sha, name, email, date, subject = parts
            commits.append(
                {
                    "sha": sha,
                    "author_name": name,
                    "author_email": email,
                    "date": date,
                    "subject": subject,
                }
            )
    return {
        "success": True,
        "command": "log",
        "branch": handle.branch,
        "commits": commits,
    }

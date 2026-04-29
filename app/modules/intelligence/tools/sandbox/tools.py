"""SimpleTool factories for the consolidated sandbox tool group.

Four tools, each with a tight focused surface:

  * ``sandbox_text_editor`` — Anthropic-style file ops (view/create/str_replace/insert)
  * ``sandbox_shell`` — single shell command
  * ``sandbox_search`` — ripgrep
  * ``sandbox_git`` — status/diff/log/commit/push

Two construction modes:

* **Legacy** — ``create_sandbox_tools()`` (no args). Tools resolve the
  workspace at call time via the contextvar set by the agent harness
  plus a DB lookup of the project's repo / branch. Required for
  back-compat with existing harness wiring.

* **Explicit** — ``create_sandbox_tools(client=..., handle=...)``. Tools
  are pre-bound to a specific :class:`WorkspaceHandle`; the schema
  drops ``project_id`` so the LLM cannot target a different project,
  and capability gating drops write tools when the handle is read-only.

The ``SimpleTool`` wrapper is local to this module so the sandbox group
has zero coupling to the legacy code_changes_manager package.
"""

from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, List, TYPE_CHECKING

if TYPE_CHECKING:
    from sandbox import SandboxClient, WorkspaceHandle


class SimpleTool:
    """Minimal LangChain-compatible tool wrapper.

    The agent runtime treats anything with ``name`` / ``description`` /
    ``func`` / ``args_schema`` like a ``StructuredTool`` so this lightweight
    holder is enough.
    """

    __slots__ = ("name", "description", "func", "args_schema")

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        args_schema: Any,
    ) -> None:
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema


from .tool_functions import (
    _exec_git,
    _exec_pull_request,
    _exec_search,
    _exec_shell,
    _exec_text_editor,
    sandbox_git_tool,
    sandbox_search_tool,
    sandbox_shell_tool,
    sandbox_text_editor_tool,
)
from .tool_inputs import (
    SandboxGitInput,
    SandboxGitInputBound,
    SandboxPullRequestInput,
    SandboxSearchInput,
    SandboxSearchInputBound,
    SandboxShellInput,
    SandboxShellInputBound,
    SandboxTextEditorInput,
    SandboxTextEditorInputBound,
)


# ----------------------------------------------------------------------
# Tool descriptions (shared between legacy and explicit forms)
# ----------------------------------------------------------------------
_TEXT_EDITOR_DESC = (
    "Read and edit files inside the project's sandboxed worktree. "
    "Modeled on Anthropic's text_editor tool — same command names "
    "and semantics. Pass a 'command' arg:\n"
    "- 'view': read a file (optionally with view_range=[start,end] "
    "to slice large files), or list a directory (when path is a dir).\n"
    "- 'create': write a NEW file from file_text. Fails if the path "
    "already exists; use str_replace for edits.\n"
    "- 'str_replace': replace old_str with new_str. old_str must "
    "occur EXACTLY ONCE — include 3-5 surrounding lines for uniqueness. "
    "Re-view after to confirm.\n"
    "- 'insert': insert new_str AFTER insert_line (1-indexed; 0 means "
    "before the first line). Use for adding imports, a new function, etc.\n"
    "Edits are durable from the moment this returns; sandbox_git "
    "command='commit' formalises them."
)

_SHELL_DESC = (
    "Run a single shell command inside the sandbox at the worktree "
    "root. Single string command (executed via /bin/sh -c) so pipes "
    "and redirects work. Returns "
    "{exit_code, stdout, stderr, timed_out, truncated}. "
    "Output is capped at ~80 KB by default — bump max_output_bytes "
    "for noisy builds. Use this for tests (pytest / vitest / cargo "
    "test), linters, type checks, file deletes / moves, anything "
    "the editor tool doesn't cover."
)

_SEARCH_DESC = (
    "Ripgrep across the worktree. Returns {path, line, snippet} hits. "
    "Smart-case by default; pass case=true to force case-sensitive. "
    "Use the glob filter to scope by path pattern (e.g. '**/*.py'). "
    "Hits are capped (default 200) — call again with a tighter "
    "pattern or glob if truncated."
)

_GIT_DESC = (
    "Git operations on the agent's worktree branch. Pass a 'command':\n"
    "- 'status': what's changed (staged / unstaged / untracked).\n"
    "- 'diff': print the diff. With base_ref='main' diffs branch vs "
    "main; without, working tree vs HEAD. paths= scopes to specific files.\n"
    "- 'log': recent commits on the branch (default last 20).\n"
    "- 'commit': stage and commit. Without paths, stages everything; "
    "with paths, only those files. Returns the commit SHA. Fails if "
    "there's nothing to commit.\n"
    "- 'push': publish the branch to origin (--set-upstream by default).\n"
    "Use sandbox_pr to open a PR after pushing."
)

_PR_DESC = (
    "Open a pull request from the agent's worktree branch into "
    "base_branch. Push the branch with sandbox_git push first; this "
    "tool only does the platform-side step. Returns "
    "{success, pr_id, url}. Available only when the workspace is "
    "writable and the harness wired a GitPlatformProvider."
)


def create_sandbox_tools(
    *,
    client: "SandboxClient | None" = None,
    handle: "WorkspaceHandle | None" = None,
    enforce_capabilities: bool = True,
    pr_repo_name: str | None = None,
    pr_repo_url: str | None = None,
    pr_auth_token: str | None = None,
) -> List[SimpleTool]:
    """The four ``sandbox_*`` tools.

    Two modes:

    * **Legacy** (``create_sandbox_tools()``) — tools resolve the
      workspace at call time via contextvars + DB. Required by the
      existing agent harness.
    * **Explicit** (``create_sandbox_tools(client=..., handle=...)``) —
      tools are pre-bound to ``handle``. The schema omits ``project_id``;
      ``enforce_capabilities=True`` (default) drops write tools when
      the handle is read-only.

    Pass ``enforce_capabilities=False`` to expose every tool regardless
    of mode (the underlying runtime still refuses writes on read-only
    workspaces — gating is a UX guardrail, not a security boundary).
    """
    if client is None and handle is None:
        return _legacy_tools()
    if client is None or handle is None:
        raise ValueError(
            "create_sandbox_tools: pass both `client` and `handle`, or neither."
        )
    return _explicit_tools(
        client,
        handle,
        enforce_capabilities=enforce_capabilities,
        pr_repo_name=pr_repo_name,
        pr_repo_url=pr_repo_url,
        pr_auth_token=pr_auth_token,
    )


def _legacy_tools() -> List[SimpleTool]:
    return [
        SimpleTool(
            name="sandbox_text_editor",
            description=_TEXT_EDITOR_DESC,
            func=sandbox_text_editor_tool,
            args_schema=SandboxTextEditorInput,
        ),
        SimpleTool(
            name="sandbox_shell",
            description=_SHELL_DESC,
            func=sandbox_shell_tool,
            args_schema=SandboxShellInput,
        ),
        SimpleTool(
            name="sandbox_search",
            description=_SEARCH_DESC,
            func=sandbox_search_tool,
            args_schema=SandboxSearchInput,
        ),
        SimpleTool(
            name="sandbox_git",
            description=_GIT_DESC,
            func=sandbox_git_tool,
            args_schema=SandboxGitInput,
        ),
    ]


def _explicit_tools(
    client: "SandboxClient",
    handle: "WorkspaceHandle",
    *,
    enforce_capabilities: bool,
    pr_repo_name: str | None,
    pr_repo_url: str | None,
    pr_auth_token: str | None,
) -> List[SimpleTool]:
    """Build a toolset pre-bound to ``handle``.

    ``search`` is always included (read-only). ``text_editor``, ``shell``,
    ``git``, and ``sandbox_pr`` are write-capable; with
    ``enforce_capabilities=True`` they're omitted when
    ``handle.capabilities.writable`` is false.

    ``sandbox_pr`` additionally requires ``pr_repo_name`` to be passed in
    (the workspace handle doesn't carry the repo identity); without it
    the tool is omitted even on writable handles.
    """
    writable = handle.capabilities.writable or not enforce_capabilities

    tools: List[SimpleTool] = [
        SimpleTool(
            name="sandbox_search",
            description=_SEARCH_DESC,
            func=_bind(_exec_search, client, handle),
            args_schema=SandboxSearchInputBound,
        ),
    ]
    if writable:
        tools.extend(
            [
                SimpleTool(
                    name="sandbox_text_editor",
                    description=_TEXT_EDITOR_DESC,
                    func=_bind(_exec_text_editor, client, handle),
                    args_schema=SandboxTextEditorInputBound,
                ),
                SimpleTool(
                    name="sandbox_shell",
                    description=_SHELL_DESC,
                    func=_bind(_exec_shell, client, handle),
                    args_schema=SandboxShellInputBound,
                ),
                SimpleTool(
                    name="sandbox_git",
                    description=_GIT_DESC,
                    func=_bind(_exec_git, client, handle),
                    args_schema=SandboxGitInputBound,
                ),
            ]
        )
        if pr_repo_name:
            tools.append(
                SimpleTool(
                    name="sandbox_pr",
                    description=_PR_DESC,
                    func=_bind_pr(
                        client,
                        handle,
                        repo_name=pr_repo_name,
                        repo_url=pr_repo_url,
                        auth_token=pr_auth_token,
                    ),
                    args_schema=SandboxPullRequestInput,
                )
            )
    return tools


def _bind(
    fn: Callable[..., Awaitable[Any]],
    client: Any,
    handle: Any,
) -> Callable[..., Awaitable[Any]]:
    """Pre-bind ``client`` and ``handle`` so the tool harness only passes
    the LLM-supplied kwargs. Each bound coroutine forwards through; we
    don't catch exceptions here — explicit-form callers can apply their
    own error envelope upstream."""

    @functools.wraps(fn)
    async def bound(**kwargs: Any) -> Any:
        return await fn(client, handle, **kwargs)

    return bound


def _bind_pr(
    client: Any,
    handle: Any,
    *,
    repo_name: str,
    repo_url: str | None,
    auth_token: str | None,
) -> Callable[..., Awaitable[Any]]:
    """Bind the PR tool with extra context the handle doesn't carry."""

    async def bound(**kwargs: Any) -> Any:
        return await _exec_pull_request(
            client,
            handle,
            repo_name=repo_name,
            repo_url=repo_url,
            auth_token=auth_token,
            **kwargs,
        )

    return bound


__all__ = ["SimpleTool", "create_sandbox_tools"]

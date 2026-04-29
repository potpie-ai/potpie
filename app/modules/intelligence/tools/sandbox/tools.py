"""SimpleTool factories for the consolidated sandbox tool group.

Four tools, each with a tight focused surface:

  * ``sandbox_text_editor`` — Anthropic-style file ops (view/create/str_replace/insert)
  * ``sandbox_shell`` — single shell command
  * ``sandbox_search`` — ripgrep
  * ``sandbox_git`` — status/diff/log/commit/push

The ``SimpleTool`` wrapper is local to this module so the sandbox group has
zero coupling to the legacy code_changes_manager package.
"""

from __future__ import annotations

from typing import Any, Callable, List


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
    sandbox_git_tool,
    sandbox_search_tool,
    sandbox_shell_tool,
    sandbox_text_editor_tool,
)
from .tool_inputs import (
    SandboxGitInput,
    SandboxSearchInput,
    SandboxShellInput,
    SandboxTextEditorInput,
)


def create_sandbox_tools() -> List[SimpleTool]:
    """The four ``sandbox_*`` tools registered with ToolService."""
    return [
        SimpleTool(
            name="sandbox_text_editor",
            description=(
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
            ),
            func=sandbox_text_editor_tool,
            args_schema=SandboxTextEditorInput,
        ),
        SimpleTool(
            name="sandbox_shell",
            description=(
                "Run a single shell command inside the sandbox at the worktree "
                "root. Single string command (executed via /bin/sh -c) so pipes "
                "and redirects work. Returns "
                "{exit_code, stdout, stderr, timed_out, truncated}. "
                "Output is capped at ~80 KB by default — bump max_output_bytes "
                "for noisy builds. Use this for tests (pytest / vitest / cargo "
                "test), linters, type checks, file deletes / moves, anything "
                "the editor tool doesn't cover."
            ),
            func=sandbox_shell_tool,
            args_schema=SandboxShellInput,
        ),
        SimpleTool(
            name="sandbox_search",
            description=(
                "Ripgrep across the worktree. Returns {path, line, snippet} hits. "
                "Smart-case by default; pass case=true to force case-sensitive. "
                "Use the glob filter to scope by path pattern (e.g. '**/*.py'). "
                "Hits are capped (default 200) — call again with a tighter "
                "pattern or glob if truncated."
            ),
            func=sandbox_search_tool,
            args_schema=SandboxSearchInput,
        ),
        SimpleTool(
            name="sandbox_git",
            description=(
                "Git operations on the agent's worktree branch. Pass a 'command':\n"
                "- 'status': what's changed (staged / unstaged / untracked).\n"
                "- 'diff': print the diff. With base_ref='main' diffs branch vs "
                "main; without, working tree vs HEAD. paths= scopes to specific files.\n"
                "- 'log': recent commits on the branch (default last 20).\n"
                "- 'commit': stage and commit. Without paths, stages everything; "
                "with paths, only those files. Returns the commit SHA. Fails if "
                "there's nothing to commit.\n"
                "- 'push': publish the branch to origin (--set-upstream by default).\n"
                "PR creation lives in code_provider_create_pr — push first, then "
                "ask the user before opening a PR."
            ),
            func=sandbox_git_tool,
            args_schema=SandboxGitInput,
        ),
    ]


__all__ = ["SimpleTool", "create_sandbox_tools"]

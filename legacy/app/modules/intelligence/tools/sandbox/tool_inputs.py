"""Pydantic input schemas for the consolidated sandbox tool surface.

Four tools, each dispatched via a ``command`` arg the LLM picks:

* ``sandbox_text_editor`` — view / create / str_replace / insert (Anthropic
  text-editor shape; LLMs already know it).
* ``sandbox_shell`` — run a single shell command.
* ``sandbox_search`` — ripgrep across the worktree.
* ``sandbox_git`` — status / diff / log / commit / push.

``project_id`` is always required; the workspace is resolved server-side from
``ProjectService`` plus the ``conversation_id`` contextvar so the LLM cannot
target a different project's worktree.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class _ProjectScope(BaseModel):
    project_id: str = Field(
        ...,
        description="The potpie project_id whose worktree the tool operates on. "
        "Pulled from the conversation context — always pass it through.",
    )


# ----------------------------------------------------------------------
# sandbox_text_editor
# ----------------------------------------------------------------------
TextEditorCommand = Literal["view", "create", "str_replace", "insert"]


class SandboxTextEditorInput(_ProjectScope):
    """File-level operations on the sandboxed worktree.

    Modeled on Anthropic's ``text_editor`` tool — same command names and
    argument semantics — so LLMs trained on that schema dispatch correctly.

    Commands:
      * ``view`` — read a file (with optional ``view_range``) or list a
        directory. ``view_range`` is ``[start, end]`` (1-indexed, inclusive);
        omit to read the whole file.
      * ``create`` — write a NEW file from ``file_text``. Fails if the path
        already exists — use ``str_replace`` (or ``view`` then re-write via
        ``create`` after delete) for existing files.
      * ``str_replace`` — replace ``old_str`` with ``new_str``. ``old_str``
        must occur EXACTLY ONCE in the file; include 3-5 lines of surrounding
        context to make the match unique.
      * ``insert`` — insert ``new_str`` after line ``insert_line`` (1-indexed;
        ``0`` means insert at the very top of the file).
    """

    command: TextEditorCommand = Field(
        ...,
        description="Operation to perform: 'view', 'create', 'str_replace', or 'insert'.",
    )
    path: str = Field(
        ...,
        description="Repo-relative path. For 'view' this can be a file or a directory.",
    )
    view_range: Optional[List[int]] = Field(
        default=None,
        description="For 'view': [start_line, end_line] (1-indexed, inclusive). "
        "Omit to read the whole file. Use a range for files >500 lines.",
    )
    file_text: Optional[str] = Field(
        default=None,
        description="For 'create': the full UTF-8 content of the new file.",
    )
    old_str: Optional[str] = Field(
        default=None,
        description="For 'str_replace': exact substring to find. Must match "
        "exactly once — pad with surrounding lines to uniqueify.",
    )
    new_str: Optional[str] = Field(
        default=None,
        description="For 'str_replace' or 'insert': the replacement / inserted text.",
    )
    insert_line: Optional[int] = Field(
        default=None,
        description="For 'insert': line number to insert AFTER (1-indexed). "
        "Use 0 to insert at the top of the file.",
    )


# ----------------------------------------------------------------------
# sandbox_shell
# ----------------------------------------------------------------------
class SandboxShellInput(_ProjectScope):
    """Run a single shell command inside the sandbox at the worktree root.

    Runs via ``/bin/sh -c`` so pipes, redirects, and command chains work.
    Output is capped (default ~80 KB) — bump ``max_output_bytes`` for noisy
    builds, lower for cheap probes.

    The sandbox image ships with ``ripgrep`` (``rg``), ``fd``, ``jq``,
    ``tree``, ``git``, ``gh``, ``python3``, and ``node`` preinstalled.
    For any code/text search prefer ``rg`` over ``grep -r`` / ``find +
    grep`` — it's an order of magnitude faster, respects ``.gitignore``,
    and gives column-accurate hits. Use ``sandbox_search`` for
    structured ripgrep results; reach for ``sandbox_shell`` + ``rg``
    only when you need pipes / extra flags (``-l``, ``-c``, ``--type``,
    ``-A``/``-B``, ``--files-with-matches | xargs ...``, etc.).
    """

    command: str = Field(
        ...,
        description="The shell command. e.g. 'pytest tests/ -v', "
        "'ruff check', 'ls -la', 'cat /etc/os-release | head'. "
        "For searching text/code, prefer ripgrep ('rg PATTERN', "
        "'rg -t py PATTERN', 'rg -l PATTERN | xargs ...') over grep / "
        "find — ripgrep is preinstalled and respects .gitignore. "
        "Use sandbox_search for plain searches; only drop down to "
        "'rg' here when you need shell pipes or flags it doesn't expose.",
    )
    timeout_s: Optional[int] = Field(
        default=120,
        description="Timeout in seconds. Default 120s. Raise for builds, "
        "lower for cheap probes.",
    )
    max_output_bytes: Optional[int] = Field(
        default=80_000,
        description="Cap on combined stdout to keep responses LLM-sized.",
    )


# ----------------------------------------------------------------------
# sandbox_search
# ----------------------------------------------------------------------
class SandboxSearchInput(_ProjectScope):
    """Ripgrep across the worktree.

    Returns ``{path, line, snippet}`` hits. Smart-case by default. Use the
    ``glob`` filter to scope (e.g. ``**/*.py``).
    """

    pattern: str = Field(..., description="Regex / fixed string passed to ripgrep.")
    glob: Optional[str] = Field(
        default=None,
        description="Optional glob filter, e.g. '**/*.py'. Omit to search everything.",
    )
    case: bool = Field(
        default=False,
        description="True for case-sensitive; False = smart-case (default).",
    )
    path: Optional[str] = Field(
        default=None,
        description="Optional sub-path to scope the search to.",
    )
    max_hits: Optional[int] = Field(
        default=200,
        description="Cap on returned hits to keep responses LLM-sized.",
    )


# ----------------------------------------------------------------------
# sandbox_git
# ----------------------------------------------------------------------
GitCommand = Literal["status", "diff", "log", "commit", "push"]


class SandboxGitInput(_ProjectScope):
    """Git operations on the agent worktree.

    Commands:
      * ``status`` — what's changed (staged / unstaged / untracked).
      * ``diff`` — print the diff. With ``base_ref``: branch vs ``base_ref``.
        Without: working tree vs HEAD. Optional ``paths`` to scope.
      * ``log`` — recent commits on the branch (default last 20).
      * ``commit`` — stage and commit. Without ``paths``, stages everything;
        with ``paths``, only those files. Returns the commit SHA.
      * ``push`` — publish the branch to origin. ``set_upstream`` defaults
        true. ``force`` uses ``--force-with-lease`` (use sparingly).
    """

    command: GitCommand = Field(
        ...,
        description="Git operation: 'status', 'diff', 'log', 'commit', or 'push'.",
    )
    # diff
    base_ref: Optional[str] = Field(
        default=None,
        description="For 'diff': base ref to diff against (e.g. 'main'). "
        "Omit for working-tree-vs-HEAD.",
    )
    paths: Optional[List[str]] = Field(
        default=None,
        description="For 'diff' / 'commit': paths to scope to. Omit for all.",
    )
    # log
    limit: Optional[int] = Field(
        default=20,
        description="For 'log': how many commits to return. Default 20.",
    )
    # commit
    message: Optional[str] = Field(
        default=None,
        description="For 'commit': required commit message.",
    )
    # push
    set_upstream: bool = Field(
        default=True,
        description="For 'push': pass --set-upstream so origin tracks this branch.",
    )
    force: bool = Field(
        default=False,
        description="For 'push': use --force-with-lease. Reserve for amend/rebase.",
    )


# ----------------------------------------------------------------------
# Handle-bound variants (P6: explicit toolset)
# ----------------------------------------------------------------------
# Same schemas as above but without ``project_id``. Used by the
# ``create_sandbox_tools(client=..., handle=...)`` form where the
# workspace is already pinned by the harness, so the LLM doesn't need
# to (and shouldn't be able to) target a different project.


class SandboxTextEditorInputBound(BaseModel):
    """File-level operations on a pre-bound workspace handle."""

    command: TextEditorCommand = Field(
        ...,
        description="Operation: 'view', 'create', 'str_replace', or 'insert'.",
    )
    path: str = Field(..., description="Repo-relative path.")
    view_range: Optional[List[int]] = Field(default=None)
    file_text: Optional[str] = Field(default=None)
    old_str: Optional[str] = Field(default=None)
    new_str: Optional[str] = Field(default=None)
    insert_line: Optional[int] = Field(default=None)


class SandboxShellInputBound(BaseModel):
    """Run a single shell command on a pre-bound workspace handle.

    Same surface as :class:`SandboxShellInput` — ``ripgrep``, ``fd``,
    ``jq``, ``git``, ``gh``, ``python3``, and ``node`` are preinstalled
    in the sandbox image. Prefer ``rg`` for any text/code search.
    """

    command: str = Field(
        ...,
        description="The shell command. For searching prefer ripgrep "
        "('rg PATTERN', 'rg -t py PATTERN', 'rg -l PATTERN | xargs ...') "
        "over grep / find — ripgrep is preinstalled and honors .gitignore.",
    )
    timeout_s: Optional[int] = Field(default=120)
    max_output_bytes: Optional[int] = Field(default=80_000)


class SandboxSearchInputBound(BaseModel):
    """Ripgrep across a pre-bound workspace handle."""

    pattern: str = Field(..., description="Regex / fixed string.")
    glob: Optional[str] = Field(default=None)
    case: bool = Field(default=False)
    path: Optional[str] = Field(default=None)
    max_hits: Optional[int] = Field(default=200)


class SandboxGitInputBound(BaseModel):
    """Git operations on a pre-bound workspace handle."""

    command: GitCommand = Field(
        ...,
        description="Git operation: 'status', 'diff', 'log', 'commit', or 'push'.",
    )
    base_ref: Optional[str] = Field(default=None)
    paths: Optional[List[str]] = Field(default=None)
    limit: Optional[int] = Field(default=20)
    message: Optional[str] = Field(default=None)
    set_upstream: bool = Field(default=True)
    force: bool = Field(default=False)


class SandboxPullRequestInput(BaseModel):
    """Open a PR via the configured `GitPlatformProvider`.

    The agent should call ``sandbox_git`` with ``command='push'`` first
    so ``head_branch`` exists on the remote — this tool is the
    platform-side step only. Returns ``{success, pr_number, url}`` so
    the caller can surface the link.
    """

    title: str = Field(..., description="PR title (single line).")
    body: str = Field(..., description="PR body (markdown).")
    base_branch: str = Field(
        ...,
        description="Branch to merge into, e.g. 'main' or 'develop'.",
    )
    head_branch: Optional[str] = Field(
        default=None,
        description="Source branch. Defaults to the workspace's bound branch.",
    )
    reviewers: Optional[List[str]] = Field(
        default=None,
        description="GitHub usernames to request review from.",
    )
    labels: Optional[List[str]] = Field(
        default=None, description="Labels to add to the PR."
    )


class SandboxPullRequestInputLegacy(BaseModel):
    """Legacy PR input (contextvar form): takes ``project_id`` so the
    tool can derive ``repo_name`` from the DB, and treats ``base_branch``
    as optional (the project's stored base branch is the default).
    """

    project_id: str = Field(..., description="Potpie project id.")
    title: str = Field(..., description="PR title (single line).")
    body: str = Field(..., description="PR body (markdown).")
    base_branch: Optional[str] = Field(
        default=None,
        description=(
            "Branch to merge into. Defaults to the project's stored "
            "base branch (typically 'main')."
        ),
    )
    head_branch: Optional[str] = Field(
        default=None,
        description="Source branch. Defaults to the workspace's bound branch.",
    )
    reviewers: Optional[List[str]] = Field(
        default=None,
        description="GitHub usernames to request review from.",
    )
    labels: Optional[List[str]] = Field(
        default=None, description="Labels to add to the PR."
    )


class SandboxPRCommentInputLegacy(BaseModel):
    """Post a comment on an existing PR (legacy contextvar form).

    Two shapes:

    * **Top-level** — only ``body`` and ``pr_number``. Posts a
      conversation comment on the PR.
    * **Inline** — set ``path`` AND ``line``. Posts a code-review
      comment at that file/line.

    Mixing the two in one call is rejected — the sandbox tool wraps a
    single platform call. The agent can issue this multiple times for
    multiple comments. ``project_id`` is used to derive ``repo_name``;
    no workspace handle is required (review flows often run without a
    worktree at all).
    """

    project_id: str = Field(..., description="Potpie project id.")
    pr_number: int = Field(..., description="PR number to comment on.")
    body: str = Field(..., description="Comment body (markdown).")
    path: Optional[str] = Field(
        default=None,
        description=(
            "File path for inline comments (relative to repo root). "
            "Required together with `line` for inline comments; omit "
            "both for a top-level conversation comment."
        ),
    )
    line: Optional[int] = Field(
        default=None,
        description="Line number for inline comments (required with `path`).",
    )
    commit_id: Optional[str] = Field(
        default=None,
        description=(
            "Commit SHA the inline comment is anchored to. Optional — "
            "the platform falls back to the PR's HEAD if omitted."
        ),
    )

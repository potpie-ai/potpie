"""Tool registration for the code_changes_manager package.

**Deprecated as an agent file-edit surface.** With the sandbox cutover,
``sandbox_text_editor`` / ``sandbox_search`` / ``sandbox_shell`` /
``sandbox_git`` cover everything an agent used to do via this module's
add_file / update_file / replace_in_file / show_diff tools. The harness no
longer registers those tools (see ``tool_service.py`` — which now wires
the sandbox tools instead).

What remains here:

  * Terminal-command tools (``execute_terminal_command``,
    ``terminal_session_output``, ``terminal_session_signal``) — orthogonal
    to file editing; they target the LocalServer tunnel, not the sandbox.
  * The ``CodeChangesManager`` Redis state, retained because the
    ``/conversations/.../sync_change`` HTTP endpoint
    (``conversations_router.py``) uses it to track IDE-side edits the
    user makes in VS Code so subsequent agent turns can see them. That
    is a different lifecycle than agent edits and stays out of the
    sandbox for now.

The legacy file-edit tools (``add_file_to_changes`` etc.) are no longer
returned by :func:`create_code_changes_management_tools`. The function
keeps its name for backwards compatibility with any out-of-tree caller
that imported it directly.
"""

import asyncio
import inspect
from typing import Callable, List, TypeVar

from app.modules.intelligence.tools.local_search_tools.execute_terminal_command_tool import (
    ExecuteTerminalCommandInput,
    execute_terminal_command_tool,
)
from app.modules.intelligence.tools.local_search_tools.terminal_session_tools import (
    TerminalSessionOutputInput,
    terminal_session_output_tool,
    TerminalSessionSignalInput,
    terminal_session_signal_tool,
)

from .tool_inputs import (
    AddFileInput,
    UpdateFileInput,
    DeleteFileInput,
    RevertFileInput,
    GetFileInput,
    ListFilesInput,
    ClearFileInput,
    ExportChangesInput,
    GetChangesForPRInput,
    UpdateFileLinesInput,
    ReplaceInFileInput,
    InsertLinesInput,
    DeleteLinesInput,
    ShowUpdatedFileInput,
    ShowDiffInput,
    GetFileDiffInput,
    GetComprehensiveMetadataInput,
)
from .tool_functions import (
    add_file_tool,
    update_file_tool,
    update_file_lines_tool,
    replace_in_file_tool,
    insert_lines_tool,
    delete_lines_tool,
    delete_file_tool,
    revert_file_tool,
    get_file_tool,
    list_files_tool,
    clear_file_tool,
    clear_all_changes_tool,
    get_changes_summary_tool,
    get_changes_for_pr_tool,
    export_changes_tool,
    show_updated_file_tool,
    show_diff_tool,
    get_file_diff_tool,
    get_comprehensive_metadata_tool,
)

T = TypeVar("T")


def _async_wrap_sync_tool(sync_func: Callable[..., T]):
    """Wrap a sync tool to run in thread pool, preventing event loop blocking.

    Code changes manager tools do sync Redis, Socket.IO, and HTTP I/O. When run on
    the event loop they block streaming (e.g. thinking parts). Running in executor
    ensures the loop stays responsive.

    asyncio.to_thread is used instead of loop.run_in_executor so that the current
    contextvars snapshot (including user_id, tunnel_url, etc.) is propagated to the
    thread. Python 3.13's run_in_executor does NOT copy contextvars automatically.
    """

    async def async_wrapper(*args: object, **kwargs: object) -> T:
        def _run() -> T:
            return sync_func(*args, **kwargs)

        return await asyncio.to_thread(_run)

    # Copy __signature__ directly so inspect.signature() returns the original
    # function's signature (needed by _adapt_func_for_from_schema to detect the
    # single-model-arg calling convention) WITHOUT setting __wrapped__.
    # Avoid functools.wraps here: Python 3.12+ makes inspect.iscoroutinefunction()
    # follow __wrapped__, which would cause it to see the sync original and return
    # False — breaking the async detection in _adapt_func_for_from_schema.
    try:
        async_wrapper.__signature__ = inspect.signature(sync_func)  # type: ignore[attr-defined]
    except (ValueError, TypeError):
        pass
    async_wrapper.__name__ = getattr(sync_func, "__name__", "async_wrapper")
    async_wrapper.__doc__ = getattr(sync_func, "__doc__", None)

    return async_wrapper  # type: ignore[return-value]


class SimpleTool:
    """Simple tool wrapper that mimics StructuredTool interface"""

    def __init__(self, name: str, description: str, func, args_schema):
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema


# Tools to exclude when local_mode=True (VS Code extension)
CODE_CHANGES_TOOLS_EXCLUDE_IN_LOCAL: frozenset[str] = frozenset(
    {
        "clear_file_from_changes",
        "clear_all_changes",
        "show_diff",
        "export_changes",
        "show_updated_file",
    }
)

# Tools to exclude when local_mode=False (web). Terminal tools require LocalServer tunnel
CODE_CHANGES_TOOLS_EXCLUDE_WHEN_NON_LOCAL: frozenset[str] = frozenset(
    {
        "execute_terminal_command",
        "terminal_session_output",
        "terminal_session_signal",
    }
)


def create_code_changes_management_tools() -> List[SimpleTool]:
    """Returns the LocalServer terminal tools only.

    The legacy file-edit tools have been retired — the sandbox tool group
    covers their use cases. The function name is preserved for backwards
    compatibility with any out-of-tree caller that imported it directly.
    """
    return [
        SimpleTool(
            name="execute_terminal_command",
            description="Execute a shell command on the user's local machine via LocalServer tunnel. Use for running tests, builds, scripts, git commands, npm/pip commands, etc. Commands run directly on the local machine within the workspace directory with security restrictions. Supports both sync (immediate results) and async (long-running) modes. Commands are validated - dangerous commands are blocked by default. Examples: 'npm test', 'git status', 'python script.py', 'npm run dev' (async mode).",
            func=_async_wrap_sync_tool(execute_terminal_command_tool),
            args_schema=ExecuteTerminalCommandInput,
        ),
        SimpleTool(
            name="terminal_session_output",
            description="Get output from an async terminal session. Use this to poll for output from a long-running command that was started with execute_terminal_command in async mode. Returns incremental output from the specified offset, allowing you to stream output from long-running processes. Use the returned offset for subsequent calls to get new output.",
            func=_async_wrap_sync_tool(terminal_session_output_tool),
            args_schema=TerminalSessionOutputInput,
        ),
        SimpleTool(
            name="terminal_session_signal",
            description="Send a signal to a terminal session (e.g., SIGINT to stop a process). Use this to control long-running processes started in async mode. Common signals: SIGINT (Ctrl+C, default), SIGTERM (graceful shutdown), SIGKILL (force kill). Example: Stop a dev server by sending SIGINT to its session.",
            func=_async_wrap_sync_tool(terminal_session_signal_tool),
            args_schema=TerminalSessionSignalInput,
        ),
    ]


def _legacy_unused_create_code_changes_management_tools() -> List[SimpleTool]:
    """Old file-edit tool registrations, retained as a string for reference.

    Not called from anywhere — left in place so a future cleanup PR can
    delete the entire ``code_changes_manager`` package once the IDE-side
    sync flow (``conversations_router.py /sync_change``) has been
    re-routed through the sandbox workspace.
    """
    tools = [
        SimpleTool(
            name="add_file_to_changes",
            description="Add a new file to the code changes manager. Use this to track new files you're creating instead of including full code in your response. This reduces token usage in conversation history. When using the VS Code extension, the response includes lines_changed, lines_added, lines_deleted; if these don't match what you intended, use get_file_from_changes to verify and fix.",
            func=_async_wrap_sync_tool(add_file_tool),
            args_schema=AddFileInput,
        ),
        SimpleTool(
            name="update_file_in_changes",
            description="Update an existing file with full content. Use ONLY when you need to replace the entire file. NEVER put placeholders like '... rest of file unchanged ...' or '// ... rest unchanged ...' in content—they are written literally and delete the rest of the file; always provide the complete file content. DON'T use when targeted edits suffice—prefer update_file_lines, replace_in_file, insert_lines, or delete_lines. When using the VS Code extension, check lines_changed/added/deleted in the response; if they don't match your intent (e.g. many lines deleted unexpectedly), use get_file_from_changes to verify and fix or revert_file then re-apply.",
            func=_async_wrap_sync_tool(update_file_tool),
            args_schema=UpdateFileInput,
        ),
        SimpleTool(
            name="update_file_lines",
            description="Update specific lines using line numbers (1-indexed). end_line is optional: omit or set to null to replace only the single line at start_line; provide end_line to replace the range start_line through end_line. DO: (1) Always provide project_id from conversation context. (2) Fetch file with get_file_from_changes with_line_numbers=true BEFORE this operation. (3) Verify changes after by refetching; check lines_changed/added/deleted in response to confirm success. (4) After insert/delete on same file, NEVER assume line numbers—refetch first. Match indentation of surrounding lines exactly. Check line stats in response to confirm the operation succeeded.",
            func=_async_wrap_sync_tool(update_file_lines_tool),
            args_schema=UpdateFileLinesInput,
        ),
        SimpleTool(
            name="replace_in_file",
            description="Exact literal string replacement (str_replace semantics). Finds old_str in the file and replaces it with new_str. old_str must match character-for-character including indentation and whitespace—no regex, no wildcards. old_str must appear EXACTLY ONCE in the file; fails if found 0 or 2+ times. Include enough surrounding lines in old_str to make it unique. DO: (1) Provide project_id from conversation context. (2) Verify after with get_file_from_changes. DON'T use for multi-occurrence replacements (fetch with get_file_from_changes and use replace_in_file or update_file_lines for each occurrence). When using the VS Code extension, use get_file_from_changes to verify the result.",
            func=_async_wrap_sync_tool(replace_in_file_tool),
            args_schema=ReplaceInFileInput,
        ),
        SimpleTool(
            name="insert_lines",
            description="Insert content at a specific line (1-indexed). Set insert_after=False to insert before. DO: (1) Always provide project_id from conversation context. (2) Fetch file with get_file_from_changes with_line_numbers=true BEFORE this operation. (3) Verify after by refetching; check lines_added in response to confirm success. (4) After insert/delete on same file, NEVER assume line numbers—refetch first. Match indentation of surrounding lines exactly.",
            func=_async_wrap_sync_tool(insert_lines_tool),
            args_schema=InsertLinesInput,
        ),
        SimpleTool(
            name="delete_lines",
            description="Delete specific lines (1-indexed). Specify start_line and optionally end_line. DO: (1) Always provide project_id from conversation context. (2) Fetch file with get_file_from_changes with_line_numbers=true BEFORE this operation. (3) Verify after by refetching; check lines_deleted in response to confirm success. (4) After insert/delete on same file, NEVER assume line numbers—refetch first. DON'T skip verification. If lines_deleted doesn't match your range, use get_file_from_changes to verify and fix.",
            func=_async_wrap_sync_tool(delete_lines_tool),
            args_schema=DeleteLinesInput,
        ),
        SimpleTool(
            name="delete_file_in_changes",
            description="Mark a file for deletion in the code changes manager. File content is preserved by default so you can reference it later. When using the VS Code extension, the response includes lines_changed, lines_added, lines_deleted (lines_deleted = file line count before delete); if the file wasn't removed as expected, use get_file_from_changes to verify.",
            func=_async_wrap_sync_tool(delete_file_tool),
            args_schema=DeleteFileInput,
        ),
        SimpleTool(
            name="revert_file",
            description=(
                "Revert a file to last saved or git HEAD (local mode only). "
                "Use when connected via the VS Code extension. "
                "target='saved' (default): restore from disk (discard unsaved changes). "
                "target='HEAD': restore from git HEAD (committed version), then save. "
                "Content is applied directly in the IDE. "
                "When using the extension, the response includes lines_changed, lines_added, lines_deleted; use these to confirm the revert applied correctly."
            ),
            func=_async_wrap_sync_tool(revert_file_tool),
            args_schema=RevertFileInput,
        ),
        SimpleTool(
            name="get_file_from_changes",
            description="Get file content from the code changes manager. REQUIRED before any line-based operation (update_file_lines, insert_lines, delete_lines)—use with_line_numbers=true to see exact line numbers. Use after EACH edit to verify changes. Check line stats (lines_changed/added/deleted) in tool responses to confirm operations succeeded. After insert/delete, refetch before subsequent line operations—never assume line numbers. DON'T skip verification steps.",
            func=_async_wrap_sync_tool(get_file_tool),
            args_schema=GetFileInput,
        ),
        SimpleTool(
            name="list_files_in_changes",
            description="List all files in the code changes manager, optionally filtered by change type (add/update/delete) or file path pattern (regex).",
            func=_async_wrap_sync_tool(list_files_tool),
            args_schema=ListFilesInput,
        ),
        SimpleTool(
            name="clear_file_from_changes",
            description="Remove a specific file from the code changes manager (discard its changes).",
            func=_async_wrap_sync_tool(clear_file_tool),
            args_schema=ClearFileInput,
        ),
        SimpleTool(
            name="clear_all_changes",
            description="Clear all files from the code changes manager (discard all changes).",
            func=_async_wrap_sync_tool(clear_all_changes_tool),
            args_schema=None,
        ),
        SimpleTool(
            name="get_changes_summary",
            description="Get a summary overview of all code changes including file counts by change type.",
            func=_async_wrap_sync_tool(get_changes_summary_tool),
            args_schema=None,
        ),
        SimpleTool(
            name="get_changes_for_pr",
            description="Get summary of code changes for a conversation by conversation_id. Use when delegated to create a PR: verify changes exist in Redis before calling create_pr_workflow. Takes conversation_id explicitly (pass from delegate context).",
            func=_async_wrap_sync_tool(get_changes_for_pr_tool),
            args_schema=GetChangesForPRInput,
        ),
        SimpleTool(
            name="export_changes",
            description="Export all code changes in various formats (dict, list, json, or diff). Use 'json' format for persistence, 'diff' format for git-style patch.",
            func=_async_wrap_sync_tool(export_changes_tool),
            args_schema=ExportChangesInput,
        ),
        SimpleTool(
            name="show_updated_file",
            description=(
                "Display the complete updated content of one or more files. This tool streams the full file content "
                "directly into the agent response without going through the LLM, allowing users to see the complete edited files. "
                "\n\n"
                "**Parameters:**\n"
                "- file_paths (optional): Array/list of file paths to show. Examples: ['src/main.py'] or ['file1.py', 'file2.py']. "
                "MUST be a list/array, not a JSON string. If omitted or empty, shows ALL changed files.\n\n"
                "**When to use:**\n"
                "- When the user asks to see updated files\n"
                "- To showcase the final result of files you just edited\n"
                "- The content is automatically shown to the user without consuming LLM context"
            ),
            func=_async_wrap_sync_tool(show_updated_file_tool),
            args_schema=ShowUpdatedFileInput,
        ),
        SimpleTool(
            name="show_diff",
            description="Display unified diffs of all changes. Use at the END of your response to show all code changes. Always provide project_id from conversation context for accurate diffs against the branch. Increase context_lines (default 3) when changes are spread across many lines. Streams directly to user without consuming LLM context. REQUIRED: Call this after completing all modifications to display the full diff.",
            func=_async_wrap_sync_tool(show_diff_tool),
            args_schema=ShowDiffInput,
        ),
        SimpleTool(
            name="get_file_diff",
            description="Get diff for a specific file against the repository branch. Always provide project_id from conversation context. Increase context_lines (default 3) when changes span many lines. Use to verify what changed in a single file.",
            func=_async_wrap_sync_tool(get_file_diff_tool),
            args_schema=GetFileDiffInput,
        ),
        SimpleTool(
            name="get_session_metadata",
            description="Get comprehensive metadata about all code changes in the current session. Shows complete state of all files being managed, including timestamps, descriptions, change types, and line counts. Use this to review your session progress and understand what files have been modified. This is your session state - all your work is tracked here.",
            func=_async_wrap_sync_tool(get_comprehensive_metadata_tool),
            args_schema=GetComprehensiveMetadataInput,
        ),
        SimpleTool(
            name="execute_terminal_command",
            description="Execute a shell command on the user's local machine via LocalServer tunnel. Use for running tests, builds, scripts, git commands, npm/pip commands, etc. Commands run directly on the local machine within the workspace directory with security restrictions. Supports both sync (immediate results) and async (long-running) modes. Commands are validated - dangerous commands are blocked by default. Examples: 'npm test', 'git status', 'python script.py', 'npm run dev' (async mode).",
            func=_async_wrap_sync_tool(execute_terminal_command_tool),
            args_schema=ExecuteTerminalCommandInput,
        ),
        SimpleTool(
            name="terminal_session_output",
            description="Get output from an async terminal session. Use this to poll for output from a long-running command that was started with execute_terminal_command in async mode. Returns incremental output from the specified offset, allowing you to stream output from long-running processes. Use the returned offset for subsequent calls to get new output.",
            func=_async_wrap_sync_tool(terminal_session_output_tool),
            args_schema=TerminalSessionOutputInput,
        ),
        SimpleTool(
            name="terminal_session_signal",
            description="Send a signal to a terminal session (e.g., SIGINT to stop a process). Use this to control long-running processes started in async mode. Common signals: SIGINT (Ctrl+C, default), SIGTERM (graceful shutdown), SIGKILL (force kill). Example: Stop a dev server by sending SIGINT to its session.",
            func=_async_wrap_sync_tool(terminal_session_signal_tool),
            args_schema=TerminalSessionSignalInput,
        ),
    ]

    return tools

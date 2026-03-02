"""Tool registration: SimpleTool, exclusion sets, create_code_changes_management_tools."""

from typing import List

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
    SearchContentInput,
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
    search_content_tool,
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
    """Create all code changes management tools"""

    tools = [
        SimpleTool(
            name="add_file_to_changes",
            description="Add a new file to the code changes manager. Use this to track new files you're creating instead of including full code in your response. This reduces token usage in conversation history. When using the VS Code extension, the response includes lines_changed, lines_added, lines_deleted; if these don't match what you intended, use get_file_from_changes to verify and fix.",
            func=add_file_tool,
            args_schema=AddFileInput,
        ),
        SimpleTool(
            name="update_file_in_changes",
            description="Update an existing file with full content. Use ONLY when you need to replace the entire file. NEVER put placeholders like '... rest of file unchanged ...' or '// ... rest unchanged ...' in content—they are written literally and delete the rest of the file; always provide the complete file content. DON'T use when targeted edits suffice—prefer update_file_lines, replace_in_file, insert_lines, or delete_lines. When using the VS Code extension, check lines_changed/added/deleted in the response; if they don't match your intent (e.g. many lines deleted unexpectedly), use get_file_from_changes to verify and fix or revert_file then re-apply.",
            func=update_file_tool,
            args_schema=UpdateFileInput,
        ),
        SimpleTool(
            name="update_file_lines",
            description="Update specific lines using line numbers (1-indexed). end_line is optional: omit or set to null to replace only the single line at start_line; provide end_line to replace the range start_line through end_line. DO: (1) Always provide project_id from conversation context. (2) Fetch file with get_file_from_changes with_line_numbers=true BEFORE this operation. (3) Verify changes after by refetching; check lines_changed/added/deleted in response to confirm success. (4) After insert/delete on same file, NEVER assume line numbers—refetch first. Match indentation of surrounding lines exactly. Check line stats in response to confirm the operation succeeded.",
            func=update_file_lines_tool,
            args_schema=UpdateFileLinesInput,
        ),
        SimpleTool(
            name="replace_in_file",
            description="Replace pattern matches using regex. Use word_boundary=True for safe replacements (prevents partial matches—e.g. replacing 'get_db' won't match 'get_database'). Supports capturing groups (\\1, \\2). Set count=0 for all occurrences. DO: (1) Provide project_id from conversation context. (2) Verify after with get_file_from_changes; check line stats in response. DON'T skip verification. When using the VS Code extension, if lines_changed/added/deleted don't match intent, use get_file_from_changes to verify and fix.",
            func=replace_in_file_tool,
            args_schema=ReplaceInFileInput,
        ),
        SimpleTool(
            name="insert_lines",
            description="Insert content at a specific line (1-indexed). Set insert_after=False to insert before. DO: (1) Always provide project_id from conversation context. (2) Fetch file with get_file_from_changes with_line_numbers=true BEFORE this operation. (3) Verify after by refetching; check lines_added in response to confirm success. (4) After insert/delete on same file, NEVER assume line numbers—refetch first. Match indentation of surrounding lines exactly.",
            func=insert_lines_tool,
            args_schema=InsertLinesInput,
        ),
        SimpleTool(
            name="delete_lines",
            description="Delete specific lines (1-indexed). Specify start_line and optionally end_line. DO: (1) Always provide project_id from conversation context. (2) Fetch file with get_file_from_changes with_line_numbers=true BEFORE this operation. (3) Verify after by refetching; check lines_deleted in response to confirm success. (4) After insert/delete on same file, NEVER assume line numbers—refetch first. DON'T skip verification. If lines_deleted doesn't match your range, use get_file_from_changes to verify and fix.",
            func=delete_lines_tool,
            args_schema=DeleteLinesInput,
        ),
        SimpleTool(
            name="delete_file_in_changes",
            description="Mark a file for deletion in the code changes manager. File content is preserved by default so you can reference it later. When using the VS Code extension, the response includes lines_changed, lines_added, lines_deleted (lines_deleted = file line count before delete); if the file wasn't removed as expected, use get_file_from_changes to verify.",
            func=delete_file_tool,
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
            func=revert_file_tool,
            args_schema=RevertFileInput,
        ),
        SimpleTool(
            name="get_file_from_changes",
            description="Get file content from the code changes manager. REQUIRED before any line-based operation (update_file_lines, insert_lines, delete_lines)—use with_line_numbers=true to see exact line numbers. Use after EACH edit to verify changes. Check line stats (lines_changed/added/deleted) in tool responses to confirm operations succeeded. After insert/delete, refetch before subsequent line operations—never assume line numbers. DON'T skip verification steps.",
            func=get_file_tool,
            args_schema=GetFileInput,
        ),
        SimpleTool(
            name="list_files_in_changes",
            description="List all files in the code changes manager, optionally filtered by change type (add/update/delete) or file path pattern (regex).",
            func=list_files_tool,
            args_schema=ListFilesInput,
        ),
        SimpleTool(
            name="search_content_in_changes",
            description="Search for a pattern in file contents using regex (grep-like functionality). Supports filtering by file path pattern. Returns matching lines with line numbers.",
            func=search_content_tool,
            args_schema=SearchContentInput,
        ),
        SimpleTool(
            name="clear_file_from_changes",
            description="Remove a specific file from the code changes manager (discard its changes).",
            func=clear_file_tool,
            args_schema=ClearFileInput,
        ),
        SimpleTool(
            name="clear_all_changes",
            description="Clear all files from the code changes manager (discard all changes).",
            func=clear_all_changes_tool,
            args_schema=None,
        ),
        SimpleTool(
            name="get_changes_summary",
            description="Get a summary overview of all code changes including file counts by change type.",
            func=get_changes_summary_tool,
            args_schema=None,
        ),
        SimpleTool(
            name="get_changes_for_pr",
            description="Get summary of code changes for a conversation by conversation_id. Use when delegated to create a PR: verify changes exist in Redis before calling create_pr_workflow. Takes conversation_id explicitly (pass from delegate context).",
            func=get_changes_for_pr_tool,
            args_schema=GetChangesForPRInput,
        ),
        SimpleTool(
            name="export_changes",
            description="Export all code changes in various formats (dict, list, json, or diff). Use 'json' format for persistence, 'diff' format for git-style patch.",
            func=export_changes_tool,
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
            func=show_updated_file_tool,
            args_schema=ShowUpdatedFileInput,
        ),
        SimpleTool(
            name="show_diff",
            description="Display unified diffs of all changes. Use at the END of your response to show all code changes. Always provide project_id from conversation context for accurate diffs against the branch. Increase context_lines (default 3) when changes are spread across many lines. Streams directly to user without consuming LLM context. REQUIRED: Call this after completing all modifications to display the full diff.",
            func=show_diff_tool,
            args_schema=ShowDiffInput,
        ),
        SimpleTool(
            name="get_file_diff",
            description="Get diff for a specific file against the repository branch. Always provide project_id from conversation context. Increase context_lines (default 3) when changes span many lines. Use to verify what changed in a single file.",
            func=get_file_diff_tool,
            args_schema=GetFileDiffInput,
        ),
        SimpleTool(
            name="get_session_metadata",
            description="Get comprehensive metadata about all code changes in the current session. Shows complete state of all files being managed, including timestamps, descriptions, change types, and line counts. Use this to review your session progress and understand what files have been modified. This is your session state - all your work is tracked here.",
            func=get_comprehensive_metadata_tool,
            args_schema=GetComprehensiveMetadataInput,
        ),
        SimpleTool(
            name="execute_terminal_command",
            description="Execute a shell command on the user's local machine via LocalServer tunnel. Use for running tests, builds, scripts, git commands, npm/pip commands, etc. Commands run directly on the local machine within the workspace directory with security restrictions. Supports both sync (immediate results) and async (long-running) modes. Commands are validated - dangerous commands are blocked by default. Examples: 'npm test', 'git status', 'python script.py', 'npm run dev' (async mode).",
            func=execute_terminal_command_tool,
            args_schema=ExecuteTerminalCommandInput,
        ),
        SimpleTool(
            name="terminal_session_output",
            description="Get output from an async terminal session. Use this to poll for output from a long-running command that was started with execute_terminal_command in async mode. Returns incremental output from the specified offset, allowing you to stream output from long-running processes. Use the returned offset for subsequent calls to get new output.",
            func=terminal_session_output_tool,
            args_schema=TerminalSessionOutputInput,
        ),
        SimpleTool(
            name="terminal_session_signal",
            description="Send a signal to a terminal session (e.g., SIGINT to stop a process). Use this to control long-running processes started in async mode. Common signals: SIGINT (Ctrl+C, default), SIGTERM (graceful shutdown), SIGKILL (force kill). Example: Stop a dev server by sending SIGINT to its session.",
            func=terminal_session_signal_tool,
            args_schema=TerminalSessionSignalInput,
        ),
    ]

    return tools

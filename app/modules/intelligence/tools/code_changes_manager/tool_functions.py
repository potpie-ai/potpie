"""Tool function implementations for code changes management."""

import os
import json
import uuid
from datetime import datetime

from app.modules.utils.logger import setup_logger

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
from .context import _get_local_mode, _get_conversation_id
from .lifecycle import _get_code_changes_manager, _get_project_id_from_conversation_id
from .routing import (
    _execute_local_write,
    _should_route_to_local_server,
    _route_to_local_server,
    _sync_file_from_local_server_to_redis,
    _fetch_file_content_from_local_server,
)
from .diff import create_unified_diff, generate_git_diff_patch, fetch_repo_file_content_for_diff
from .content_resolver import read_file_from_codebase
from .models import ChangeType

logger = setup_logger(__name__)

# Tool functions
def add_file_tool(input_data: AddFileInput) -> str:
    """Add a new file to the code changes manager"""
    # Resolve project_id: use provided value or look up from conversation_id (non-local mode only)
    project_id = input_data.project_id
    if not project_id and not _get_local_mode():
        conversation_id = _get_conversation_id()
        project_id = _get_project_id_from_conversation_id(conversation_id)
        if project_id:
            logger.info(
                f"Tool add_file_tool: Resolved project_id={project_id} from conversation_id={conversation_id}"
            )

    logger.info(f"🔧 [Tool Call] add_file_tool: Adding file '{input_data.file_path}'")

    # LOCAL-FIRST: Try local execution first
    local_result = _execute_local_write(
        "add_file",
        {
            "file_path": input_data.file_path,
            "content": input_data.content,
            "description": input_data.description,
        },
        input_data.file_path,
    )

    if local_result is not None:
        # Local execution was attempted - return result (success or error)
        return local_result

    # No tunnel available - use cloud CodeChangesManager (for web UI users)
    try:
        manager = _get_code_changes_manager()
        db = None
        if project_id:
            from app.core.database import get_db
            db = next(get_db())
        success = manager.add_file(
            file_path=input_data.file_path,
            content=input_data.content,
            description=input_data.description,
            project_id=project_id,
            db=db,
        )

        if success:
            summary = manager.get_summary()
            return f"✅ Added file '{input_data.file_path}' (cloud)\n\nTotal files in changes: {summary['total_files']}"
        else:
            return f"❌ File '{input_data.file_path}' already exists in changes. Use update_file to modify it."
    except Exception:
        logger.exception(
            "Tool add_file_tool: Error adding file", file_path=input_data.file_path
        )
        return "❌ Error adding file"


def update_file_tool(input_data: UpdateFileInput) -> str:
    """Update a file in the code changes manager"""
    logger.info(
        f"🔧 [Tool Call] update_file_tool: Updating file '{input_data.file_path}'"
    )

    # LOCAL-FIRST: Try local execution first
    local_result = _execute_local_write(
        "update_file",
        {
            "file_path": input_data.file_path,
            "content": input_data.content,
            "description": input_data.description,
        },
        input_data.file_path,
    )

    if local_result is not None:
        # Local execution was attempted - return result (success or error)
        return local_result

    # No tunnel available - use cloud CodeChangesManager (for web UI users)
    try:
        manager = _get_code_changes_manager()
        db = None
        if input_data.project_id:
            from app.core.database import get_db
            db = next(get_db())
        success = manager.update_file(
            file_path=input_data.file_path,
            content=input_data.content,
            description=input_data.description,
            preserve_previous=input_data.preserve_previous,
            project_id=input_data.project_id,
            db=db,
        )

        if success:
            summary = manager.get_summary()
            return f"✅ Updated file '{input_data.file_path}' (cloud)\n\nTotal files in changes: {summary['total_files']}"
        else:
            return f"❌ Error updating file '{input_data.file_path}'"
    except Exception:
        logger.exception(
            "Tool update_file_tool: Error updating file", file_path=input_data.file_path
        )
        return "❌ Error updating file"


def revert_file_tool(input_data: RevertFileInput) -> str:
    """Revert a file to last saved or git HEAD (local mode only).

    Only available when connected via the VS Code extension (LocalServer).
    Restore from disk (saved) or from git HEAD; content is applied in the IDE.
    """
    logger.info(
        f"🔧 [Tool Call] revert_file_tool: Reverting '{input_data.file_path}' to {input_data.target or 'saved'}"
    )

    # Local-only: revert is implemented by LocalServer (POST /api/files/revert)
    data = {
        "path": input_data.file_path,
        "file_path": input_data.file_path,
        "target": input_data.target or "saved",
        "description": input_data.description,
    }
    local_result = _execute_local_write("revert_file", data, input_data.file_path)

    if local_result is not None:
        return local_result

    return (
        "❌ **Revert is only available in local mode.**\n\n"
        "Connect via the VS Code extension (Potpie) so the agent can revert files "
        "to the last saved version or to git HEAD directly in your IDE."
    )


def delete_file_tool(input_data: DeleteFileInput) -> str:
    """Delete a file locally or mark for deletion in cloud"""
    logger.info(f"Tool delete_file_tool: Deleting file '{input_data.file_path}'")

    # LOCAL-FIRST: Try local execution first
    local_result = _execute_local_write(
        "delete_file",
        {
            "file_path": input_data.file_path,
            "description": input_data.description,
        },
        input_data.file_path,
    )

    if local_result is not None:
        # Local execution was attempted - return result (success or error)
        return local_result

    # No tunnel available - use cloud CodeChangesManager (for web UI users)
    try:
        manager = _get_code_changes_manager()
        db = None
        if input_data.project_id:
            from app.core.database import get_db
            db = next(get_db())
        success = manager.delete_file(
            file_path=input_data.file_path,
            description=input_data.description,
            preserve_content=input_data.preserve_content,
            project_id=input_data.project_id,
            db=db,
        )

        if success:
            summary = manager.get_summary()
            return f"✅ Marked file '{input_data.file_path}' for deletion (cloud)\n\nTotal files in changes: {summary['total_files']}"
        else:
            return f"❌ Error deleting file '{input_data.file_path}'"
    except Exception:
        logger.exception(
            "Tool delete_file_tool: Error deleting file", file_path=input_data.file_path
        )
        return "❌ Error deleting file"


def get_file_tool(input_data: GetFileInput) -> str:
    """Get comprehensive change information and metadata for a specific file"""
    logger.info(f"Tool get_file_tool: Retrieving file '{input_data.file_path}'")

    # Check if we should route to LocalServer
    if _should_route_to_local_server():
        logger.info(f"🔧 [Tool Call] Routing get_file_tool to LocalServer")
        result = _route_to_local_server(
            "get_file",
            {
                "file_path": input_data.file_path,
            },
        )
        if result:
            return result
        # LocalServer returned nothing (e.g. file not in workspace) - sync from local so fallback has fresh state
        _sync_file_from_local_server_to_redis(input_data.file_path)

    # Fall back to CodeChangesManager (or use manager after syncing from local)
    try:
        manager = _get_code_changes_manager()
        file_data = manager.get_file(input_data.file_path)

        if file_data:
            change_emoji = {"add": "➕", "update": "✏️", "delete": "🗑️"}
            emoji = change_emoji.get(file_data["change_type"], "📄")

            result = f"{emoji} **{file_data['file_path']}**\n\n"
            result += f"**Change Type:** {file_data['change_type']}\n"
            result += f"**Created:** {file_data['created_at']}\n"
            result += f"**Last Updated:** {file_data['updated_at']}\n"

            if file_data.get("description"):
                result += f"**Description:** {file_data['description']}\n"

            # Line count information
            if file_data["change_type"] == "delete":
                result += "\n⚠️ **File marked for deletion**\n"
                if file_data.get("previous_content"):
                    prev_lines = file_data["previous_content"].split("\n")
                    result += f"**Original Lines:** {len(prev_lines)}\n"
                    result += f"**Original Size:** {len(file_data['previous_content'])} chars\n"
                    result += f"\n**Previous content preview (first 300 chars):**\n```\n{file_data['previous_content'][:300]}...\n```\n"
            else:
                lines = []
                if file_data.get("content"):
                    lines = file_data["content"].split("\n")
                    result += f"\n**Current Lines:** {len(lines)}\n"
                    result += f"**Current Size:** {len(file_data['content'])} chars\n"
                    content_preview = file_data["content"][:500]
                    result += f"\n**Content preview (first 500 chars):**\n```\n{content_preview}\n```\n"
                    if len(file_data["content"]) > 500:
                        result += f"\n... ({len(file_data['content']) - 500} more characters)\n"

                if (
                    file_data.get("previous_content")
                    and file_data["change_type"] == "update"
                    and lines
                ):
                    prev_lines = file_data["previous_content"].split("\n")
                    result += f"\n**Previous Lines:** {len(prev_lines)}\n"
                    result += f"**Previous Size:** {len(file_data['previous_content'])} chars\n"
                    result += (
                        f"**Line Change:** {len(lines) - len(prev_lines):+d} lines\n"
                    )

            result += "\n💡 **Tip:** Use `get_file_diff` to see the diff for this file against the repository branch."

            return result
        else:
            return f"❌ File '{input_data.file_path}' not found in changes"
    except Exception:
        logger.exception(
            "Tool get_file_tool: Error retrieving file", file_path=input_data.file_path
        )
        return "❌ Error retrieving file"


def list_files_tool(input_data: ListFilesInput) -> str:
    """List all files with changes, optionally filtered"""
    logger.info(
        f"Tool list_files_tool: Listing files (filter: {input_data.change_type_filter}, pattern: {input_data.path_pattern})"
    )
    try:
        manager = _get_code_changes_manager()

        change_type_filter = None
        if input_data.change_type_filter:
            try:
                change_type_filter = ChangeType(input_data.change_type_filter.lower())
            except ValueError:
                return f"❌ Invalid change type '{input_data.change_type_filter}'. Valid types: add, update, delete"

        files = manager.list_files(
            change_type_filter=change_type_filter,
            path_pattern=input_data.path_pattern,
        )

        if not files:
            filter_text = ""
            if input_data.change_type_filter:
                filter_text += f" with change type '{input_data.change_type_filter}'"
            if input_data.path_pattern:
                filter_text += f" matching pattern '{input_data.path_pattern}'"
            return f"📋 No files found{filter_text}"

        result = f"📋 **Files in Changes** ({len(files)} files)\n\n"

        change_emoji = {"add": "➕", "update": "✏️", "delete": "🗑️"}

        for file_data in files:
            emoji = change_emoji.get(file_data["change_type"], "📄")
            result += (
                f"{emoji} **{file_data['file_path']}** ({file_data['change_type']})\n"
            )
            if file_data.get("description"):
                result += f"   Description: {file_data['description']}\n"
            result += f"   Updated: {file_data['updated_at']}\n\n"

        return result
    except Exception:
        logger.exception("Tool list_files_tool: Error listing files")
        return "❌ Error listing files"


def search_content_tool(input_data: SearchContentInput) -> str:
    """Search for pattern in file contents (grep-like functionality)"""
    logger.info(
        f"Tool search_content_tool: Searching for pattern '{input_data.pattern}' (file_pattern: {input_data.file_pattern})"
    )
    try:
        manager = _get_code_changes_manager()
        matches = manager.search_content(
            pattern=input_data.pattern,
            file_pattern=input_data.file_pattern,
            case_sensitive=input_data.case_sensitive,
        )

        if not matches:
            filter_text = ""
            if input_data.file_pattern:
                filter_text = f" in files matching '{input_data.file_pattern}'"
            return (
                f"🔍 No matches found for pattern '{input_data.pattern}'{filter_text}"
            )

        # Group matches by file
        matches_by_file: Dict[str, List[Dict[str, Any]]] = {}
        for match in matches:
            if "error" in match:
                return f"❌ {match['error']}"
            file_path = match["file_path"]
            if file_path not in matches_by_file:
                matches_by_file[file_path] = []
            matches_by_file[file_path].append(match)

        result = f"🔍 **Search Results** ({len(matches)} matches in {len(matches_by_file)} files)\n\n"
        result += f"Pattern: `{input_data.pattern}`\n\n"

        for file_path, file_matches in matches_by_file.items():
            result += f"📄 **{file_path}** ({len(file_matches)} matches):\n"
            for match in file_matches[:10]:  # Show first 10 matches per file
                result += f"  Line {match['line_number']}: {match['line']}\n"
            if len(file_matches) > 10:
                result += f"  ... and {len(file_matches) - 10} more matches\n"
            result += "\n"

        return result
    except Exception:
        logger.exception(
            "Tool search_content_tool: Error searching content",
            pattern=input_data.pattern,
        )
        return "❌ Error searching content"


def clear_file_tool(input_data: ClearFileInput) -> str:
    """Clear changes for a specific file"""
    logger.info(f"Tool clear_file_tool: Clearing file '{input_data.file_path}'")
    try:
        manager = _get_code_changes_manager()
        success = manager.clear_file(input_data.file_path)

        if success:
            summary = manager.get_summary()
            return f"✅ Cleared changes for '{input_data.file_path}'\n\nTotal files in changes: {summary['total_files']}"
        else:
            return f"❌ File '{input_data.file_path}' not found in changes"
    except Exception:
        logger.exception(
            "Tool clear_file_tool: Error clearing file", file_path=input_data.file_path
        )
        return "❌ Error clearing file"


def clear_all_changes_tool() -> str:
    """Clear all changes from the code changes manager"""
    logger.info("Tool clear_all_changes_tool: Clearing all changes")
    try:
        manager = _get_code_changes_manager()
        count = manager.clear_all()
        return f"✅ Cleared all changes ({count} files removed)"
    except Exception:
        logger.exception("Tool clear_all_changes_tool: Error clearing all changes")
        return "❌ Error clearing all changes"


def get_changes_summary_tool() -> str:
    """Get a summary of all code changes"""
    logger.info("Tool get_changes_summary_tool: Getting summary")
    try:
        manager = _get_code_changes_manager()
        summary = manager.get_summary()

        cid = summary.get("conversation_id") or "(ephemeral)"
        result = f"📊 **Code Changes Summary** (Conversation: {cid})\n\n"
        result += f"Total files: {summary['total_files']}\n\n"

        change_emoji = {"add": "➕", "update": "✏️", "delete": "🗑️"}

        result += "**Change Types:**\n"
        for change_type, count in summary["change_counts"].items():
            emoji = change_emoji.get(change_type, "📄")
            result += f"{emoji} {change_type.title()}: {count}\n"

        if summary["files"]:
            result += "\n**Files:**\n"
            for file_info in summary["files"][:10]:  # Show first 10
                emoji = change_emoji.get(file_info["change_type"], "📄")
                result += (
                    f"{emoji} {file_info['file_path']} ({file_info['change_type']})\n"
                )
            if len(summary["files"]) > 10:
                result += f"... and {len(summary['files']) - 10} more files\n"

        return result
    except Exception:
        logger.exception("Tool get_summary_tool: Error getting summary")
        return "❌ Error getting summary"


def get_changes_for_pr_tool(input_data: GetChangesForPRInput) -> str:
    """
    Get summary of code changes for a conversation (for delegated PR flows).

    Use when delegated to create a PR: verify changes exist in Redis before calling
    create_pr_workflow. Takes conversation_id explicitly so sub-agents can fetch
    changes for the parent conversation.
    """
    logger.info(
        f"Tool get_changes_for_pr_tool: Getting changes for conversation {input_data.conversation_id}"
    )
    try:
        manager = CodeChangesManager(conversation_id=input_data.conversation_id)
        summary = manager.get_summary()

        if summary["total_files"] == 0:
            return "📋 No changes found for this conversation. Run code modifications first, or ensure the conversation_id is correct."

        change_emoji = {"add": "➕", "update": "✏️", "delete": "🗑️"}

        result = (
            f"📋 **Changes for PR** (conversation: {input_data.conversation_id})\n\n"
            f"**Total files:** {summary['total_files']}\n\n"
        )
        result += "**Change Types:**\n"
        for change_type, count in summary["change_counts"].items():
            if count > 0:
                emoji = change_emoji.get(change_type, "📄")
                result += f"{emoji} {change_type.title()}: {count}\n"

        result += "\n**Files:**\n"
        for file_info in summary["files"]:
            emoji = change_emoji.get(file_info["change_type"], "📄")
            result += f"{emoji} {file_info['file_path']} ({file_info['change_type']})\n"

        result += "\n💡 Call `create_pr_workflow` with project_id, conversation_id, branch_name, commit_message, pr_title, pr_body to create the PR."
        return result
    except Exception:
        logger.exception("Tool get_changes_for_pr_tool: Error getting changes")
        return "❌ Error fetching changes. Check conversation_id is correct and changes exist in Redis."


def update_file_lines_tool(input_data: UpdateFileLinesInput) -> str:
    """Update specific lines in a file using line numbers"""
    logger.info(
        f"🔧 [Tool Call] update_file_lines_tool: Updating lines {input_data.start_line}-{input_data.end_line or input_data.start_line} "
        f"in '{input_data.file_path}' (project_id={input_data.project_id})"
    )

    # LOCAL-FIRST: Try local execution first
    local_result = _execute_local_write(
        "update_file_lines",
        {
            "file_path": input_data.file_path,
            "start_line": input_data.start_line,
            "end_line": input_data.end_line,
            "new_content": input_data.new_content,
            "description": input_data.description,
            "project_id": input_data.project_id,
        },
        input_data.file_path,
    )

    if local_result is not None:
        # Local execution was attempted - return result (success or error)
        return local_result

    # No tunnel available - use cloud CodeChangesManager (for web UI users)
    try:
        manager = _get_code_changes_manager()
        db = None
        if input_data.project_id:
            logger.info(
                f"Tool update_file_lines_tool: Project ID provided ({input_data.project_id}), fetching database session"
            )
            from app.core.database import get_db

            db = next(get_db())
            logger.debug("Tool update_file_lines_tool: Database session obtained")
        # project_id is now required, so this shouldn't happen, but keep for safety
        if not input_data.project_id:
            logger.error(
                "Tool update_file_lines_tool: ERROR - project_id is required but was not provided!"
            )
            return "❌ Error: project_id is required to update file lines. Please provide the project_id from the conversation context."
        result = manager.update_file_lines(
            file_path=input_data.file_path,
            start_line=input_data.start_line,
            end_line=input_data.end_line,
            new_content=input_data.new_content,
            description=input_data.description,
            project_id=input_data.project_id,
            db=db,
        )

        if result.get("success"):
            context_str = ""
            if result.get("updated_context"):
                context_start = result.get("context_start_line", result["start_line"])
                context_end = result.get("context_end_line", result["end_line"])
                context_str = f"\nUpdated lines with context (lines {context_start}-{context_end}):\n```{input_data.file_path}\n{result['updated_context']}\n```"
            return (
                f"✅ Updated lines {result['start_line']}-{result['end_line']} in '{input_data.file_path}'\n\n"
                + f"Replaced {result['lines_replaced']} lines with {result['lines_added']} new lines\n"
                + f"Replaced content:\n```\n{result['replaced_content'][:200]}{'...' if len(result['replaced_content']) > 200 else ''}\n```"
                + context_str
            )
        else:
            return f"❌ Error updating lines: {result.get('error', 'Unknown error')}"
    except Exception:
        logger.exception(
            "Tool update_file_lines_tool: Error updating file lines",
            file_path=input_data.file_path,
        )
        return "❌ Error updating file lines"


def replace_in_file_tool(input_data: ReplaceInFileInput) -> str:
    """Replace an exact literal string in a file (str_replace semantics)."""
    # Resolve project_id: use provided value or look up from conversation_id (non-local mode only)
    project_id = input_data.project_id
    if not project_id and not _get_local_mode():
        conversation_id = _get_conversation_id()
        project_id = _get_project_id_from_conversation_id(conversation_id)
        if project_id:
            logger.info(
                f"Tool replace_in_file_tool: Resolved project_id={project_id} from conversation_id={conversation_id}"
            )

    logger.info(
        f"🔧 [Tool Call] replace_in_file_tool: str_replace in '{input_data.file_path}' "
        f"(project_id={project_id})"
    )

    # LOCAL-FIRST: Try local execution first
    local_result = _execute_local_write(
        "replace_in_file",
        {
            "file_path": input_data.file_path,
            "old_str": input_data.old_str,
            "new_str": input_data.new_str,
            "description": input_data.description,
        },
        input_data.file_path,
    )

    if local_result is not None:
        return local_result

    # No tunnel available - use cloud CodeChangesManager (for web UI users)
    try:
        manager = _get_code_changes_manager()
        db = None
        if project_id:
            logger.info(
                f"Tool replace_in_file_tool: Project ID available ({project_id}), fetching database session"
            )
            from app.core.database import get_db

            db = next(get_db())
            logger.debug("Tool replace_in_file_tool: Database session obtained")
        result = manager.replace_in_file(
            file_path=input_data.file_path,
            old_str=input_data.old_str,
            new_str=input_data.new_str,
            description=input_data.description,
            project_id=project_id,
            db=db,
        )

        if result.get("success"):
            msg = (
                f"✅ str_replace applied in '{input_data.file_path}' "
                f"(match at line ~{result['match_line']})"
            )
            # Append diff and line stats from manager (cloud path has no LocalServer)
            file_data = manager.get_file(input_data.file_path)
            if (
                file_data
                and file_data.get("previous_content")
                and file_data.get("content")
            ):
                try:
                    diff = create_unified_diff(
                        file_data["previous_content"],
                        file_data["content"],
                        input_data.file_path,
                        input_data.file_path,
                        3,
                    )
                    if diff:
                        msg += (
                            "\n\n**Diff (uncommitted changes):**\n```diff\n"
                            + diff
                            + "\n```"
                        )
                    old_lines = len(file_data["previous_content"].splitlines())
                    new_lines = len(file_data["content"].splitlines())
                    msg += (
                        f"\n\n**Line stats:** lines_added={max(0, new_lines - old_lines)}, "
                        f"lines_deleted={max(0, old_lines - new_lines)}"
                    )
                except Exception:
                    pass
            return msg
        else:
            return f"❌ str_replace failed: {result.get('error', 'Unknown error')}"
    except Exception:
        logger.exception(
            "Tool replace_in_file_tool: Error replacing in file",
            file_path=input_data.file_path,
        )
        return "❌ Error replacing in file"


def insert_lines_tool(input_data: InsertLinesInput) -> str:
    """Insert content at a specific line in a file"""
    position = "after" if input_data.insert_after else "before"
    logger.info(
        f"🔧 [Tool Call] insert_lines_tool: Inserting lines {position} line {input_data.line_number} "
        f"in '{input_data.file_path}' (project_id={input_data.project_id})"
    )

    # LOCAL-FIRST: Try local execution first
    local_result = _execute_local_write(
        "insert_lines",
        {
            "file_path": input_data.file_path,
            "line_number": input_data.line_number,
            "content": input_data.content,
            "description": input_data.description,
            "insert_after": input_data.insert_after,
            "project_id": input_data.project_id,
        },
        input_data.file_path,
    )

    if local_result is not None:
        # Local execution was attempted - return result (success or error)
        return local_result

    # No tunnel available - use cloud CodeChangesManager (for web UI users)
    try:
        manager = _get_code_changes_manager()
        db = None
        if input_data.project_id:
            logger.info(
                f"Tool insert_lines_tool: Project ID provided ({input_data.project_id}), fetching database session"
            )
            from app.core.database import get_db

            db = next(get_db())
            logger.debug("Tool insert_lines_tool: Database session obtained")
        # project_id is now required, so this shouldn't happen, but keep for safety
        if not input_data.project_id:
            logger.error(
                "Tool insert_lines_tool: ERROR - project_id is required but was not provided!"
            )
            return "❌ Error: project_id is required to insert lines. Please provide the project_id from the conversation context."
        result = manager.insert_lines(
            file_path=input_data.file_path,
            line_number=input_data.line_number,
            content=input_data.content,
            description=input_data.description,
            insert_after=input_data.insert_after,
            project_id=input_data.project_id,
            db=db,
        )

        if result.get("success"):
            position = "after" if result["position"] == "after" else "before"
            context_str = ""
            if result.get("inserted_context"):
                context_start = result.get("context_start_line", input_data.line_number)
                context_end = result.get(
                    "context_end_line",
                    input_data.line_number + result["lines_inserted"],
                )
                context_str = f"\n\nInserted lines with context (lines {context_start}-{context_end}):\n```{input_data.file_path}\n{result['inserted_context']}\n```"
            return f"✅ Inserted {result['lines_inserted']} line(s) {position} line {input_data.line_number} in '{input_data.file_path}'{context_str}"
        else:
            return f"❌ Error inserting lines: {result.get('error', 'Unknown error')}"
    except Exception:
        logger.exception(
            "Tool insert_lines_tool: Error inserting lines",
            file_path=input_data.file_path,
            line_number=input_data.line_number,
        )
        return "❌ Error inserting lines"


def delete_lines_tool(input_data: DeleteLinesInput) -> str:
    """Delete specific lines from a file"""
    logger.info(
        f"Tool delete_lines_tool: Deleting lines {input_data.start_line}-{input_data.end_line or input_data.start_line} "
        f"from '{input_data.file_path}' (project_id={input_data.project_id})"
    )

    # LOCAL-FIRST: Try local execution first
    local_result = _execute_local_write(
        "delete_lines",
        {
            "file_path": input_data.file_path,
            "start_line": input_data.start_line,
            "end_line": input_data.end_line,
            "description": input_data.description,
            "project_id": input_data.project_id,
        },
        input_data.file_path,
    )

    if local_result is not None:
        # Local execution was attempted - return result (success or error)
        return local_result

    # No tunnel available - use cloud CodeChangesManager (for web UI users)
    try:
        manager = _get_code_changes_manager()
        db = None
        if input_data.project_id:
            logger.info(
                f"Tool delete_lines_tool: Project ID provided ({input_data.project_id}), fetching database session"
            )
            from app.core.database import get_db

            db = next(get_db())
            logger.debug("Tool delete_lines_tool: Database session obtained")
        result = manager.delete_lines(
            file_path=input_data.file_path,
            start_line=input_data.start_line,
            end_line=input_data.end_line,
            description=input_data.description,
            project_id=input_data.project_id,
            db=db,
        )

        if result.get("success"):
            deleted_preview = result["deleted_content"][:200]
            return (
                f"✅ Deleted lines {result['start_line']}-{result['end_line']} from '{input_data.file_path}'\n\n"
                + f"Deleted {result['lines_deleted']} line(s)\n"
                + f"Deleted content:\n```\n{deleted_preview}{'...' if len(result['deleted_content']) > 200 else ''}\n```"
            )
        else:
            return f"❌ Error deleting lines: {result.get('error', 'Unknown error')}"
    except Exception:
        logger.exception(
            "Tool delete_lines_tool: Error deleting lines",
            file_path=input_data.file_path,
        )
        return "❌ Error deleting lines"


def show_updated_file_tool(input_data: ShowUpdatedFileInput) -> str:
    """
    Display the complete updated content of one or more files. This tool streams the full file content
    directly into the agent response without going through the LLM, allowing users to see
    the complete edited files. Use this when the user asks to see the updated file content.

    Args:
        input_data: ShowUpdatedFileInput with optional file_paths list
            - file_paths: List of file paths (e.g., ['src/main.py', 'src/utils.py'])
            - If file_paths is None or empty, shows ALL changed files
            - If a single file path string is provided, it will be converted to a list
    """
    logger.info(
        f"Tool show_updated_file_tool: Showing updated content for '{input_data.file_paths or 'all files'}'"
    )

    # Check if we should route to LocalServer (for single file)
    if input_data.file_paths and len(input_data.file_paths) == 1:
        if _should_route_to_local_server():
            logger.info(f"🔧 [Tool Call] Routing show_updated_file_tool to LocalServer")
            result = _route_to_local_server(
                "show_updated_file",
                {
                    "file_path": input_data.file_paths[0],
                },
            )
            if result:
                return result

    # Fall back to CodeChangesManager
    try:
        manager = _get_code_changes_manager()
        summary = manager.get_summary()

        if summary["total_files"] == 0:
            return "📋 **No files to display**\n\nNo files have been modified yet."

        # Determine which files to show
        if input_data.file_paths:
            files_to_show = input_data.file_paths
        else:
            # Show all files
            files_to_show = [f["file_path"] for f in summary["files"]]

        if not files_to_show:
            return "📋 **No files to display**\n\nNo matching files found."

        change_emoji = {"add": "➕", "update": "✏️", "delete": "🗑️"}
        result = "\n\n---\n\n## 📝 **Updated Files**\n\n"

        if len(files_to_show) > 1:
            result += f"Showing {len(files_to_show)} files:\n\n"

        # Display each file
        for file_path in files_to_show:
            file_data = manager.get_file(file_path)

            if not file_data:
                result += f"❌ File '{file_path}' not found in changes\n\n"
                continue

            if file_data["change_type"] == "delete":
                result += f"⚠️ **{file_path}** - marked for deletion\n\n"
                continue

            content = file_data.get("content")
            if not content:
                result += f"❌ No content found for '{file_path}'\n\n"
                continue

            # Format the result with markdown code block
            change_type = file_data["change_type"]
            emoji = change_emoji.get(change_type, "📄")

            result += f"{emoji} **Updated File: {file_path}** ({change_type})\n\n"
            result += f"```\n{content}\n```\n\n"

            logger.info(
                f"Tool show_updated_file_tool: Successfully formatted file content for '{file_path}' "
                f"({len(content)} chars)"
            )

        result += "---\n\n"
        logger.info(
            f"Tool show_updated_file_tool: Successfully displayed {len(files_to_show)} file(s)"
        )

        return result
    except Exception:
        logger.exception(
            "Tool show_updated_file_tool: Error showing updated files",
        )
        return "❌ Error showing updated files"


def show_diff_tool(input_data: ShowDiffInput) -> str:
    """
    Display unified diffs showing changes between managed code and the actual codebase.
    This tool streams the formatted diffs directly into the agent response without going through
    the LLM, allowing users to see exactly what was changed. Use this at the end of your response
    to show all the code changes you've made. The content is automatically shown to the user
    without consuming LLM context.
    """
    if _get_local_mode():
        return (
            "❌ **show_diff is not available in local mode.** "
            "The VSCode Extension handles diff display directly. Use get_file_diff per file to verify changes."
        )

    # Resolve project_id: use provided value or look up from conversation_id (non-local mode only)
    project_id = input_data.project_id
    if not project_id and not _get_local_mode():
        conversation_id = _get_conversation_id()
        project_id = _get_project_id_from_conversation_id(conversation_id)
        if project_id:
            logger.info(
                f"Tool show_diff_tool: Resolved project_id={project_id} from conversation_id={conversation_id}"
            )

    logger.info(
        f"Tool show_diff_tool: Displaying diff(s) (file_path: {input_data.file_path}, context_lines: {input_data.context_lines}, project_id: {project_id})"
    )
    try:
        manager = _get_code_changes_manager()
        summary = manager.get_summary()

        if summary["total_files"] == 0:
            return (
                "📋 **No code changes to display**\n\nNo files have been modified yet."
            )

        # Get database session if project_id provided
        db = None
        if project_id:
            logger.info(
                f"Tool show_diff_tool: Project ID available ({project_id}), fetching database session"
            )
            from app.core.database import get_db

            db = next(get_db())

        # Generate git-style diffs for each file
        files_to_diff = (
            [input_data.file_path]
            if input_data.file_path
            else list(manager.changes.keys())
        )
        git_diffs = []

        for file_path in files_to_diff:
            if file_path not in manager.changes:
                continue

            change = manager.changes[file_path]

            # Get old content
            if change.change_type == ChangeType.DELETE:
                old_content = change.previous_content or ""
                new_content = ""
            elif change.change_type == ChangeType.ADD:
                old_content = ""
                new_content = change.content or ""
            else:  # UPDATE
                new_content = change.content or ""
                if change.previous_content is not None:
                    old_content = change.previous_content
                else:
                    # Try to get from repository first if project_id/db provided
                    old_content = None
                    if project_id and db:
                        try:
                            from app.modules.code_provider.code_provider_service import (
                                CodeProviderService,
                            )
                            from app.modules.code_provider.git_safe import (
                                safe_git_operation,
                                GitOperationError,
                            )
                            from app.modules.projects.projects_model import Project

                            project = (
                                db.query(Project)
                                .filter(Project.id == project_id)
                                .first()
                            )
                            if project:
                                cp_service = CodeProviderService(db)

                                def _fetch_old_content():
                                    return cp_service.get_file_content(
                                        repo_name=project.repo_name,
                                        file_path=file_path,
                                        branch_name=project.branch_name,
                                        start_line=None,
                                        end_line=None,
                                        project_id=project_id,
                                        commit_id=project.commit_id,
                                    )

                                try:
                                    # Use timeout to prevent blocking worker
                                    repo_content = safe_git_operation(
                                        _fetch_old_content,
                                        max_retries=1,
                                        timeout=20.0,
                                        max_total_timeout=25.0,
                                        operation_name=f"show_diff_get_old_content({file_path})",
                                    )
                                except GitOperationError as git_error:
                                    logger.warning(
                                        f"Tool show_diff_tool: Git operation timed out: {git_error}"
                                    )
                                    repo_content = None

                                if repo_content:
                                    old_content = repo_content
                        except Exception as e:
                            logger.warning(
                                f"Tool show_diff_tool: Error fetching from repository: {e}"
                            )
                            old_content = None

                    # Fallback to filesystem
                    if old_content is None:
                        old_content = read_file_from_codebase(file_path)

                    # If file doesn't exist, treat as new file
                    if old_content is None or old_content == "":
                        old_content = ""

            # Generate git-style diff
            git_diff = generate_git_diff_patch(
                file_path=file_path,
                old_content=old_content or "",
                new_content=new_content or "",
                context_lines=input_data.context_lines,
            )

            if git_diff:
                git_diffs.append(git_diff)

        if not git_diffs:
            return "📋 **No diffs to display**\n\nNo changes found."

        # Combine all diffs into a single string
        combined_diff = "\n".join(git_diffs)

        # Write diff to .data folder as JSON
        try:
            data_dir = ".data"
            os.makedirs(data_dir, exist_ok=True)

            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"diff_{timestamp}_{uuid.uuid4().hex[:8]}.json"
            filepath = os.path.join(data_dir, filename)

            # Get reasoning hash from reasoning manager
            reasoning_hash = None
            try:
                from app.modules.intelligence.tools.reasoning_manager import (
                    _get_reasoning_manager,
                )

                reasoning_manager = _get_reasoning_manager()
                reasoning_hash = reasoning_manager.get_reasoning_hash()
                # If not finalized yet, try to finalize it
                if not reasoning_hash:
                    reasoning_hash = reasoning_manager.finalize_and_save()
            except Exception as e:
                logger.warning(
                    f"Tool show_diff_tool: Failed to get reasoning hash: {e}"
                )

            # Create JSON with model_patch and reasoning_hash fields
            diff_data = {"model_patch": combined_diff}
            if reasoning_hash:
                diff_data["reasoning_hash"] = reasoning_hash

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(diff_data, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Tool show_diff_tool: Diff written to {filepath} "
                f"(reasoning_hash: {reasoning_hash})"
            )
        except Exception as e:
            logger.warning(
                f"Tool show_diff_tool: Failed to write diff to .data folder: {e}"
            )

        # Output clean diff format
        result = "--generated diff--\n\n"
        result += "```\n"
        result += combined_diff
        result += "\n```\n\n--generated diff--\n"

        # Non-local mode: remind agent to apply changes to worktree
        if project_id:
            result += (
                "\n**Next step (non-local mode):** Call `apply_changes(project_id, conversation_id)` "
                "to write these changes to the worktree. Then ask the user if they want to create a PR.\n"
            )

        return result
    except Exception:
        logger.exception(
            "Tool show_diff_tool: Error displaying diff", project_id=project_id
        )
        return "❌ Error displaying diff"


def get_file_diff_tool(input_data: GetFileDiffInput) -> str:
    """
    Get the diff for a specific file against the repository branch.
    This shows what has changed in this file compared to the original repository version.
    """
    logger.info(
        f"Tool get_file_diff_tool: Getting diff for '{input_data.file_path}' (context_lines: {input_data.context_lines}, project_id: {input_data.project_id})"
    )
    try:
        # In local mode, sync file from LocalServer to Redis so diff reflects current local content
        if _should_route_to_local_server():
            _sync_file_from_local_server_to_redis(input_data.file_path)
        manager = _get_code_changes_manager()
        file_data = manager.get_file(input_data.file_path)

        # Local mode: if file is not in the manager, fetch from LocalServer and build diff directly
        if not file_data and _should_route_to_local_server():
            local_content = _fetch_file_content_from_local_server(input_data.file_path)
            if local_content is None:
                return (
                    f"❌ Could not read '{input_data.file_path}' from local workspace. "
                    "The file may not exist or the VS Code extension tunnel may be disconnected."
                )
            db = None
            if input_data.project_id:
                from app.core.database import get_db

                db = next(get_db())
            old_content = ""
            if input_data.project_id and db:
                repo_content = fetch_repo_file_content_for_diff(
                    input_data.project_id, input_data.file_path, db
                )
                if repo_content is not None:
                    old_content = repo_content
            if old_content:
                diff_content = create_unified_diff(
                    old_content,
                    local_content,
                    input_data.file_path,
                    input_data.file_path,
                    input_data.context_lines,
                )
            else:
                diff_content = create_unified_diff(
                    "",
                    local_content,
                    "/dev/null",
                    input_data.file_path,
                    input_data.context_lines,
                )
            result = (
                f"📝 **Diff for {input_data.file_path}** (✏️ local file vs repo)\n\n"
                f"**Source:** Local workspace (file not in session changes)\n\n"
                "```diff\n"
            )
            result += diff_content
            result += "\n```\n"
            return result

        if not file_data:
            return f"❌ File '{input_data.file_path}' not found in changes"

        # Get database session if project_id provided
        db = None
        if input_data.project_id:
            logger.info(
                f"Tool get_file_diff_tool: Project ID provided ({input_data.project_id}), fetching database session"
            )
            from app.core.database import get_db

            db = next(get_db())

        # Generate diff for this specific file
        diffs = manager.generate_diff(
            file_path=input_data.file_path,
            context_lines=input_data.context_lines,
            project_id=input_data.project_id,
            db=db,
        )

        if not diffs or input_data.file_path not in diffs:
            return f"❌ No diff generated for '{input_data.file_path}'"

        diff_content = diffs[input_data.file_path]
        change_emoji = {"add": "➕", "update": "✏️", "delete": "🗑️"}
        emoji = change_emoji.get(file_data["change_type"], "📄")

        result = f"📝 **Diff for {input_data.file_path}** ({emoji} {file_data['change_type']})\n\n"
        if file_data.get("description"):
            result += f"*{file_data['description']}*\n\n"
        result += f"**Last updated:** {file_data['updated_at']}\n\n"
        result += "```diff\n"
        result += diff_content
        result += "\n```\n"

        return result
    except Exception:
        logger.exception(
            "Tool get_file_diff_tool: Error getting file diff",
            project_id=input_data.project_id,
            file_path=input_data.file_path,
        )
        return "❌ Error getting file diff"


def get_comprehensive_metadata_tool(input_data: GetComprehensiveMetadataInput) -> str:
    """
    Get comprehensive metadata about all code changes in the current session.
    This shows the complete state of all files being managed, including timestamps,
    descriptions, change types, and line counts. Use this to review your session progress
    and understand what files have been modified.
    """
    logger.info("Tool get_comprehensive_metadata_tool: Getting comprehensive metadata")
    try:
        manager = _get_code_changes_manager()
        summary = manager.get_summary()

        cid = summary.get("conversation_id") or "(ephemeral)"
        result = (
            f"📊 **Complete Session State** (Conversation: {cid})\n\n"
        )
        result += f"**Total Files Changed:** {summary['total_files']}\n\n"

        change_emoji = {"add": "➕", "update": "✏️", "delete": "🗑️"}

        # Summary by change type
        result += "**Summary by Change Type:**\n"
        for change_type, count in summary["change_counts"].items():
            if count > 0:
                emoji = change_emoji.get(change_type, "📄")
                result += f"- {emoji} {change_type.title()}: {count}\n"
        result += "\n"

        # Detailed file information
        if summary["files"]:
            result += "**Detailed File Information:**\n\n"
            for file_info in summary["files"]:
                emoji = change_emoji.get(file_info["change_type"], "📄")
                result += f"{emoji} **{file_info['file_path']}**\n"
                result += f"  - Type: {file_info['change_type']}\n"
                result += f"  - Last Updated: {file_info['updated_at']}\n"
                if file_info.get("description"):
                    result += f"  - Description: {file_info['description']}\n"

                # Get file data for line counts
                file_data = manager.get_file(file_info["file_path"])
                if file_data:
                    if file_data["change_type"] == "delete":
                        if file_data.get("previous_content"):
                            lines = file_data["previous_content"].split("\n")
                            result += f"  - Original Lines: {len(lines)}\n"
                    else:
                        if file_data.get("content"):
                            lines = file_data["content"].split("\n")
                            result += f"  - Current Lines: {len(lines)}\n"
                        if file_data.get("previous_content"):
                            prev_lines = file_data["previous_content"].split("\n")
                            result += f"  - Original Lines: {len(prev_lines)}\n"
                result += "\n"
        else:
            result += "No files have been modified yet.\n"

        result += "\n💡 **Tip:** Use `get_file_from_changes` to see detailed information about a specific file, "
        result += "or `get_file_diff` to see the diff for a file against the repository branch."

        return result
    except Exception:
        logger.exception(
            "Tool get_comprehensive_metadata_tool: Error getting metadata",
            project_id=input_data.project_id,
        )
        return "❌ Error getting metadata"


def export_changes_tool(input_data: ExportChangesInput) -> str:
    """Export all changes in the specified format"""
    logger.info(
        f"Tool export_changes_tool: Exporting changes in '{input_data.format}' format"
    )
    try:
        manager = _get_code_changes_manager()

        # Get database session if project_id provided (needed for diff format)
        db = None
        if input_data.project_id:
            logger.info(
                f"Tool export_changes_tool: Project ID provided ({input_data.project_id}), fetching database session"
            )
            from app.core.database import get_db

            db = next(get_db())

        exported = manager.export_changes(
            format=input_data.format, project_id=input_data.project_id, db=db
        )

        if input_data.format == "json":
            # Return JSON directly (might be long, but that's expected)
            return f"📦 **Exported Changes (JSON)**\n\n```json\n{exported}\n```"
        elif input_data.format == "dict":
            if not isinstance(exported, dict):
                return f"❌ Expected dict format, got {type(exported)}"
            result = f"📦 **Exported Changes (Dictionary)** - {len(exported)} files\n\n"
            items_list = list(exported.items())[:5]  # Show first 5
            for file_path, content in items_list:
                result += f"**{file_path}** ({len(content)} chars):\n```\n{content[:200]}...\n```\n\n"
            if len(exported) > 5:
                result += f"... and {len(exported) - 5} more files\n"
            return result
        elif input_data.format == "diff":
            # Return diff patch format with "Generated Diff:" heading
            if not exported or not isinstance(exported, str):
                return "❌ No diff generated or invalid format"
            return f"Generated Diff:\n\n```\n{exported}\n```"
        else:  # list format
            if not isinstance(exported, list):
                return f"❌ Expected list format, got {type(exported)}"
            result = f"📦 **Exported Changes (List)** - {len(exported)} files\n\n"
            for change in exported[:5]:  # Show first 5
                if isinstance(change, dict):
                    result += f"**{change.get('file_path', 'unknown')}** ({change.get('change_type', 'unknown')})\n"
            if len(exported) > 5:
                result += f"... and {len(exported) - 5} more files\n"
            return result
    except Exception:
        logger.exception(
            "Tool export_changes_tool: Error exporting changes",
            format=input_data.format,
        )
        return "❌ Error exporting changes"


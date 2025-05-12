import asyncio
import difflib
import re
import traceback
from typing import Dict, Any, List, Tuple, Optional, Union
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from io import StringIO


class GeneratePatchDiffInput(BaseModel):
    project_id: str = Field(description="Project ID to fetch the file from.")
    file_path: str = Field(description="Path to the file to generate a patch for.")
    changes: List[Dict[str, Any]] = Field(
        description="""List of changes to apply to the file. Each change must include:
        - 'type': One of 'add', 'remove', or 'replace'
        - 'start_line': Line number where the change begins (1-indexed, must exist in file)
        - 'context_before': String that must EXACTLY match the line(s) before the change (for location verification)
        - 'context_after': String that must EXACTLY match the line(s) after the change (for location verification)
        - 'context_lines': Number of unchanged lines to include before and after the change (default: 3)
        
        For 'add' operations:
        - 'content': The new line(s) to add (must preserve correct indentation)
        
        For 'remove' operations:
        - 'count': Number of lines to remove (starting from start_line)
        
        For 'replace' operations:
        - 'old_content': The original line(s) to be replaced (must match exactly for verification)
        - 'new_content': The new line(s) to replace with (must preserve correct indentation)
        """
    )


class GeneratePatchDiff:
    """
    Tool to generate correctly formatted unified diff patches for file changes.
    Uses a direct string manipulation approach for precise control over patch format.
    """

    name = "GeneratePatchDiff"
    description = """Generates a properly formatted unified diff/patch for specified changes to a file.
    
    ⚠️ KEY ADVANTAGES ⚠️
    - Creates patches in the same format as git diff
    - Ensures correct file headers, hunk headers, and line prefixes
    - Handles context lines correctly to ensure patches apply cleanly
    - Provides precise error messages for debugging
    
    ⚠️ IMPORTANT GUIDELINES FOR RELIABLE PATCH GENERATION ⚠️
    - Always specify start_line as accurately as possible (1-indexed line number)
    - Always provide context_before or context_after to ensure correct positioning
    - Ensure all indentation in new content matches the file's style precisely
    - Make sure old_content matches the exact content in the file (whitespace matters!)
    
    INPUT EXAMPLE:
    ```python
    {
        "project_id": "my-project",
        "file_path": "src/main.py",
        "changes": [
            {
                "type": "add",
                "start_line": 50,  # Line number where to add content (1-indexed)
                "content": "    def new_method(self):\\n        return True",  # Preserve indentation!
                "context_before": "    def existing_method(self):",  # Line before the addition
                "context_lines": 3  # Number of unchanged lines to include for context
            },
            {
                "type": "remove",
                "start_line": 75,
                "count": 2,  # Number of lines to remove
                "context_after": "    def another_method(self):"  # Line after removal
            },
            {
                "type": "replace",
                "start_line": 100,
                "old_content": "    return old_value",  # Exact content to replace
                "new_content": "    return new_value",  # New content with matching indentation
                "context_lines": 3
            }
        ]
    }
    ```
    
    ⚠️ TROUBLESHOOTING TIPS ⚠️
    1. If your patch fails to apply, check for:
       - Whitespace discrepancies (tabs vs spaces, trailing whitespace)
       - Line ending differences (\\r\\n vs \\n)
       - Content that doesn't exactly match the file
    2. If location can't be found, try:
       - Providing more unique context lines
       - Using both context_before AND context_after
       - Specifying exact start_line
    
    The tool will return a detailed error message if it encounters any issues, helping you debug and fix the patch.
    """

    def __init__(self, fetchfiletool: StructuredTool):
        self.fetchfiletool = fetchfiletool

    def find_change_location(
        self,
        file_lines: List[str],
        start_line: int,
        context_before: Optional[str] = None,
        context_after: Optional[str] = None,
        change_idx: int = 0,
    ) -> Tuple[int, Optional[str]]:
        """
        Finds the exact location for a change using line number and context lines.
        Returns (actual_line_idx, error_message_if_any)
        """
        # Convert to 0-indexed
        start_idx = max(0, start_line - 1)

        # Basic bounds check
        if start_idx >= len(file_lines):
            return (
                -1,
                f"Error in change #{change_idx+1}: Start line {start_line} exceeds file length ({len(file_lines)} lines)",
            )

        # If no context provided, use the line number directly
        if not context_before and not context_after:
            return start_idx, None

        # Clean context lines (remove trailing newlines but preserve other whitespace)
        clean_context_before = context_before.rstrip("\r\n") if context_before else None
        clean_context_after = context_after.rstrip("\r\n") if context_after else None

        # Check exact position first
        exact_match = True

        if clean_context_before and start_idx > 0:
            if file_lines[start_idx - 1].rstrip("\r\n") != clean_context_before:
                exact_match = False

        if exact_match and clean_context_after and start_idx < len(file_lines):
            if file_lines[start_idx].rstrip("\r\n") != clean_context_after:
                exact_match = False

        if exact_match:
            return start_idx, None

        # If exact match failed, search in a larger window
        search_range = 20
        start_search = max(0, start_idx - search_range)
        end_search = min(len(file_lines), start_idx + search_range)

        # Approach 1: Look for both context_before and context_after in sequence
        if clean_context_before and clean_context_after:
            for i in range(start_search, end_search):
                if i > 0 and i < len(file_lines):
                    if (
                        file_lines[i - 1].rstrip("\r\n") == clean_context_before
                        and file_lines[i].rstrip("\r\n") == clean_context_after
                    ):
                        return i, None

        # Approach 2: Find best context_before match
        elif clean_context_before:
            for i in range(start_search, end_search):
                if i > 0 and file_lines[i - 1].rstrip("\r\n") == clean_context_before:
                    return i, None

        # Approach 3: Find best context_after match
        elif clean_context_after:
            for i in range(start_search, end_search):
                if (
                    i < len(file_lines)
                    and file_lines[i].rstrip("\r\n") == clean_context_after
                ):
                    return i, None

        # If we get here, we couldn't find a good match
        error_details = []
        if clean_context_before:
            error_details.append(
                f"Context before: '{clean_context_before[:40] + ('...' if len(clean_context_before) > 40 else '')}'"
            )
        if clean_context_after:
            error_details.append(
                f"Context after: '{clean_context_after[:40] + ('...' if len(clean_context_after) > 40 else '')}'"
            )

        return -1, (
            f"Error in change #{change_idx+1}: Could not find a matching position near line {start_line}.\n"
            f"Searched {search_range} lines before and after, but couldn't find: {', '.join(error_details)}\n"
            f"Check for exact whitespace/indentation matches and line endings."
        )

    def verify_content_match(
        self, file_lines: List[str], start_idx: int, content: str, change_idx: int
    ) -> Optional[str]:
        """
        Verify content matches file exactly, with character-by-character comparison.
        Returns error message if mismatch, None if match.
        """
        content_lines = content.rstrip("\r\n").split("\n")

        # Check bounds
        if start_idx + len(content_lines) > len(file_lines):
            return (
                f"Error in change #{change_idx+1}: Not enough lines in file starting at line {start_idx+1}. "
                f"Need {len(content_lines)} lines but only {len(file_lines) - start_idx} remain."
            )

        # Compare each line exactly
        for i, content_line in enumerate(content_lines):
            file_line = file_lines[start_idx + i].rstrip("\r\n")
            clean_content = content_line.rstrip("\r\n")

            if file_line != clean_content:
                # Show detailed diff for debugging
                return (
                    f"Error in change #{change_idx+1}: Content mismatch at line {start_idx+i+1}.\n"
                    f"  Expected: '{repr(clean_content)}'\n"
                    f"  Found in file: '{repr(file_line)}'\n"
                    f"  Check for spaces, tabs, and all whitespace characters."
                )

        return None

    def manually_generate_patch(
        self, file_path: str, file_content: str, changes: List[Dict[str, Any]]
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Generate a unified diff patch manually with precise control over format.
        Uses direct string manipulation for maximum control over patch format.
        Returns (success, error_message, patch_content)
        """
        # Split file content into lines without losing line endings
        file_lines_with_endings = file_content.splitlines(True)

        # Also create a version without line endings for easier comparison
        file_lines = [line.rstrip("\r\n") for line in file_lines_with_endings]

        # Create a working copy to track changes
        working_lines = file_lines.copy()

        # Create patch output buffer
        patch_output = StringIO()

        # Write file headers in git format
        patch_output.write(f"diff --git a/{file_path} b/{file_path}\n")
        patch_output.write(f"--- a/{file_path}\n")
        patch_output.write(f"+++ b/{file_path}\n")

        # Sort changes by line number to process from top to bottom
        sorted_changes = sorted(changes, key=lambda c: c.get("start_line", 0))

        # Validate basic change parameters first
        validation_errors = []
        for i, change in enumerate(sorted_changes):
            change_type = change.get("type")
            if not change_type or change_type not in ["add", "remove", "replace"]:
                validation_errors.append(
                    f"Error in change #{i+1}: Invalid change type '{change_type}'"
                )
                continue

            start_line = change.get("start_line")
            if not isinstance(start_line, int) or start_line < 1:
                validation_errors.append(
                    f"Error in change #{i+1}: Invalid start_line: {start_line}"
                )
                continue

            # Check type-specific required fields
            if change_type == "add" and "content" not in change:
                validation_errors.append(
                    f"Error in change #{i+1}: Missing 'content' for add operation"
                )
            elif change_type == "remove" and "count" not in change:
                validation_errors.append(
                    f"Error in change #{i+1}: Missing 'count' for remove operation"
                )
            elif change_type == "replace":
                if "old_content" not in change:
                    validation_errors.append(
                        f"Error in change #{i+1}: Missing 'old_content' for replace operation"
                    )
                if "new_content" not in change:
                    validation_errors.append(
                        f"Error in change #{i+1}: Missing 'new_content' for replace operation"
                    )

        if validation_errors:
            return False, "\n".join(validation_errors), None

        # Process each change and generate the corresponding hunk
        line_offset = 0  # Track line offsets from previous changes

        for change_idx, change in enumerate(sorted_changes):
            change_type = change.get("type")
            start_line = change.get("start_line", 0)
            context_lines = change.get("context_lines", 3)
            context_before = change.get("context_before")
            context_after = change.get("context_after")

            # Adjust start line based on previous changes
            adjusted_start_line = start_line + line_offset

            # Find the real position using context if provided
            position, error = self.find_change_location(
                working_lines,
                adjusted_start_line,
                context_before,
                context_after,
                change_idx,
            )

            if position < 0:
                return (
                    False,
                    error or f"Failed to find location for change #{change_idx+1}",
                    None,
                )

            actual_start_idx = position

            # For replace operations, verify old content matches
            if change_type == "replace":
                old_content = change.get("old_content", "")
                error = self.verify_content_match(
                    working_lines, actual_start_idx, old_content, change_idx
                )

                if error:
                    return False, error, None

            # Calculate hunk boundaries with proper context lines
            hunk_start = max(0, actual_start_idx - context_lines)

            if change_type == "add":
                content = change.get("content", "")
                new_lines = content.split("\n")

                # Calculate hunk end and line counts for header
                hunk_end = min(len(working_lines), actual_start_idx + context_lines)
                original_line_count = hunk_end - hunk_start
                modified_line_count = original_line_count + len(new_lines)

                # Write hunk header with correct line numbers
                patch_output.write(
                    f"@@ -{hunk_start+1},{original_line_count} +{hunk_start+1},{modified_line_count} @@\n"
                )

                # Write context lines before the change (with space prefix)
                for i in range(hunk_start, actual_start_idx):
                    patch_output.write(f" {working_lines[i]}\n")

                # Write added lines (with + prefix)
                for line in new_lines:
                    patch_output.write(f"+{line}\n")

                # Write context lines after the change (with space prefix)
                for i in range(actual_start_idx, hunk_end):
                    patch_output.write(f" {working_lines[i]}\n")

                # Update working lines for future changes
                working_lines[actual_start_idx:actual_start_idx] = new_lines
                line_offset += len(new_lines)

            elif change_type == "remove":
                count = change.get("count", 1)

                # Ensure we're not removing more lines than exist
                if actual_start_idx + count > len(working_lines):
                    return (
                        False,
                        f"Error in change #{change_idx+1}: Cannot remove {count} lines from position {actual_start_idx+1}, "
                        f"only {len(working_lines) - actual_start_idx} lines remain",
                        None,
                    )

                # Calculate hunk end and line counts for header
                hunk_end = min(
                    len(working_lines), actual_start_idx + count + context_lines
                )
                original_line_count = hunk_end - hunk_start
                modified_line_count = original_line_count - count

                # Write hunk header with correct line numbers
                patch_output.write(
                    f"@@ -{hunk_start+1},{original_line_count} +{hunk_start+1},{modified_line_count} @@\n"
                )

                # Write context lines before the change (with space prefix)
                for i in range(hunk_start, actual_start_idx):
                    patch_output.write(f" {working_lines[i]}\n")

                # Write removed lines (with - prefix)
                for i in range(actual_start_idx, actual_start_idx + count):
                    patch_output.write(f"-{working_lines[i]}\n")

                # Write context lines after the change (with space prefix)
                for i in range(actual_start_idx + count, hunk_end):
                    patch_output.write(f" {working_lines[i]}\n")

                # Update working lines for future changes
                del working_lines[actual_start_idx : actual_start_idx + count]
                line_offset -= count

            elif change_type == "replace":
                old_content = change.get("old_content", "")
                new_content = change.get("new_content", "")

                old_lines = old_content.split("\n")
                new_lines = new_content.split("\n")

                # Calculate hunk end and line counts for header
                hunk_end = min(
                    len(working_lines),
                    actual_start_idx + len(old_lines) + context_lines,
                )
                original_line_count = hunk_end - hunk_start
                modified_line_count = (
                    original_line_count - len(old_lines) + len(new_lines)
                )

                # Write hunk header with correct line numbers
                patch_output.write(
                    f"@@ -{hunk_start+1},{original_line_count} +{hunk_start+1},{modified_line_count} @@\n"
                )

                # Write context lines before the change (with space prefix)
                for i in range(hunk_start, actual_start_idx):
                    patch_output.write(f" {working_lines[i]}\n")

                # Write removed lines (with - prefix)
                for i in range(len(old_lines)):
                    patch_output.write(f"-{working_lines[actual_start_idx + i]}\n")

                # Write new lines (with + prefix)
                for line in new_lines:
                    patch_output.write(f"+{line}\n")

                # Write context lines after the change (with space prefix)
                for i in range(actual_start_idx + len(old_lines), hunk_end):
                    patch_output.write(f" {working_lines[i]}\n")

                # Update working lines for future changes
                working_lines[actual_start_idx : actual_start_idx + len(old_lines)] = (
                    new_lines
                )
                line_offset += len(new_lines) - len(old_lines)

        # Ensure the patch ends with a newline
        patch_content = patch_output.getvalue()
        if not patch_content.endswith("\n"):
            patch_content += "\n"

        return True, "", patch_content

    async def arun(
        self, project_id: str, file_path: str, changes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Asynchronous method to generate a patch based on specified changes.
        Uses manual string manipulation for maximum control over patch format.
        """
        try:
            # Fetch file content
            resp = self.fetchfiletool.func(project_id=project_id, file_path=file_path)  # type: ignore

            if not resp or "content" not in resp:
                error_message = (
                    resp.get("error", "Failed to fetch file content (unknown reason)")
                    if isinstance(resp, dict)
                    else "Failed to fetch file content (unexpected response format)"
                )
                return {
                    "success": False,
                    "error": {
                        "stage": "fetch_file",
                        "reason": "Failed to fetch file content",
                        "details": error_message,
                        "file_path": file_path,
                    },
                }

            file_content = resp["content"]

            # Debug print each change
            print("Debug - Changes to be applied:")
            for change in changes:
                change_type = change.get("type", "")
                if change_type == "replace":
                    print(
                        f"Replace:\n-----\nOld: {change.get('old_content', '')}\nNew: {change.get('new_content', '')}\n-----"
                    )
                elif change_type == "add":
                    print(f"Add:\n-----\n{change.get('content', '')}\n-----")
                elif change_type == "remove":
                    print(
                        f"Remove {change.get('count', 0)} lines starting at line {change.get('start_line', 0)}\n-----"
                    )

            # Generate the patch directly with manual string manipulation
            success, error_msg, patch = self.manually_generate_patch(
                file_path, file_content, changes
            )

            if not success:
                return {
                    "success": False,
                    "error": {
                        "stage": "patch_generation",
                        "reason": "Failed to generate patch",
                        "details": error_msg,
                        "file_path": file_path,
                    },
                }

            # Debug output for patch
            print("Debug - Generated patch:")
            print(patch)

            return {
                "success": True,
                "patch": patch,
                "file_path": file_path,
                "message": "Successfully generated patch",
                "changes_count": len(changes),
            }

        except Exception as e:
            # Catch any unexpected errors
            error_details = f"Error generating patch for {file_path}: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            print(f"Debug - Exception: {error_details}")
            return {
                "success": False,
                "error": {
                    "stage": "patch_generation",
                    "reason": "Error during patch generation",
                    "details": error_details,
                    "file_path": file_path,
                },
            }

    def run(
        self, project_id: str, file_path: str, changes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for the arun method.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, submit the coroutine to it
                return asyncio.run_coroutine_threadsafe(
                    self.arun(project_id, file_path, changes), loop
                ).result()
            else:
                # If no loop is running, run the coroutine until complete
                return loop.run_until_complete(
                    self.arun(project_id, file_path, changes)
                )
        except RuntimeError:
            # Handle cases where get_event_loop might fail
            return asyncio.run(self.arun(project_id, file_path, changes))
        except Exception as e:
            # Catch any other unexpected errors in the synchronous wrapper
            error_details = f"Error in run method: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            print(f"Debug - Run Exception: {error_details}")
            return {
                "success": False,
                "error": {
                    "stage": "runtime",
                    "reason": "Unexpected error in synchronous wrapper",
                    "details": error_details,
                    "file_path": file_path,
                },
            }


def generate_patch_diff_tool(fetchfiletool: StructuredTool) -> StructuredTool:
    """
    Returns: StructuredTool for generating diff patches based on specified changes.
    Uses direct string manipulation for precise control over patch format.
    """
    tool = GeneratePatchDiff(fetchfiletool)
    return StructuredTool.from_function(
        coroutine=tool.arun,
        func=tool.run,
        name=tool.name,
        description=tool.description,
        args_schema=GeneratePatchDiffInput,
    )

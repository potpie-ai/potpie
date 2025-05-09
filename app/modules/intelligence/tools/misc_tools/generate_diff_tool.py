import asyncio
import os
import re
import tempfile
import traceback
from typing import Dict, Any, Optional, Tuple, List
from pydantic import BaseModel, Field
from io import StringIO
from langchain_core.tools import StructuredTool
import difflib

# Attempt to import required libraries. Handle the case where they are not installed.
try:
    from unidiff import PatchSet, PatchedFile
    import difflib

    _HAS_UNIDIFF = True
except ImportError:
    _HAS_UNIDIFF = False

    # Define dummy classes if libraries are not available
    class PatchSet:
        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 0

    class PatchedFile:
        def __init__(self, *args, **kwargs):
            pass

    class Hunk:
        def __init__(self, *args, **kwargs):
            pass

    class Line:
        def __init__(self, *args, **kwargs):
            pass


class GeneratePatchDiffInput(BaseModel):
    file_path: str = Field(
        description="Path of the file to patch (used in diff headers)"
    )
    project_id: str = Field(description="Project ID to fetch the file from.")
    new_content: Optional[str] = Field(
        None, description="The complete new file content with changes applied"
    )
    changes: Optional[List[Dict[str, str]]] = Field(
        None, description="List of change descriptions (context, old_code, new_code)"
    )
    context_lines: int = Field(
        3, description="Number of context lines to include in the diff"
    )


class GeneratePatchDiff:
    """
    Tool to generate unified diff patches from file content and change descriptions.
    Uses unidiff for creating properly formatted git-compatible diffs.
    """

    name = "GeneratePatchDiff"
    description = """Generates a properly formatted unified diff patch from original file content and changes.
        Supports two methods:
        1. Full file replacement: provide original file_path and new_content
        2. Targeted changes: provide original file_path and a list of changes (context, old_code, new_code)
        
        The tool handles proper context line generation, line numbers, and diff formatting.
        Returns a valid unified diff that can be applied with git apply or patch commands.
        
        IMPORTANT: This tool relieves AI agents from calculating line numbers or formatting diffs manually.
    
        """

    def __init__(self, fetchfiletool: StructuredTool):
        self.fetchfiletool = fetchfiletool

    @staticmethod
    def basic_diff_validation(diff: str) -> Tuple[bool, str]:
        """Basic validation of generated diff format."""
        if not diff or not isinstance(diff, str):
            return False, "Generated diff is empty or not a string."

        # Check for file headers
        if not re.search(r"^--- .+\n\+\+\+ .+", diff, re.MULTILINE):
            return False, "Missing unified diff file headers (---, +++)."

        # Check for hunk headers
        if not re.search(r"^@@ .+? @@", diff, re.MULTILINE):
            return False, "No hunk headers (starting with @@ ... @@) found."

        return True, "Diff format looks valid."

    @staticmethod
    def generate_patch_from_full_files(
        file_path: str, original_content, new_content: str, context_lines: int = 3
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Generate a unified diff by comparing the entire original and new file contents.

        Args:
            file_path: Path of the file (used in the diff header)
            original_content: The complete original file content
            new_content: The complete new file content with changes applied
            context_lines: Number of context lines to include (default 3)

        Returns:
            Tuple of (success, message, diff_content)
        """
        if not _HAS_UNIDIFF:
            return (
                False,
                "Required library (unidiff) not available for diff generation.",
                None,
            )

        # Create temporary files to write the content
        old_path = None
        new_path = None

        try:
            with tempfile.NamedTemporaryFile(mode="w+", delete=False) as old_file:
                old_file.write(original_content)
                old_path = old_file.name

            with tempfile.NamedTemporaryFile(mode="w+", delete=False) as new_file:
                new_file.write(new_content)
                new_path = new_file.name

            try:
                # Use the diff command line tool to generate the diff
                import subprocess

                result = subprocess.run(
                    ["diff", f"-U{context_lines}", old_path, new_path],
                    capture_output=True,
                    text=True,
                )

                # If return code is 0, files are identical; if 1, differences found; if >1, error
                if result.returncode > 1:  # Error case
                    return False, f"Error generating diff: {result.stderr}", None
                elif result.returncode == 0:  # Files are identical
                    return (
                        False,
                        "Files are identical, no changes to generate diff from.",
                        None,
                    )

                # Process the output to match the desired format
                diff_output = result.stdout

                # Format the diff header to match the expected git diff format
                formatted_diff = f"diff --git a/{file_path} b/{file_path}\n"
                formatted_diff += f"--- a/{file_path}\n"
                formatted_diff += f"+++ b/{file_path}\n"

                # Add the diff content, skipping the initial headers from the diff command
                lines = diff_output.split("\n")
                for i, line in enumerate(lines):
                    if i > 1 and not (line.startswith("---") or line.startswith("+++")):
                        formatted_diff += line + "\n"

                # Validate the generated diff
                valid, msg = GeneratePatchDiff.basic_diff_validation(formatted_diff)
                if not valid:
                    return False, f"Generated invalid diff: {msg}", None

                return (
                    True,
                    "Successfully generated diff from full files.",
                    formatted_diff.rstrip(),
                )

            except Exception as e:
                return (
                    False,
                    f"Error during diff generation: {type(e).__name__}: {e}\n{traceback.format_exc()}",
                    None,
                )

        finally:
            # Clean up the temporary files
            if old_path and os.path.exists(old_path):
                os.unlink(old_path)
            if new_path and os.path.exists(new_path):
                os.unlink(new_path)

    @staticmethod
    def generate_patch_from_changes(
        file_path: str,
        original_content: str,
        changes: List[Dict],
        context_lines: int = 3,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Generate a unified diff based on specific changes to be applied. This works by first
        applying the changes to the original content to get the new content, then using difflib
        to generate the unified diff.

        Args:
            file_path: Path of the file (used in the diff header)
            changes: List of change descriptions, each containing:
                - context: Text pattern to locate the change (function name or unique line)
                - old_code: The code to be replaced
                - new_code: The replacement code
            context_lines: Number of context lines to include

        Returns:
            Tuple of (success, message, diff_content)
        """
        try:
            # Step 1: apply all changes to create the new content
            orig_lines = original_content.splitlines(keepends=True)
            new_lines = orig_lines.copy()

            for change in changes:
                old_code = change.get("old_code", "")
                new_code = change.get("new_code", "")
                context_keyword = change.get("context", "")
                if not old_code:
                    return False, "Each change must have 'old_code' specified.", None

                # Find the index to apply the change (using context if provided)
                old_code_lines = old_code.splitlines()
                found_idx = -1
                # Search for block of lines matching old_code (with context bias if given)
                context_idx = -1
                if context_keyword:
                    for idx, line in enumerate(new_lines):
                        if context_keyword in line:
                            context_idx = idx
                            break
                search_range = (
                    range(context_idx, min(context_idx + 50, len(new_lines)))
                    if context_idx > 0
                    else range(len(new_lines))
                )
                for i in search_range:
                    if [l.strip() for l in new_lines[i : i + len(old_code_lines)]] == [
                        l.strip() for l in old_code_lines
                    ]:
                        found_idx = i
                        break

                if found_idx == -1:
                    return (
                        False,
                        f"Could not locate the change block in file for old_code='{old_code}'.",
                        None,
                    )

                # Replace old_code lines with new_code lines
                new_lines = (
                    new_lines[:found_idx]
                    + [line + "\n" for line in new_code.splitlines()]
                    + new_lines[found_idx + len(old_code_lines) :]
                )

            # Step 2: generate unified diff via difflib
            diff_lines = list(
                difflib.unified_diff(
                    orig_lines,
                    new_lines,
                    fromfile="a/" + file_path,
                    tofile="b/" + file_path,
                    lineterm="",
                    n=context_lines,
                )
            )
            if not diff_lines or all(l.startswith(("---", "+++")) for l in diff_lines):
                return False, "No changes detected; generated diff is empty.", None

            # Add diff --git header
            formatted_diff = f"diff --git a/{file_path} b/{file_path}\n" + "\n".join(
                diff_lines
            )

            # Validate the generated diff
            valid, msg = GeneratePatchDiff.basic_diff_validation(formatted_diff)
            if not valid:
                return False, f"Generated invalid diff: {msg}", None

            return (
                True,
                "Successfully generated diff from specified changes.",
                formatted_diff,
            )
        except Exception as e:
            return (
                False,
                f"Error during targeted diff generation: {type(e).__name__}: {e}\n{traceback.format_exc()}",
                None,
            )

    async def arun(
        self,
        file_path: str,
        project_id: str,
        new_content: Optional[str] = None,
        changes: Optional[List[Dict[str, str]]] = None,
        context_lines: int = 3,
    ) -> Dict[str, Any]:
        """
        Asynchronous method to generate a unified diff patch.
        """
        # Input validation
        if not file_path or not isinstance(file_path, str):
            return {
                "success": False,
                "error": {
                    "reason": "Invalid file path",
                    "details": "File path must be a non-empty string.",
                },
            }

        resp = self.fetchfiletool.func(project_id=project_id, file_path=file_path)  # type: ignore
        original_content: str = resp.get("content")

        if not original_content or not isinstance(original_content, str):
            return {
                "success": False,
                "error": {
                    "reason": "Invalid original content",
                    "details": "Original file content must be a non-empty string.",
                },
            }

        if new_content is None and (
            not changes or not isinstance(changes, list) or len(changes) == 0
        ):
            return {
                "success": False,
                "error": {
                    "reason": "Invalid input parameters",
                    "details": "Either new_content or changes must be provided.",
                },
            }

        # Check for unidiff library availability
        if not _HAS_UNIDIFF:
            return {
                "success": False,
                "error": {
                    "reason": "Missing dependencies",
                    "details": "Required library (unidiff) is not available. Please install it with 'pip install unidiff'.",
                },
            }

        try:
            # Generate diff based on the input parameters
            if new_content is not None:
                success, message, diff = self.generate_patch_from_full_files(
                    file_path, original_content, new_content, context_lines
                )
            elif changes is not None:
                success, message, diff = self.generate_patch_from_changes(
                    file_path, original_content, changes, context_lines
                )
            else:
                # This should not happen due to the validation above, but adding as a safeguard
                return {
                    "success": False,
                    "error": {
                        "reason": "Invalid parameters",
                        "details": "Either new_content or changes must be provided.",
                    },
                }

            if not success or not diff:
                return {
                    "success": False,
                    "error": {"reason": "Diff generation failed", "details": message},
                }

            print(diff)
            return {"success": True, "message": message, "diff": diff}

        except Exception as e:
            # Catch any unexpected errors during the generation process
            return {
                "success": False,
                "error": {
                    "reason": "Unexpected error",
                    "details": f"Error during diff generation: {type(e).__name__}: {e}\n{traceback.format_exc()}",
                },
            }

    def run(
        self,
        file_path: str,
        project_id: str,
        new_content: Optional[str] = None,
        changes: Optional[List[Dict[str, str]]] = None,
        context_lines: int = 3,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for the arun method.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, submit the coroutine to it
                return asyncio.run_coroutine_threadsafe(
                    self.arun(
                        file_path, project_id, new_content, changes, context_lines
                    ),
                    loop,
                ).result()
            else:
                # If no loop is running, run the coroutine until complete
                return loop.run_until_complete(
                    self.arun(
                        file_path, project_id, new_content, changes, context_lines
                    )
                )
        except RuntimeError:
            # Handle cases where get_event_loop might fail
            return asyncio.run(
                self.arun(file_path, project_id, new_content, changes, context_lines)
            )
        except Exception as e:
            # Catch any other unexpected errors in the synchronous wrapper
            return {
                "success": False,
                "error": {
                    "reason": "Runtime error",
                    "details": f"Error in run method: {type(e).__name__}: {e}\n{traceback.format_exc()}",
                },
            }


def generate_patch_diff_tool(
    fetchfiletool: StructuredTool,
) -> Any:  # Type hint as Any to avoid importing StructuredTool
    """
    Returns: StructuredTool for generating unified diff patches.

    """
    tool = GeneratePatchDiff(fetchfiletool)

    return StructuredTool.from_function(
        coroutine=tool.arun,
        func=tool.run,
        name=tool.name,
        description=tool.description,
        args_schema=GeneratePatchDiffInput,
    )

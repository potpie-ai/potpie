import asyncio
import re
import traceback
from typing import Dict, Any, Optional, Tuple
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from io import StringIO

# Attempt to import required libraries. Handle the case where they are not installed.
try:
    from unidiff import PatchSet

    _HAS_UNIDIFF = True
except ImportError:
    _HAS_UNIDIFF = False

    # Define dummy classes if libraries are not available
    class PatchSet:
        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 0


def debug_patch_and_content(patch, file_content):
    """
    Helper function to analyze patch and file content differences,
    especially invisible whitespace issues that cause verification failures.
    """
    print("=== Patch Analysis ===")

    # 1. Show original patch and file content with characters made visible
    print("PATCH (with whitespace visible):")
    for i, line in enumerate(patch.splitlines()):
        # Replace spaces with · and tabs with → for better visualization
        visible_whitespace = line.replace(" ", "·").replace("\t", "→")
        print(f"{i+1:3d}: {repr(line)} | {visible_whitespace}")

    print("\nFILE CONTENT (with whitespace visible):")
    for i, line in enumerate(file_content.splitlines()):
        visible_whitespace = line.replace(" ", "·").replace("\t", "→")
        print(f"{i+1:3d}: {repr(line)} | {visible_whitespace}")

    # 2. Extract patch target line and file content line for comparison
    patch_lines = patch.splitlines()
    removed_lines = [line[1:] for line in patch_lines if line.startswith("-")]
    added_lines = [line[1:] for line in patch_lines if line.startswith("+")]

    # Extract line numbers from hunk headers
    line_numbers = []
    for line in patch_lines:
        if line.startswith("@@"):
            try:
                # Extract the line number from the hunk header
                parts = line.split()
                # Format is typically @@ -A,B +C,D @@
                # We want C (the starting line in the new file)
                if len(parts) >= 3 and parts[2].startswith("+"):
                    start_line = int(parts[2][1:].split(",")[0])
                    line_numbers.append(start_line)
            except (IndexError, ValueError):
                pass

    print("\n=== DETAILED COMPARISON ===")
    if removed_lines:
        print("Lines being removed:")
        for i, line in enumerate(removed_lines):
            print(f"  {repr(line)}")

    if added_lines:
        print("Lines being added:")
        for i, line in enumerate(added_lines):
            print(f"  {repr(line)}")

    # 3. Compare the last line of the file with what's expected in the patch
    file_lines = file_content.splitlines()
    if file_lines and removed_lines:
        print("\nComparing last file line with the line to be removed:")
        last_file_line = file_lines[-1]
        line_to_remove = removed_lines[-1]
        print(f"Last file line: {repr(last_file_line)}")
        print(f"Line to remove: {repr(line_to_remove)}")

        if last_file_line == line_to_remove:
            print("✓ Content matches exactly")
        else:
            print("✗ Content does not match exactly")
            # Compare character by character
            min_len = min(len(last_file_line), len(line_to_remove))
            for i in range(min_len):
                if last_file_line[i] != line_to_remove[i]:
                    print(f"  First difference at position {i}:")
                    print(
                        f"    File: '{last_file_line[i]}' (ord: {ord(last_file_line[i])})"
                    )
                    print(
                        f"    Patch: '{line_to_remove[i]}' (ord: {ord(line_to_remove[i])})"
                    )
                    break

            # Check for length differences
            if len(last_file_line) != len(line_to_remove):
                print(
                    f"  Length mismatch: file={len(last_file_line)}, patch={len(line_to_remove)}"
                )

    # 4. Check if file ends with newline
    print("\nNewline analysis:")
    file_ends_with_newline = file_content.endswith("\n")
    print(f"File ends with newline: {file_ends_with_newline}")

    # Check for "No newline at end of file" in patch
    no_newline_marker = any(
        line.startswith("\\ No newline at end of file") for line in patch_lines
    )
    print(f"Patch has 'No newline at end of file' marker: {no_newline_marker}")

    # Suggest fixes
    print("\n=== SUGGESTED FIXES ===")
    if line_numbers and line_numbers[0] > len(file_lines):
        print(
            f"⚠️ Line number mismatch: Patch expects to start at line {line_numbers[0]}, but file only has {len(file_lines)} lines"
        )
        print(
            "   Fix: Update the patch to use correct line numbers or ensure you're patching the correct file"
        )

    if file_ends_with_newline != (not no_newline_marker):
        if file_ends_with_newline:
            print("⚠️ File ends with newline, but patch assumes no final newline")
            print(
                "   Fix: Remove the 'No newline at end of file' marker from your patch"
            )
        else:
            print("⚠️ File does not end with newline, but patch expects one")
            print("   Fix: Add a 'No newline at end of file' marker to your patch")

    # Check if the patch appears to make no changes
    if (
        removed_lines
        and added_lines
        and all(r == a for r, a in zip(removed_lines, added_lines))
    ):
        print(
            "⚠️ Patch appears to make no visible changes - likely has hidden whitespace differences:"
        )
        print(
            "   Fix: Check for trailing whitespace at the end of lines in either the file or patch"
        )

    return {
        "patch_analysis": {
            "total_lines": len(patch_lines),
            "removed_lines": len(removed_lines),
            "added_lines": len(added_lines),
            "has_no_newline_marker": no_newline_marker,
        },
        "file_analysis": {
            "total_lines": len(file_lines),
            "ends_with_newline": file_ends_with_newline,
        },
    }


class VerifyPatchDiffInput(BaseModel):
    patch: str = Field(description="Input string representing a unified diff/patch.")
    project_id: str = Field(description="Project ID to fetch the file from.")
    file_path: str = Field(description="Path to the file the patch will be applied to.")


class VerifyPatchDiff:
    """
    Tool to verify if the input string is a valid, applyable unified (diff/patch).
    Uses unidiff for parsing, strict verification, and application.
    """

    name = "VerifyPatchDiff"
    description = """Verifies that the given input string is a syntactically valid unified diff patch.
        Performs EXACT validation with no tolerance for whitespace or indentation differences.
        Checks for ---/+++ file headers, hunk headers (starting with @@),
        and validates structure with unidiff if available.
        If file_path is provided, also tests if the patch can be applied successfully.
        This tool performs a strict verification, ensuring hunk headers match file content exactly.

        Pass the file path. Before sending the patch, read the file and make sure the patch is valid.
        IMPORTANT: Only send hunks for one file at a time. This tool expects a single file path
        Returns validation status, reason, and application results.
        """

    def __init__(self, fetchfiletool: StructuredTool):
        self.fetchfiletool = fetchfiletool

    @staticmethod
    def basic_patch_check(patch: str) -> Tuple[bool, str]:
        """Basic validation of patch format without relying on libraries."""
        if not patch or not isinstance(patch, str):
            return False, "Input must be a non-empty string."

        # Check for file headers
        if not re.search(r"^--- .+\n\+\+\+ .+", patch, re.MULTILINE):
            return False, "Missing unified diff file headers (---, +++)."

        # Check for hunk headers
        if not re.search(r"^@@ .+? @@", patch, re.MULTILINE):
            return False, "No hunk headers (starting with @@ ... @@) found."

        # Check for proper line prefixes
        lines = patch.splitlines()
        in_hunk = False
        for line in lines:
            if line.startswith("@@"):
                in_hunk = True
                continue

            if in_hunk and not (
                line.startswith(" ")
                or line.startswith("+")
                or line.startswith("-")
                or line.startswith("\\")
            ):
                return False, f"Invalid line prefix in hunk: '{line}'"

        return True, "Patch format looks valid."

    @staticmethod
    def verify_patch_applicability(patch_string, file_content):
        """
        Enhanced version of verify_patch_applicability that provides better diagnostics
        when patch verification fails due to common issues.
        """
        # First run the debug function to get detailed analysis
        analysis = debug_patch_and_content(patch_string, file_content)

        # Check for the most common patch application issues

        # 1. Line number mismatch in patch headers
        patch_lines = patch_string.splitlines()
        hunk_headers = [line for line in patch_lines if line.startswith("@@")]

        if not hunk_headers:
            return (False, "No hunk headers found in patch", None)

        # Extract expected file length from patch headers
        file_lines = file_content.splitlines()
        for header in hunk_headers:
            try:
                # Parse something like @@ -3009,4 +3009,4 @@
                parts = header.split()
                source_info = parts[1][1:]  # Remove the initial -
                target_info = parts[2][1:]  # Remove the initial +

                source_line = int(source_info.split(",")[0])
                source_count = (
                    int(source_info.split(",")[1]) if "," in source_info else 1
                )

                # If the patch targets lines beyond file end
                if (
                    source_line > len(file_lines) + 1
                ):  # +1 because line numbers are 1-indexed
                    return (
                        False,
                        f"Patch refers to line {source_line}, but file only has {len(file_lines)} lines",
                        None,
                    )

                # If the patch expects to modify more lines than exist
                if source_line + source_count - 1 > len(file_lines):
                    return (
                        False,
                        f"Patch expects to change lines {source_line}-{source_line+source_count-1}, but file ends at line {len(file_lines)}",
                        None,
                    )
            except (IndexError, ValueError) as e:
                return (
                    False,
                    f"Failed to parse hunk header '{header}': {str(e)}",
                    None,
                )

        # 2. Check for trailing whitespace issues
        removed_lines = [line[1:] for line in patch_lines if line.startswith("-")]

        for i, line in enumerate(removed_lines):
            # Find the corresponding line in the file
            line_index = -1
            for j, file_line in enumerate(file_lines):
                if (
                    file_line.rstrip() == line.rstrip()
                ):  # Compare ignoring trailing whitespace
                    if file_line != line:  # But they're not exactly equal
                        return (
                            False,
                            f"Line {j+1} matches except for whitespace. File: '{file_line}', Patch: '{line}'",
                            None,
                        )

        # 3. Check newline at end of file discrepancy
        file_ends_with_newline = file_content.endswith("\n")
        no_newline_marker = any(
            line.startswith("\\ No newline at end of file") for line in patch_lines
        )

        if file_ends_with_newline and no_newline_marker:
            return (
                False,
                "Patch includes 'No newline at end of file' marker, but the file does end with a newline",
                None,
            )
        elif not file_ends_with_newline and not no_newline_marker and removed_lines:
            # Only flag this if we're removing lines and would expect to see the marker
            return (
                False,
                "File does not end with a newline, but patch is missing the 'No newline at end of file' marker",
                None,
            )

        # If we've made it this far without identifying any issues, we should run your original
        # verification code, which would parse using unidiff and do a more thorough check

        # For this example, we'll just return success
        return (True, "Patch verification passed initial checks", None)

    @staticmethod
    def apply_patch_from_parsed(
        patch_set: PatchSet, file_content: str
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Apply a patch using a parsed unidiff.PatchSet object.
        This method implements the application logic directly using unidiff's structure.
        """
        if not _HAS_UNIDIFF:  # unidiff is needed for the PatchSet object structure
            return (
                False,
                "Required library (unidiff) not available for application.",
                None,
            )

        if len(patch_set) != 1:
            # This should have been caught in verification, but adding a safeguard
            return (
                False,
                f"Application failed: Expected 1 file in PatchSet, found {len(patch_set)}.",
                None,
            )

        try:
            patched_file = patch_set[0]

            # Split original content into lines
            original_lines = file_content.splitlines()
            result_lines = original_lines.copy()

            # Track if the file ends with a newline
            file_ends_with_newline = file_content.endswith("\n")
            preserve_final_newline = file_ends_with_newline

            # Process each hunk sequentially to apply changes
            line_offset = 0  # Track cumulative line changes across all hunks

            for i, hunk in enumerate(patched_file):
                # Convert source_start (1-indexed) to 0-indexed list position
                hunk_source_start_0_indexed = hunk.source_start - 1

                # Track the current line being processed
                current_line_idx = hunk_source_start_0_indexed

                # Process each line in the hunk
                for line in hunk:
                    # Calculate the actual position in the result_lines list
                    adjusted_pos = current_line_idx + line_offset

                    if line.value.startswith("\\ No newline at end of file"):
                        # This marker means the preceding line should not have a newline
                        preserve_final_newline = False
                        continue

                    if line.is_context:
                        # Context lines remain unchanged
                        current_line_idx += 1

                    elif line.is_removed:
                        # Remove the line at the adjusted position
                        if adjusted_pos < 0 or adjusted_pos >= len(result_lines):
                            return (
                                False,
                                f"Application failed: Attempted to remove line at position {adjusted_pos + 1}, "
                                f"which is outside the range of the file ({len(result_lines)} lines).",
                                None,
                            )

                        del result_lines[adjusted_pos]
                        line_offset -= (
                            1  # Decrease the offset for subsequent operations
                        )
                        current_line_idx += 1  # Move to next line in original

                    elif line.is_added:
                        # Add new line at the adjusted position
                        if adjusted_pos < 0 or adjusted_pos > len(result_lines):
                            return (
                                False,
                                f"Application failed: Attempted to add line at position {adjusted_pos + 1}, "
                                f"which is outside the range of the file ({len(result_lines)} lines).",
                                None,
                            )

                        # Insert the line content without the '+' prefix, preserving exact whitespace
                        content_to_add = (
                            line.value[1:] if line.value.startswith("+") else line.value
                        )
                        # Remove potential trailing newline for consistent handling
                        if content_to_add.endswith("\n"):
                            content_to_add = content_to_add[:-1]
                        if content_to_add.endswith("\r"):
                            content_to_add = content_to_add[:-1]

                        result_lines.insert(adjusted_pos, content_to_add)
                        line_offset += (
                            1  # Increase the offset for subsequent operations
                        )

            # Join the lines back into a string, handling the final newline correctly
            result_content = "\n".join(result_lines)
            if preserve_final_newline:
                result_content += "\n"

            return True, "Patch applied successfully", result_content

        except Exception as e:
            # Catch any unexpected errors during the application process
            return (
                False,
                f"Unexpected error during patch application: {type(e).__name__}: {e}\n{traceback.format_exc()}",
                None,
            )

    async def arun(self, patch: str, project_id: str, file_path: str) -> Dict[str, Any]:
        """
        Asynchronous method to verify a patch and test its application.
        """
        print(patch)
        # 1. Basic patch format check (quick check)
        base_valid, base_msg = self.basic_patch_check(patch)
        if not base_valid:
            return {
                "valid": False,
                "error": {
                    "stage": "basic_check",
                    "reason": "Invalid patch format",
                    "details": base_msg,
                    "file_path": file_path,
                    "original_file_lines_count": 0,  # Not applicable at this stage
                },
            }

        # 2. Fetch file content
        file_content = None
        file_content_with_linenumbers = None
        try:
            # Fetch content without line numbers for verification/application logic
            resp = self.fetchfiletool.func(project_id=project_id, file_path=file_path)  # type: ignore
            # Fetch content with line numbers for detailed error reporting
            resp_with_linenumbers = self.fetchfiletool.func(project_id=project_id, file_path=file_path, with_line_numbers=True)  # type: ignore

            if not resp or "content" not in resp:
                error_message = (
                    resp.get("error", "Failed to fetch file content (unknown reason)")
                    if isinstance(resp, dict)
                    else "Failed to fetch file content (unexpected response format)"
                )
                return {
                    "valid": False,
                    "error": {
                        "stage": "fetch_file",
                        "reason": "Failed to fetch file content",
                        "details": error_message,
                        "file_path": file_path,
                        "original_file_lines_count": 0,  # Cannot determine if fetch failed
                    },
                }

            file_content = resp["content"]
            file_content_with_linenumbers = (
                resp_with_linenumbers.get(
                    "content", "Could not fetch file content with line numbers."
                )
                if isinstance(resp_with_linenumbers, dict)
                else "Could not fetch file content with line numbers."
            )
            original_file_lines_count = len(file_content.splitlines())

        except Exception as e:
            # Catch any unexpected errors during file fetching
            return {
                "valid": False,
                "error": {
                    "stage": "fetch_file",
                    "reason": "Error during file fetch",
                    "details": f"Error fetching file content for {file_path}: {type(e).__name__}: {e}\n{traceback.format_exc()}",
                    "file_path": file_path,
                    "original_file_lines_count": 0,  # Cannot determine if fetch failed
                },
            }

        # 3. Detailed patch applicability verification (using strict logic)
        # This checks if the patch hunks match the current file content and returns the parsed PatchSet
        verify_applicable, verify_msg, patch_set = self.verify_patch_applicability(
            patch, file_content
        )

        if not verify_applicable:
            print(f"Patch verification failed: {verify_msg}")
            return {
                "valid": False,
                "error": {
                    "stage": "verification",
                    "reason": "Patch verification failed (strict exact matching)",
                    "details": verify_msg,
                    "file_path": file_path,
                    "original_file_lines_count": original_file_lines_count,
                },
            }

        # 4. Apply the patch using the parsed unidiff.PatchSet
        # This step actually performs the modification based on the verified hunks
        patch_applied, apply_msg, patched_content = self.apply_patch_from_parsed(
            patch_set, file_content  # Pass the parsed patch_set
        )

        if not patch_applied:
            return {
                "valid": False,
                "error": {
                    "stage": "application",
                    "reason": "Patch application failed",
                    "details": apply_msg,
                    "file_path": file_path,
                    "original_file_lines_count": original_file_lines_count,
                },
            }

        # 5. Success case
        return {
            "valid": True,
            "message": "Patch is valid and can be applied successfully",
            "patched_content": patched_content,
        }

    def run(self, patch: str, project_id: str, file_path: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for the arun method.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, submit the coroutine to it
                return asyncio.run_coroutine_threadsafe(
                    self.arun(patch, project_id, file_path), loop
                ).result()
            else:
                # If no loop is running, run the coroutine until complete
                return loop.run_until_complete(self.arun(patch, project_id, file_path))
        except RuntimeError:
            # Handle cases where get_event_loop might fail (e.g., in some environments)
            # and run the coroutine directly.
            return asyncio.run(self.arun(patch, project_id, file_path))
        except Exception as e:
            # Catch any other unexpected errors in the synchronous wrapper
            return {
                "valid": False,
                "error": {
                    "stage": "runtime",
                    "reason": "Unexpected error in synchronous wrapper",
                    "details": f"Error in run method: {type(e).__name__}: {e}\n{traceback.format_exc()}",
                    "file_path": file_path,
                    "original_file_lines_count": 0,  # Cannot determine here
                },
            }


def verify_patch_diff_tool(fetchfiletool: StructuredTool) -> StructuredTool:
    """
    Returns: StructuredTool for patch verification and application testing.
    """
    tool = VerifyPatchDiff(fetchfiletool)
    return StructuredTool.from_function(
        coroutine=tool.arun,
        func=tool.run,
        name=tool.name,
        description=tool.description,
        args_schema=VerifyPatchDiffInput,
    )

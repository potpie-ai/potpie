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
    def verify_patch_applicability(
        patch_string: str, file_content: str
    ) -> Tuple[bool, str, Optional[PatchSet]]:
        """
        Detailed check if the patch string can be applied to the provided file content
        by strictly verifying hunk headers and content against the file content.
        EXACT matching with no tolerance for whitespace differences.
        Returns success status, message, and the parsed PatchSet object if successful.
        """
        if not _HAS_UNIDIFF:  # unidiff is needed for parsing and verification
            return (
                False,
                "Required library (unidiff) not available for strict verification.",
                None,
            )

        try:
            # Parse the patch string into a PatchSet object
            patch_set = PatchSet(patch_string)
            print(f"Parsed patch set with {len(patch_set)} files")

        except Exception as e:
            # Handle errors during patch parsing
            return (
                False,
                f"Failed to parse patch string for verification: {type(e).__name__}: {e}\n{traceback.format_exc()}",
                None,
            )

        if len(patch_set) != 1:
            # Ensure the patch applies to exactly one file
            return (
                False,
                f"Verification failed: Expected 1 file in PatchSet, found {len(patch_set)}.",
                None,
            )

        try:
            patched_file = patch_set[0]
            print(f"Processing file: {patched_file.path}")

            # Split original content into lines with exact line endings preserved
            original_lines = file_content.splitlines(keepends=True)
            original_lines_without_endings = file_content.splitlines(keepends=False)
            print(f"Original file has {len(original_lines)} lines")

            # Track if the file ends with a newline
            file_ends_with_newline = file_content.endswith("\n")
            print(f"File ends with newline: {file_ends_with_newline}")

            for i, hunk in enumerate(patched_file):
                # Get the hunk header details
                hunk_source_start = (
                    hunk.source_start
                )  # Line number where the hunk starts (1-indexed)
                hunk_source_length = (
                    hunk.source_length
                )  # Number of context and removed lines

                print(
                    f"Verifying hunk #{i+1}: source_start={hunk_source_start}, source_length={hunk_source_length}"
                )

                # Track the current line in the original file (convert to 0-indexed)
                current_line_idx = hunk_source_start - 1

                # Count the number of lines processed from the hunk
                source_lines_processed = 0

                # Process each line in the hunk
                for line in hunk:
                    # Handle context and removed lines (lines that should exist in the original file)
                    if line.is_context or line.is_removed:
                        # Ensure we don't process more lines than the hunk header specifies
                        if source_lines_processed >= hunk_source_length:
                            return (
                                False,
                                f"Verification failed: Processed {source_lines_processed} lines from hunk #{i+1}, "
                                f"but hunk header specifies only {hunk_source_length} lines.",
                                None,
                            )

                        # Check if current line is within bounds of the file
                        if current_line_idx < 0 or current_line_idx >= len(
                            original_lines
                        ):
                            return (
                                False,
                                f"Verification failed: Line {current_line_idx + 1} is out of bounds for file with {len(original_lines)} lines.",
                                None,
                            )

                        # Get content from the original file at this line
                        file_line = (
                            original_lines_without_endings[current_line_idx]
                            if current_line_idx < len(original_lines_without_endings)
                            else ""
                        )

                        # Get content from the hunk line (remove the prefix character but keep exact whitespace)
                        hunk_line_content = ""
                        if line.is_context:
                            hunk_line_content = (
                                line.value[1:]
                                if line.value.startswith(" ")
                                else line.value
                            )
                        elif line.is_removed:
                            hunk_line_content = (
                                line.value[1:]
                                if line.value.startswith("-")
                                else line.value
                            )

                        # Remove line ending from hunk line for comparison
                        if hunk_line_content.endswith("\n"):
                            hunk_line_content = hunk_line_content[:-1]
                        if hunk_line_content.endswith("\r"):
                            hunk_line_content = hunk_line_content[:-1]

                        # Debug output to see what's being compared
                        print(f"Comparing line {current_line_idx + 1}:")
                        print(f"  File line: '{file_line}'")
                        print(f"  Hunk line: '{hunk_line_content}'")

                        # EXACT comparison - no tolerance for any differences
                        if hunk_line_content != file_line:
                            # Generate detailed error message
                            return (
                                False,
                                f"Verification failed: Content mismatch at line {current_line_idx + 1}.\n"
                                f"Expected (from file): '{file_line}'\n"
                                f"Found (in patch): '{hunk_line_content}'\n"
                                f"This is an exact mismatch with no tolerance for whitespace or indentation differences.",
                                None,
                            )

                        # Move to the next line in the file
                        current_line_idx += 1
                        # Count this line as processed from the source
                        source_lines_processed += 1

                    # For added lines, we don't need to verify against the original file
                    # and don't increment current_line_idx or source_lines_processed

                # After processing all lines in the hunk, verify we've processed the expected number of source lines
                if source_lines_processed != hunk_source_length:
                    return (
                        False,
                        f"Verification failed: Processed {source_lines_processed} lines from hunk #{i+1}, "
                        f"but hunk header specifies {hunk_source_length} lines.",
                        None,
                    )

            # Check for "No newline at end of file" representation correctness
            for patched_file in patch_set:
                for hunk in patched_file:
                    for line in hunk:
                        # Check if we have the "\ No newline at end of file" marker
                        if line.value.startswith("\\ No newline at end of file"):
                            # The previous line should be the last line of the file
                            # Check if our file actually has no newline at end
                            if file_ends_with_newline:
                                return (
                                    False,
                                    "Verification failed: Patch includes 'No newline at end of file' marker, "
                                    "but the actual file ends with a newline.",
                                    None,
                                )

                # Check the opposite case - file has no newline but patch doesn't have the marker
                if not file_ends_with_newline:
                    # This is more complicated to verify - we would need to check if the last line
                    # of the last hunk should have had the marker
                    pass

            # If we've reached here, all hunks have been verified successfully
            return (
                True,
                "Patch verification successful (strict exact matching)",
                patch_set,
            )

        except Exception as e:
            # Catch any unexpected errors during the verification process
            return (
                False,
                f"Unexpected error during strict patch verification: {type(e).__name__}: {e}\n{traceback.format_exc()}",
                None,
            )

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

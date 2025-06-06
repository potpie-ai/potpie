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
        try:
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
        except Exception as e:
            return False, f"Error during basic patch check: {type(e).__name__}: {e}"

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
        if not _HAS_UNIDIFF:
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
            return (
                False,
                f"Failed to parse patch string for verification: {type(e).__name__}: {e}",
                None,
            )

        if len(patch_set) != 1:
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
                hunk_source_start = hunk.source_start
                hunk_source_length = hunk.source_length

                print(
                    f"Verifying hunk #{i+1}: source_start={hunk_source_start}, source_length={hunk_source_length}"
                )

                current_line_idx = hunk_source_start - 1
                source_lines_processed = 0

                for line in hunk:
                    if line.is_context or line.is_removed:
                        if source_lines_processed >= hunk_source_length:
                            return (
                                False,
                                f"Verification failed: Processed {source_lines_processed} lines from hunk #{i+1}, "
                                f"but hunk header specifies only {hunk_source_length} lines.",
                                None,
                            )

                        if current_line_idx < 0 or current_line_idx >= len(
                            original_lines
                        ):
                            return (
                                False,
                                f"Verification failed: Line {current_line_idx + 1} is out of bounds for file with {len(original_lines)} lines.",
                                None,
                            )

                        file_line = (
                            original_lines_without_endings[current_line_idx]
                            if current_line_idx < len(original_lines_without_endings)
                            else ""
                        )

                        hunk_line_content = line.value
                        if hunk_line_content.endswith("\n"):
                            hunk_line_content = hunk_line_content[:-1]
                        if hunk_line_content.endswith("\r"):
                            hunk_line_content = hunk_line_content[:-1]

                        print(f"Comparing line {current_line_idx + 1}:")
                        print(f"  File line: '{file_line}'")
                        print(f"  Hunk line: '{hunk_line_content}'")

                        if hunk_line_content != file_line:
                            return (
                                False,
                                f"Verification failed: Content mismatch at line {current_line_idx + 1}.\n"
                                f"Expected (from file): '{file_line}'\n"
                                f"Found (in patch): '{hunk_line_content}'\n"
                                f"This is an exact mismatch with no tolerance for whitespace or indentation differences.",
                                None,
                            )

                        current_line_idx += 1
                        source_lines_processed += 1

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
                        if line.value.startswith("\\ No newline at end of file"):
                            if file_ends_with_newline:
                                return (
                                    False,
                                    "Verification failed: Patch includes 'No newline at end of file' marker, "
                                    "but the actual file ends with a newline.",
                                    None,
                                )

            return (
                True,
                "Patch verification successful (strict exact matching)",
                patch_set,
            )

        except Exception as e:
            return (
                False,
                f"Unexpected error during strict patch verification: {type(e).__name__}: {e}",
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
        if not _HAS_UNIDIFF:
            return (
                False,
                "Required library (unidiff) not available for application.",
                None,
            )

        if len(patch_set) != 1:
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
            line_offset = 0

            for i, hunk in enumerate(patched_file):
                hunk_source_start_0_indexed = hunk.source_start - 1
                current_line_idx = hunk_source_start_0_indexed

                for line in hunk:
                    adjusted_pos = current_line_idx + line_offset

                    if line.value.startswith("\\ No newline at end of file"):
                        preserve_final_newline = False
                        continue

                    if line.is_context:
                        current_line_idx += 1

                    elif line.is_removed:
                        if adjusted_pos < 0 or adjusted_pos >= len(result_lines):
                            return (
                                False,
                                f"Application failed: Attempted to remove line at position {adjusted_pos + 1}, "
                                f"which is outside the range of the file ({len(result_lines)} lines).",
                                None,
                            )

                        del result_lines[adjusted_pos]
                        line_offset -= 1
                        current_line_idx += 1

                    elif line.is_added:
                        if adjusted_pos < 0 or adjusted_pos > len(result_lines):
                            return (
                                False,
                                f"Application failed: Attempted to add line at position {adjusted_pos + 1}, "
                                f"which is outside the range of the file ({len(result_lines)} lines).",
                                None,
                            )

                        content_to_add = (
                            line.value[1:] if line.value.startswith("+") else line.value
                        )
                        if content_to_add.endswith("\n"):
                            content_to_add = content_to_add[:-1]
                        if content_to_add.endswith("\r"):
                            content_to_add = content_to_add[:-1]

                        result_lines.insert(adjusted_pos, content_to_add)
                        line_offset += 1

            # Join the lines back into a string, handling the final newline correctly
            result_content = "\n".join(result_lines)
            if preserve_final_newline:
                result_content += "\n"

            return True, "Patch applied successfully", result_content

        except Exception as e:
            return (
                False,
                f"Unexpected error during patch application: {type(e).__name__}: {e}",
                None,
            )

    def _safe_fetch_file(
        self, project_id: str, file_path: str
    ) -> Tuple[bool, str, Optional[str], Optional[str], int]:
        """
        Safely fetch file content with proper error handling.
        Returns: (success, error_msg, file_content, file_content_with_linenumbers, line_count)
        """
        try:
            # Validate inputs
            if not project_id or not isinstance(project_id, str):
                return False, "Invalid project_id parameter", None, None, 0

            if not file_path or not isinstance(file_path, str):
                return False, "Invalid file_path parameter", None, None, 0

            # Check if fetchfiletool has the expected interface
            if not hasattr(self.fetchfiletool, "func") or not callable(
                self.fetchfiletool.func
            ):
                return (
                    False,
                    "fetchfiletool does not have expected 'func' attribute",
                    None,
                    None,
                    0,
                )

            # Fetch content without line numbers
            resp = self.fetchfiletool.func(project_id=project_id, file_path=file_path)

            # Fetch content with line numbers for detailed error reporting
            try:
                resp_with_linenumbers = self.fetchfiletool.func(
                    project_id=project_id, file_path=file_path, with_line_numbers=True
                )
            except Exception as e:
                print(f"Warning: Could not fetch file with line numbers: {e}")
                resp_with_linenumbers = None

            # Validate response
            if not resp or not isinstance(resp, dict) or "content" not in resp:
                error_message = (
                    resp.get("error", "Failed to fetch file content (unknown reason)")
                    if isinstance(resp, dict)
                    else f"Failed to fetch file content (unexpected response format: {type(resp)})"
                )
                return False, error_message, None, None, 0

            file_content = resp["content"]
            if not isinstance(file_content, str):
                return (
                    False,
                    f"File content is not a string: {type(file_content)}",
                    None,
                    None,
                    0,
                )

            file_content_with_linenumbers = None
            if resp_with_linenumbers and isinstance(resp_with_linenumbers, dict):
                file_content_with_linenumbers = resp_with_linenumbers.get("content")

            original_file_lines_count = len(file_content.splitlines())

            return (
                True,
                "",
                file_content,
                file_content_with_linenumbers,
                original_file_lines_count,
            )

        except Exception as e:
            error_msg = (
                f"Error fetching file content for {file_path}: {type(e).__name__}: {e}"
            )
            print(f"File fetch error: {error_msg}")
            return False, error_msg, None, None, 0

    async def arun(self, patch: str, project_id: str, file_path: str) -> Dict[str, Any]:
        """
        Asynchronous method to verify a patch and test its application.
        """
        try:
            # Input validation
            if not patch or not isinstance(patch, str):
                return {
                    "valid": False,
                    "error": {
                        "stage": "input_validation",
                        "reason": "Invalid patch parameter",
                        "details": "Patch must be a non-empty string",
                        "file_path": file_path,
                        "original_file_lines_count": 0,
                    },
                }

            if not project_id or not isinstance(project_id, str):
                return {
                    "valid": False,
                    "error": {
                        "stage": "input_validation",
                        "reason": "Invalid project_id parameter",
                        "details": "project_id must be a non-empty string",
                        "file_path": file_path,
                        "original_file_lines_count": 0,
                    },
                }

            if not file_path or not isinstance(file_path, str):
                return {
                    "valid": False,
                    "error": {
                        "stage": "input_validation",
                        "reason": "Invalid file_path parameter",
                        "details": "file_path must be a non-empty string",
                        "file_path": file_path,
                        "original_file_lines_count": 0,
                    },
                }

            print(f"Processing patch for file: {file_path}")

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
                        "original_file_lines_count": 0,
                    },
                }

            # 2. Fetch file content
            (
                fetch_success,
                fetch_error,
                file_content,
                file_content_with_linenumbers,
                original_file_lines_count,
            ) = self._safe_fetch_file(project_id, file_path)

            if not fetch_success:
                return {
                    "valid": False,
                    "error": {
                        "stage": "fetch_file",
                        "reason": "Failed to fetch file content",
                        "details": fetch_error,
                        "file_path": file_path,
                        "original_file_lines_count": 0,
                    },
                }

            # 3. Detailed patch applicability verification
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
            patch_applied, apply_msg, patched_content = self.apply_patch_from_parsed(
                patch_set, file_content
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

        except Exception as e:
            # Catch-all error handler to prevent unhandled exceptions
            error_msg = (
                f"Unexpected error in VerifyPatchDiff.arun: {type(e).__name__}: {e}"
            )
            print(f"Critical error: {error_msg}")
            print(traceback.format_exc())

            return {
                "valid": False,
                "error": {
                    "stage": "critical_error",
                    "reason": "Unexpected error during patch verification",
                    "details": error_msg,
                    "file_path": file_path,
                    "original_file_lines_count": 0,
                },
            }

    def run(self, patch: str, project_id: str, file_path: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for the arun method that works in any thread.
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context - use thread pool
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self.arun(patch, project_id, file_path))
                    )
                    return future.result()
            else:
                # No running loop, we can use asyncio.run
                return asyncio.run(self.arun(patch, project_id, file_path))

        except RuntimeError as e:
            if (
                "no current event loop" in str(e).lower()
                or "no running event loop" in str(e).lower()
            ):
                # No event loop, create one
                return asyncio.run(self.arun(patch, project_id, file_path))
            else:
                # Other RuntimeError, try thread pool approach
                try:
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            lambda: asyncio.run(self.arun(patch, project_id, file_path))
                        )
                        return future.result()
                except Exception as inner_e:
                    return {
                        "valid": False,
                        "error": {
                            "stage": "runtime",
                            "reason": "Event loop error in synchronous wrapper",
                            "details": f"Original error: {e}. Thread pool error: {inner_e}",
                            "file_path": file_path,
                            "original_file_lines_count": 0,
                        },
                    }
        except Exception as e:
            return {
                "valid": False,
                "error": {
                    "stage": "runtime",
                    "reason": "Unexpected error in synchronous wrapper",
                    "details": f"Error in run method: {type(e).__name__}: {e}",
                    "file_path": file_path,
                    "original_file_lines_count": 0,
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

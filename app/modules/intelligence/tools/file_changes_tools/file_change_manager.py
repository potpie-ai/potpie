import random
import re
import difflib
import os
import string
import time
from typing import List, Dict, Optional, Tuple, Any
import subprocess
import tempfile
import os


def generate_git_diff(
    original_content: str, modified_content: str, file_path: str
) -> str:
    """
    Generate a unified diff using git diff instead of difflib.

    Args:
        original_content: Original file content as string
        modified_content: Modified file content as string (or list of lines)
        file_path: Path to the file (used for diff headers)

    Returns:
        Unified diff as a string in git format
    """
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Ensure paths exist
        os.makedirs(
            os.path.join(temp_dir, "a", os.path.dirname(file_path)), exist_ok=True
        )
        os.makedirs(
            os.path.join(temp_dir, "b", os.path.dirname(file_path)), exist_ok=True
        )

        # Create file paths
        original_file_path = os.path.join(temp_dir, "a", file_path)
        modified_file_path = os.path.join(temp_dir, "b", file_path)

        # Normalize content to handle newline endings consistently
        def normalize_content(content):
            if isinstance(content, list):
                content = "\n".join(content)
            # Ensure consistent line ending handling
            return content

        original_normalized = normalize_content(original_content)
        modified_normalized = normalize_content(modified_content)

        # Write original content
        with open(original_file_path, "w", encoding="utf-8", newline="") as f:
            f.write(original_normalized)

        # Write modified content
        with open(modified_file_path, "w", encoding="utf-8", newline="") as f:
            f.write(modified_normalized)

        # Run git diff
        try:
            # The --no-index tells git to diff files that aren't in a repo
            result = subprocess.run(
                [
                    "git",
                    "diff",
                    "--no-index",
                    "--unified",
                    original_file_path,
                    modified_file_path,
                ],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                check=False,  # Don't raise on non-zero exit (git diff returns 1 if files differ)
            )

            # git diff returns exit code 1 if files differ (which is what we want)
            if result.returncode not in (0, 1):
                raise RuntimeError(f"git diff failed: {result.stderr}")

            # Process the diff output to make it prettier
            diff_output = result.stdout

            # Create a cleaned diff output
            cleaned_diff = []
            for line in diff_output.splitlines():
                # Skip the original diff --git line that contains temp paths
                if line.startswith("diff --git"):
                    cleaned_diff.append(f"diff --git a/{file_path} b/{file_path}")
                    continue

                # Replace the temp paths in --- and +++ lines
                if line.startswith("--- "):
                    cleaned_diff.append(f"--- a/{file_path}")
                    continue
                if line.startswith("+++ "):
                    cleaned_diff.append(f"+++ b/{file_path}")
                    continue

                # Keep all other lines as they are
                cleaned_diff.append(line)

            return "\n".join(cleaned_diff)

        except FileNotFoundError:
            raise RuntimeError(
                "Git executable not found. Make sure git is installed and in your PATH."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate git diff: {str(e)}")


class FileChangeManager:
    """Inmemory changes tracker for files. Store current files agent will be working on and maintains states of changes"""

    def __init__(self, preserve_final_newline=True):
        """
        Initialize an empty file manager with no cached files.

        Args:
            preserve_final_newline: Whether to preserve the original file's final newline behavior
        """
        # Store original file contents: path -> content (string)
        self.original_files = {}
        # Store modified file contents: path -> content (list of lines)
        self.modified_files = {}
        # Track whether original files ended with newlines
        self.original_ends_with_newline = {}
        self.preserve_final_newline = preserve_final_newline

        characters = string.ascii_letters + string.digits

        # Use random.choices to pick characters with replacement
        # k=length specifies the number of characters to pick
        self.curr_hash = "".join(random.choices(characters, k=10))
        # Ensure the changes directory exists
        os.makedirs("changes", exist_ok=True)

    def _detect_final_newline(self, content: str) -> bool:
        """
        Detect if the original content ends with a newline.

        Args:
            content: File content as string

        Returns:
            True if content ends with newline, False otherwise
        """
        return (
            content.endswith("\n") or content.endswith("\r\n") or content.endswith("\r")
        )

    def _reconstruct_content(self, file_path: str) -> str:
        """
        Reconstruct content from lines, preserving original newline behavior.

        Args:
            file_path: Path to the file

        Returns:
            Content as string with proper newline handling
        """
        lines = self.modified_files[file_path]
        if not lines:
            return ""

        content = "\n".join(lines)

        # Preserve original newline behavior if requested
        if self.preserve_final_newline:
            original_had_final_newline = self.original_ends_with_newline.get(
                file_path, True
            )
            if original_had_final_newline and not content.endswith("\n"):
                content += "\n"
            elif not original_had_final_newline and content.endswith("\n"):
                content = content.rstrip("\n")
        else:
            # Default behavior: ensure files end with newline
            if not content.endswith("\n"):
                content += "\n"

        return content

    def _write_changes(self):
        """Write all modified files to disk whenever a change occurs."""
        changed_files = self.get_changed_files()

        # Create the changes directory if it doesn't exist
        changes_dir = f"changes_{self.curr_hash}"
        os.makedirs(changes_dir, exist_ok=True)

        for i, file_path in enumerate(changed_files, 1):
            # Use numbers as filenames (1, 2, 3, 4...)
            output_filename = f"{changes_dir}/file_{i}.py"
            patch_filename = f"{changes_dir}/patch_{i}.txt"

            # Reconstruct content with proper newline handling
            content = self._reconstruct_content(file_path)

            # Prepend the original file path to the content
            output_content = f"{file_path}\n{content}"

            # Write to file (create if it doesn't exist)
            with open(output_filename, "w+", encoding="utf-8") as f:
                f.write(output_content)

            # Generate and write the patch information
            try:
                patch = self.generate_unified_diff(file_path)
                with open(patch_filename, "w+", encoding="utf-8") as f:
                    f.write(patch)
            except Exception as e:
                # If there's an error generating the diff, create an error file
                with open(
                    f"{changes_dir}/patch_{i}_error", "w+", encoding="utf-8"
                ) as f:
                    f.write(f"Error generating patch for {file_path}: {str(e)}")

    def load_file(self, file_path: str, content: str) -> List[str]:
        """
        Load a file into the manager.

        Args:
            file_path: Path to the file relative to repo root
            content: File content as string

        Returns:
            List of lines in the file
        """

        # Store original content and detect newline behavior
        self.original_files[file_path] = content
        self.original_ends_with_newline[file_path] = self._detect_final_newline(content)

        # Initialize modified content as list of lines
        # Use splitlines() to handle different line ending types
        self.modified_files[file_path] = content.splitlines()

        # Write changes immediately
        self._write_changes()

        return self.modified_files[file_path]

    def get_current_content(self, file_path: str) -> List[str]:
        """
        Get the current content (possibly modified) of a file.

        Args:
            file_path: Path to the file

        Returns:
            List of lines in the current state of the file

        Raises:
            FileNotFoundError: If the file hasn't been loaded yet
        """
        if file_path not in self.modified_files:
            raise FileNotFoundError(
                f"File '{file_path}' hasn't been loaded. Call load_file() first."
            )

        return self.modified_files[file_path]

    def get_current_content_as_string(self, file_path: str) -> str:
        """
        Get the current content as a string with proper newline handling.

        Args:
            file_path: Path to the file

        Returns:
            Content as string with proper newlines
        """
        return self._reconstruct_content(file_path)

    def get_lines(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> List[str]:
        """
        Get specific lines from a file.

        Args:
            file_path: Path to the file
            start_line: Starting line number (1-based index)
            end_line: Ending line number (inclusive, 1-based index)

        Returns:
            List of requested lines

        Raises:
            FileNotFoundError: If the file hasn't been loaded
            ValueError: If line numbers are invalid
        """
        lines = self.get_current_content(file_path)

        if start_line is None and end_line is None:
            return lines

        if start_line is not None and (start_line < 1 or start_line > len(lines)):
            raise ValueError(
                f"Start line {start_line} is out of range for file '{file_path}' "
                f"which has {len(lines)} lines.\nFirst few lines:\n"
                f"{self._format_preview(lines, 5)}"
            )

        if end_line is not None and (end_line < 1 or end_line > len(lines)):
            raise ValueError(
                f"End line {end_line} is out of range for file '{file_path}' "
                f"which has {len(lines)} lines.\nLast few lines:\n"
                f"{self._format_preview(lines[-5:], len(lines)-4)}"
            )

        start_idx = (start_line - 1) if start_line is not None else 0
        end_idx = end_line if end_line is not None else len(lines)

        return lines[start_idx:end_idx]

    def get_context(
        self, file_path: str, line_number: int, context_lines: int = 3
    ) -> List[str]:
        """
        Get context around a specific line.

        Args:
            file_path: Path to the file
            line_number: Line number to center context around (1-based)
            context_lines: Number of lines before and after to include

        Returns:
            List of lines with context and line numbers

        Raises:
            FileNotFoundError: If file hasn't been loaded
            ValueError: If line number is invalid
        """
        lines = self.get_current_content(file_path)

        if line_number < 1 or line_number > len(lines):
            raise ValueError(
                f"Line {line_number} is out of range for file '{file_path}' "
                f"which has {len(lines)} lines"
            )

        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)

        # Return with line numbers for context
        context_with_numbers = []
        for i in range(start, end):
            line_num = i + 1
            prefix = "-> " if line_num == line_number else "   "
            context_with_numbers.append(f"{prefix}{line_num}: {lines[i]}")

        return context_with_numbers

    def replace_lines(
        self, file_path: str, start_line: int, end_line: int, new_content: str
    ) -> None:
        """
        Replace a range of lines with new content.

        Args:
            file_path: Path to the file
            start_line: Starting line number to replace (1-based)
            end_line: Ending line number to replace (inclusive, 1-based)
            new_content: New content as a string (will be split by lines)

        Raises:
            FileNotFoundError: If file hasn't been loaded
            ValueError: If line numbers are invalid
        """
        lines = self.get_current_content(file_path)

        if start_line < 1 or start_line > len(lines) + 1:
            # Allow start_line to be one past the end for appending
            context = self.get_context(file_path, min(start_line, len(lines)))
            raise ValueError(
                f"Start line {start_line} is out of range for file '{file_path}' "
                f"which has {len(lines)} lines.\nContext:\n"
                f"{self._format_context(context)}"
            )

        if end_line < start_line - 1 or end_line > len(lines):
            context = self.get_context(file_path, min(end_line, len(lines)))
            raise ValueError(
                f"End line {end_line} is out of range or before start line for file '{file_path}'.\n"
                f"Context:\n{self._format_context(context)}"
            )

        # Split the new content into lines using splitlines() to handle different line endings
        new_lines = new_content.splitlines()

        # Replace the lines
        start_idx = start_line - 1
        end_idx = end_line
        self.modified_files[file_path][start_idx:end_idx] = new_lines

        # Write changes immediately
        self._write_changes()

    def insert_lines(self, file_path: str, after_line: int, new_content: str) -> None:
        """
        Insert content after a specific line.

        Args:
            file_path: Path to the file
            after_line: Line number to insert after (1-based)
            new_content: Content to insert (will be split by lines)

        Raises:
            FileNotFoundError: If file hasn't been loaded
            ValueError: If line number is invalid
        """
        lines = self.get_current_content(file_path)

        if after_line < 0 or after_line > len(lines):
            context = self.get_context(file_path, min(after_line, len(lines)))
            raise ValueError(
                f"Line {after_line} is out of range for file '{file_path}' "
                f"which has {len(lines)} lines.\nContext:\n"
                f"{self._format_context(context)}"
            )

        # Split the new content into lines using splitlines()
        new_lines = new_content.splitlines()

        # Insert the lines
        self.modified_files[file_path][after_line:after_line] = new_lines

        # Write changes immediately
        self._write_changes()

    def delete_lines(self, file_path: str, start_line: int, end_line: int) -> None:
        """
        Delete a range of lines.

        Args:
            file_path: Path to the file
            start_line: First line to delete (1-based)
            end_line: Last line to delete (inclusive, 1-based)

        Raises:
            FileNotFoundError: If file hasn't been loaded
            ValueError: If line numbers are invalid
        """
        lines = self.get_current_content(file_path)

        if start_line < 1 or start_line > len(lines):
            context = self.get_context(file_path, min(start_line, len(lines)))
            raise ValueError(
                f"Start line {start_line} is out of range for file '{file_path}' "
                f"which has {len(lines)} lines.\nContext:\n"
                f"{self._format_context(context)}"
            )

        if end_line < start_line or end_line > len(lines):
            context = self.get_context(file_path, min(end_line, len(lines)))
            raise ValueError(
                f"End line {end_line} is out of range or before start line for file '{file_path}'.\n"
                f"Context:\n{self._format_context(context)}"
            )

        # Delete the lines
        del self.modified_files[file_path][start_line - 1 : end_line]

        # Write changes immediately
        self._write_changes()

    def search_in_file(
        self, file_path: str, pattern: str, is_regex: bool = False
    ) -> List[Tuple[int, str]]:
        """
        Search for a pattern in file and return matching lines with line numbers.

        Args:
            file_path: Path to the file
            pattern: String pattern or regex pattern to search for
            is_regex: Whether the pattern is a regular expression

        Returns:
            List of tuples with (line_number, line_content)

        Raises:
            FileNotFoundError: If file hasn't been loaded
            re.error: If regex pattern is invalid
        """
        lines = self.get_current_content(file_path)
        matches = []

        if is_regex:
            regex = re.compile(pattern)
            for i, line in enumerate(lines):
                if regex.search(line):
                    matches.append((i + 1, line))
        else:
            for i, line in enumerate(lines):
                if pattern in line:
                    matches.append((i + 1, line))

        return matches

    def find_function_definition(
        self, file_path: str, function_name: str
    ) -> Optional[Tuple[int, int]]:
        """
        Find a function definition in a file.

        Args:
            file_path: Path to the file
            function_name: Name of the function to find

        Returns:
            Tuple of (start_line, end_line) if found, None otherwise

        Raises:
            FileNotFoundError: If file hasn't been loaded
        """
        lines = self.get_current_content(file_path)

        # Simplified function detection - adapt based on your code's language
        function_patterns = [
            # JavaScript/TypeScript
            re.compile(r"(function\s+" + re.escape(function_name) + r"\s*\()"),
            re.compile(
                r"(const\s+" + re.escape(function_name) + r"\s*=\s*function\s*\()"
            ),
            re.compile(r"(const\s+" + re.escape(function_name) + r"\s*=\s*\()"),
            # Python
            re.compile(r"(def\s+" + re.escape(function_name) + r"\s*\()"),
            # Java/C#/C++
            re.compile(r"(\w+\s+" + re.escape(function_name) + r"\s*\()"),
        ]

        # Try to find function start
        start_line = None
        for i, line in enumerate(lines):
            for pattern in function_patterns:
                if pattern.search(line):
                    start_line = i + 1
                    break
            if start_line:
                break

        if not start_line:
            return None

        # Simple brace/indentation counting for end detection
        # This is a simple heuristic and may need adjustment for your codebase
        if "{" in lines[start_line - 1]:  # Brace-style language
            brace_count = 0
            for i in range(start_line - 1, len(lines)):
                brace_count += lines[i].count("{") - lines[i].count("}")
                if brace_count <= 0 and i >= start_line - 1:
                    return (start_line, i + 1)
        else:  # Indentation-style language (like Python)
            base_indent = len(lines[start_line - 1]) - len(
                lines[start_line - 1].lstrip()
            )
            for i in range(start_line, len(lines)):
                if (
                    lines[i].strip()
                    and len(lines[i]) - len(lines[i].lstrip()) <= base_indent
                ):
                    return (start_line, i)

        # If we can't determine end, just return a reasonable estimate
        return (start_line, min(start_line + 20, len(lines)))

    def get_changed_files(self) -> List[str]:
        """Get list of all files that have been changed."""
        changed_files = []

        for file_path in self.modified_files:
            # Compare using proper content reconstruction
            original = self.original_files[file_path]
            modified = self._reconstruct_content(file_path)

            if original != modified:
                changed_files.append(file_path)

        return changed_files

    def generate_unified_diff_old(self, file_path: str) -> str:
        """
        Generate a unified diff for a file in Git-style format.

        Args:
            file_path: Path to the file
        Returns:
            Unified diff as a string
        Raises:
            FileNotFoundError: If file hasn't been loaded
            ValueError: If file hasn't been changed
        """
        if file_path not in self.modified_files:
            raise FileNotFoundError(f"File '{file_path}' hasn't been loaded")

        original = self.original_files[file_path].splitlines()
        modified = self.modified_files[file_path]

        if original == modified:
            raise ValueError(f"File '{file_path}' hasn't been changed")

        diff = difflib.unified_diff(
            original,
            modified,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        )

        # Convert diff iterator to a list of lines
        diff_lines = list(diff)

        # Add Git-style header line
        git_header = f"diff --git a/{file_path} b/{file_path}"

        # Combine header with diff content
        return git_header + "\n" + "\n".join(diff_lines)

    def generate_unified_diff(self, file_path: str) -> str:
        """
        Generate a unified diff for a file using git diff.

        Args:
            file_path: Path to the file
        Returns:
            Unified diff as a string
        Raises:
            FileNotFoundError: If file hasn't been loaded
            ValueError: If file hasn't been changed
        """
        if file_path not in self.modified_files:
            raise FileNotFoundError(f"File '{file_path}' hasn't been loaded")

        original = self.original_files[file_path]
        modified = self._reconstruct_content(file_path)

        if original == modified:
            raise ValueError(f"File '{file_path}' hasn't been changed")

        return generate_git_diff(original, modified, file_path)

    def generate_all_diffs(self) -> Dict[str, str]:
        """
        Generate unified diffs for all changed files.

        Returns:
            Dict mapping file paths to their unified diffs
        """
        diffs = {}
        for file_path in self.get_changed_files():
            try:
                diffs[file_path] = self.generate_unified_diff(file_path)
            except ValueError:
                # Skip files that haven't been changed
                pass

        return diffs

    def ensure_final_newline(self, file_path: str) -> None:
        """
        Ensure the file ends with a newline character.

        Args:
            file_path: Path to the file
        """
        if file_path not in self.modified_files:
            raise FileNotFoundError(f"File '{file_path}' hasn't been loaded")

        self.original_ends_with_newline[file_path] = True
        self._write_changes()

    def remove_final_newline(self, file_path: str) -> None:
        """
        Ensure the file does NOT end with a newline character.

        Args:
            file_path: Path to the file
        """
        if file_path not in self.modified_files:
            raise FileNotFoundError(f"File '{file_path}' hasn't been loaded")

        self.original_ends_with_newline[file_path] = False
        self._write_changes()

    def _format_preview(self, lines: List[str], start_line: int = 1) -> str:
        """Format a preview of lines with line numbers."""
        result = []
        for i, line in enumerate(lines):
            result.append(f"{start_line + i}: {line}")
        return "\n".join(result)

    def _format_context(self, context_lines: List[str]) -> str:
        """Format context lines for error messages."""
        return "\n".join(context_lines)

    def write_modified_files(self):
        """Write all modified files to disk."""
        self._write_changes()


def inspect_line_differences(line1, line2):
    """
    Inspects the detailed differences between two strings that appear visually identical.
    Checks for whitespace differences, line endings, and shows hex representation.

    Args:
        line1 (str): First string to compare
        line2 (str): Second string to compare

    Returns:
        dict: Information about the differences
    """
    result = {
        "are_identical": line1 == line2,
        "length_line1": len(line1),
        "length_line2": len(line2),
        "hex_line1": " ".join(f"{ord(c):02x}" for c in line1),
        "hex_line2": " ".join(f"{ord(c):02x}" for c in line2),
        "differences": [],
    }

    # Find position-by-position differences
    for i in range(max(len(line1), len(line2))):
        if i < len(line1) and i < len(line2):
            if line1[i] != line2[i]:
                result["differences"].append(
                    {
                        "position": i,
                        "char1": repr(line1[i]),
                        "char2": repr(line2[i]),
                        "hex1": f"{ord(line1[i]):02x}",
                        "hex2": f"{ord(line2[i]):02x}",
                    }
                )
        elif i < len(line1):
            result["differences"].append(
                {
                    "position": i,
                    "char1": repr(line1[i]),
                    "char2": "missing",
                    "hex1": f"{ord(line1[i]):02x}",
                    "hex2": "n/a",
                }
            )
        else:
            result["differences"].append(
                {
                    "position": i,
                    "char1": "missing",
                    "char2": repr(line2[i]),
                    "hex1": "n/a",
                    "hex2": f"{ord(line2[i]):02x}",
                }
            )

    # Check common whitespace and line ending issues
    result["trailing_whitespace_line1"] = line1.rstrip() != line1
    result["trailing_whitespace_line2"] = line2.rstrip() != line2
    result["has_CR_line1"] = "\r" in line1
    result["has_CR_line2"] = "\r" in line2
    result["has_LF_line1"] = "\n" in line1
    result["has_LF_line2"] = "\n" in line2

    return result


def modify_file_change_manager(FileChangeManager):
    """
    Modifies the FileChangeManager class to add protection against invisible whitespace issues.

    Args:
        FileChangeManager: The original FileChangeManager class

    Returns:
        A subclass with improved line comparison capabilities
    """

    class EnhancedFileChangeManager(FileChangeManager):
        def generate_unified_diff(self, file_path: str) -> str:
            """
            Override to add protection against invisible differences.
            """
            if file_path not in self.modified_files:
                raise FileNotFoundError(f"File '{file_path}' hasn't been loaded")

            original = self.original_files[file_path]
            modified = self._reconstruct_content(file_path)

            # Check for invisible differences
            original_lines = original.splitlines()
            modified_lines = self.modified_files[file_path]

            if len(original_lines) == len(modified_lines):
                has_real_changes = False
                suspicious_lines = []

                for i, (orig, mod) in enumerate(zip(original_lines, modified_lines)):
                    # Skip lines that are obviously different
                    if orig.strip() != mod.strip():
                        has_real_changes = True
                        continue

                    # Look for invisible differences
                    if orig != mod:
                        details = inspect_line_differences(orig, mod)
                        suspicious_lines.append(
                            {"line_number": i + 1, "details": details}
                        )

                # If we only have suspicious whitespace differences, print warnings
                if not has_real_changes and suspicious_lines:
                    print(
                        "WARNING: Detected only whitespace/invisible character differences:"
                    )
                    for sus in suspicious_lines:
                        print(
                            f"  Line {sus['line_number']}: Characters appear identical but differ in whitespace/line-endings"
                        )
                        diffs = sus["details"]["differences"]
                        for diff in diffs:
                            print(
                                f"    Position {diff['position']}: {diff['char1']} vs {diff['char2']}"
                            )

            # Continue with original implementation
            if original == modified:
                raise ValueError(f"File '{file_path}' hasn't been changed")

            return generate_git_diff(original, modified, file_path)

        def normalize_line_endings(self, file_path: str, line_ending="\n"):
            """
            Normalize line endings for a file in memory.

            Args:
                file_path: Path to the file
                line_ending: Target line ending (default: LF)
            """
            if file_path not in self.modified_files:
                raise FileNotFoundError(f"File '{file_path}' hasn't been loaded")

            # Get current content as string
            content = self._reconstruct_content(file_path)

            # Replace all line endings with the target
            content = content.replace("\r\n", "\n").replace("\r", "\n")
            if line_ending != "\n":
                content = content.replace("\n", line_ending)

            # Update the modified files and track newline behavior
            self.modified_files[file_path] = content.splitlines()
            self.original_ends_with_newline[file_path] = content.endswith(line_ending)
            self._write_changes()

        def strip_trailing_whitespace(self, file_path: str):
            """
            Strip trailing whitespace from all lines in a file.

            Args:
                file_path: Path to the file
            """
            if file_path not in self.modified_files:
                raise FileNotFoundError(f"File '{file_path}' hasn't been loaded")

            # Strip trailing whitespace from each line
            self.modified_files[file_path] = [
                line.rstrip() for line in self.modified_files[file_path]
            ]
            self._write_changes()

    return EnhancedFileChangeManager


def verify_diff_integrity(original_line, modified_line):
    """
    Utility function to check if two lines that appear identical have invisible differences.

    Args:
        original_line (str): The original line
        modified_line (str): The modified line

    Returns:
        tuple: (are_visually_same, are_actually_same, difference_details)
    """
    visually_same = original_line.strip() == modified_line.strip()
    actually_same = original_line == modified_line

    if visually_same and not actually_same:
        details = inspect_line_differences(original_line, modified_line)
        return True, False, details

    return visually_same, actually_same, None


# Function to inspect a git-style diff for whitespace/invisible character issues
def inspect_git_diff(diff_text):
    """
    Analyzes a git-style diff to detect potential whitespace-only changes or invisible character issues.

    Args:
        diff_text (str): The git-style diff text

    Returns:
        dict: Analysis results with any suspicious hunks identified
    """
    results = {
        "suspicious_hunks": [],
        "has_whitespace_only_changes": False,
        "line_ending_differences": False,
        "has_newline_at_eof_changes": False,
        "recommendations": [],
    }

    lines = diff_text.splitlines()
    current_hunk = {"line_number": 0, "suspicious_lines": []}
    hunk_header = None

    for i, line in enumerate(lines):
        # Track hunk headers (starting with @@)
        if line.startswith("@@"):
            if current_hunk["suspicious_lines"]:
                results["suspicious_hunks"].append(current_hunk)
            current_hunk = {
                "line_number": i + 1,
                "hunk_header": line,
                "suspicious_lines": [],
            }
            hunk_header = line
            continue

        # Check for "No newline at end of file" messages
        if line.strip() == "\\ No newline at end of file":
            results["has_newline_at_eof_changes"] = True
            continue

        # Look for removal/addition pairs that look visually identical
        if i + 1 < len(lines) and line.startswith("-") and lines[i + 1].startswith("+"):
            removed = line[1:]  # Remove the - prefix
            added = lines[i + 1][1:]  # Remove the + prefix

            # Check if they're visually the same but actually different
            visually_same, actually_same, details = verify_diff_integrity(
                removed, added
            )

            if visually_same and not actually_same:
                current_hunk["suspicious_lines"].append(
                    {"removed_line": i + 1, "added_line": i + 2, "details": details}
                )

                # Track specific types of differences
                if any(
                    d.get("char1") == "'\r'" or d.get("char2") == "'\r'"
                    for d in details["differences"]
                ):
                    results["line_ending_differences"] = True

                results["has_whitespace_only_changes"] = True

    # Add the last hunk if it has suspicious lines
    if current_hunk["suspicious_lines"]:
        results["suspicious_hunks"].append(current_hunk)

    # Generate recommendations
    if results["line_ending_differences"]:
        results["recommendations"].append(
            "Normalize line endings using the normalize_line_endings() method"
        )
    if results["has_whitespace_only_changes"]:
        results["recommendations"].append(
            "Use strip_trailing_whitespace() method to remove trailing whitespace"
        )
    if results["has_newline_at_eof_changes"]:
        results["recommendations"].append(
            "Use ensure_final_newline() or remove_final_newline() to control end-of-file newlines"
        )

    return results

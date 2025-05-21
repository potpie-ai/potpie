import asyncio
import traceback
import re
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class SearchInFileInput(BaseModel):
    """Input schema for SearchInFile tool."""

    file_path: str = Field(description="Path to the file to search in.")
    pattern: str = Field(description="String pattern or regex pattern to search for.")
    is_regex: bool = Field(
        default=False,
        description="Whether the pattern is a regular expression (True) or a plain string (False).",
    )
    context_lines: int = Field(
        default=2,
        description="Number of lines to include before and after each matching line for context.",
    )


class SearchInFile:
    """
    Tool to search for patterns within files loaded in FileChangeManager,
    returning matches with their line numbers and surrounding context.
    """

    name = "SearchInFile"
    description = """Searches for a pattern within a file that has been previously loaded 
        with LoadFileForEditing in FileChangeManager and returns all matching lines with 
        their line numbers and surrounding context.

        This tool is useful for finding specific parts of a file that match a particular 
        pattern, which can then be used to guide further modifications using tools like 
        ReplaceLines, InsertLines, or DeleteLines.

        Parameters:
        - `file_path`: The path to the file to search in.
        - `pattern`: The string or regex pattern to search for.
        - `is_regex`: Boolean indicating whether the pattern is a regular expression (True) or a plain string (False). Default is False.
        - `context_lines`: Number of lines to include before and after each matching line for context. Default is 2.

        The output will include:
        - All matching lines with their line numbers
        - Context lines surrounding each match
        - Total number of matches found
        - File Diffs for the current state of FileChangeManager (Use this to confirm your changes)

        IMPORTANT:
        - This tool searches the in-memory content from FileChangeManager, 
          reflecting all modifications made so far.
        - If using regex patterns (is_regex=True), ensure the pattern is valid according to Python's re module.
        - Matches are returned in order of their appearance in the file.
        - Use this tool to locate specific sections before making targeted changes.
        - This is NOT the current content of the repo, this is only state of the agent's current session of execution
        """

    def __init__(self, file_manager):
        self.file_manager = file_manager

    def _get_context_for_match(
        self, lines: List[str], match_line: int, context_lines: int
    ) -> List[str]:
        """
        Get context lines around a matching line.

        Args:
            lines: All lines in the file
            match_line: Line number of the match (1-based)
            context_lines: Number of context lines before and after

        Returns:
            List of formatted lines with context indicators
        """
        start = max(0, match_line - context_lines - 1)
        end = min(len(lines), match_line + context_lines)

        context_with_numbers = []
        for i in range(start, end):
            line_num = i + 1
            if line_num == match_line:
                prefix = "-> "  # Indicator for the matching line
            else:
                prefix = "   "
            context_with_numbers.append(f"{prefix}{line_num}: {lines[i]}")

        return context_with_numbers

    async def arun(
        self,
        file_path: str,
        pattern: str,
        is_regex: bool = False,
        context_lines: int = 2,
    ) -> Dict[str, Any]:
        """
        Asynchronous method to search for patterns in a file.
        """
        try:
            # Check if file is loaded
            if file_path not in self.file_manager.original_files:
                return {
                    "success": False,
                    "error": {
                        "reason": "File not loaded",
                        "details": f"File '{file_path}' has not been loaded. Use LoadFileForEditing first.",
                        "file_path": file_path,
                    },
                }

            # Get the current content
            lines = self.file_manager.get_current_content(file_path)

            # Perform the search
            try:
                matches = []
                if is_regex:
                    try:
                        regex = re.compile(pattern)
                        for i, line in enumerate(lines):
                            if regex.search(line):
                                match_line_num = i + 1
                                matches.append((match_line_num, line))
                    except re.error as e:
                        return {
                            "success": False,
                            "error": {
                                "reason": "Invalid regex pattern",
                                "details": f"Invalid regex pattern '{pattern}': {str(e)}",
                                "pattern": pattern,
                            },
                        }
                else:
                    for i, line in enumerate(lines):
                        if pattern in line:
                            match_line_num = i + 1
                            matches.append((match_line_num, line))

                # Format the results with context
                results = []
                for match_line_num, match_line in matches:
                    context = self._get_context_for_match(
                        lines, match_line_num, context_lines
                    )
                    results.append(
                        {
                            "line_number": match_line_num,
                            "line_content": match_line,
                            "context": context,
                        }
                    )

                return {
                    "success": True,
                    "message": f"Found {len(matches)} matches for '{pattern}' in '{file_path}'.",
                    "file_path": file_path,
                    "pattern": pattern,
                    "is_regex": is_regex,
                    "match_count": len(matches),
                    "matches": results,
                    "current_diff_state_in_file_changes_manager": self.file_manager.generate_all_diffs(),
                }

            except Exception as e:
                return {
                    "success": False,
                    "error": {
                        "reason": "Search error",
                        "details": f"Error searching in '{file_path}': {type(e).__name__}: {e}\n{traceback.format_exc()}",
                        "file_path": file_path,
                        "pattern": pattern,
                    },
                }

        except FileNotFoundError as e:
            return {
                "success": False,
                "error": {
                    "reason": "File not found in manager",
                    "details": str(e),
                    "file_path": file_path,
                },
            }
        except Exception as e:
            # Catch any unexpected errors
            return {
                "success": False,
                "error": {
                    "reason": "Error searching file",
                    "details": f"Error searching '{file_path}': {type(e).__name__}: {e}\n{traceback.format_exc()}",
                    "file_path": file_path,
                    "pattern": pattern,
                },
            }

    def run(
        self,
        file_path: str,
        pattern: str,
        is_regex: bool = False,
        context_lines: int = 2,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for the arun method.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, submit the coroutine to it
                return asyncio.run_coroutine_threadsafe(
                    self.arun(file_path, pattern, is_regex, context_lines), loop
                ).result()
            else:
                # If no loop is running, run the coroutine until complete
                return loop.run_until_complete(
                    self.arun(file_path, pattern, is_regex, context_lines)
                )
        except RuntimeError:
            # Handle cases where get_event_loop might fail (e.g., no running loop in some contexts)
            return asyncio.run(self.arun(file_path, pattern, is_regex, context_lines))
        except Exception as e:
            # Catch any other unexpected errors in the synchronous wrapper
            return {
                "success": False,
                "error": {
                    "reason": "Unexpected runtime error",
                    "details": f"Error in run method: {type(e).__name__}: {e}\n{traceback.format_exc()}",
                    "file_path": file_path,
                    "pattern": pattern,
                },
            }


def search_in_file_tool(file_manager) -> StructuredTool:
    """
    Returns: StructuredTool for searching patterns in files managed by the FileChangeManager.
    """
    tool = SearchInFile(file_manager)
    return StructuredTool.from_function(
        coroutine=tool.arun,
        func=tool.run,
        name=tool.name,
        description=tool.description,
        args_schema=SearchInFileInput,
    )

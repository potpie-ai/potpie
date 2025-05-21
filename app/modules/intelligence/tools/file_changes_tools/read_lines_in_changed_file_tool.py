import asyncio
import traceback
from typing import Dict, Any, List, Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class GetLinesFromFileInput(BaseModel):
    """Input schema for GetLinesFromFile tool."""

    file_path: str = Field(description="Path to the file to read lines from.")
    start_line: Optional[int] = Field(
        default=None,
        description="Starting line number (1-based index). If not provided, starts from the beginning of the file.",
    )
    end_line: Optional[int] = Field(
        default=None,
        description="Ending line number (inclusive, 1-based index). If not provided, reads to the end of the file.",
    )


class GetLinesFromFile:
    """
    Tool to read a specific range of lines from the current content of a file
    loaded in FileChangeManager, including any modifications made by other tools.
    """

    name = "GetLinesFromFile"
    description = """Reads and returns a specified range of lines from the *current*
        content of a file that has been previously loaded with LoadFileForEditing in FileChangeManager

        This tool is useful for inspecting specific parts of a file after modifications
        have been made using tools like ReplaceLines, InsertLines, or DeleteLines,
        but before generating a final diff.

        You can specify a `start_line`, an `end_line`, or both.
        - If only `start_line` is provided, it reads from `start_line` to the end of the file.
        - If only `end_line` is provided, it reads from the beginning of the file to `end_line`.
        - If neither is provided, it reads the entire current content of the file.
        - Line numbers are 1-based.

        The output will be a list of strings, where each string is a line of content,
        reflecting all changes applied up to the point this tool is called.
        File Diffs for the current state of FileChangeManager (Use this to confirm your changes)

        IMPORTANT:
        - This tool returns content from the in-memory FileChangeManager,
          NOT directly from the filesystem.
        - The content returned includes all changes made by other file modification tools.
        - Use this to precisely verify specific sections of a file during your editing process.
        - This is NOT the current content of the repo, this is only state of the agent's current session of execution
        """

    def __init__(self, file_manager):
        self.file_manager = file_manager

    async def arun(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronous method to read specific lines from a file.
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

            # Get the current content (which includes all modifications) using the manager's method
            try:
                lines = self.file_manager.get_lines(file_path, start_line, end_line)
            except ValueError as e:
                return {
                    "success": False,
                    "error": {
                        "reason": "Invalid line range",
                        "details": str(e),
                        "file_path": file_path,
                        "start_line": start_line,
                        "end_line": end_line,
                    },
                }

            # Join lines with line numbers for display
            formatted_lines = []
            actual_start_line = (
                start_line if start_line is not None else 1
            )  # For formatting
            for i, line in enumerate(lines):
                formatted_lines.append(f"{actual_start_line + i}: {line}")

            return {
                "success": True,
                "message": f"Successfully retrieved lines from '{file_path}'.",
                "file_path": file_path,
                "retrieved_lines": formatted_lines,
                "line_count": len(lines),
                "content": "\n".join(lines),  # Provide raw content as well if needed
                "current_diff_state_in_file_changes_manager": self.file_manager.generate_all_diffs(),
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
                    "reason": "Error retrieving lines from file",
                    "details": f"Error retrieving lines for '{file_path}': {type(e).__name__}: {e}\n{traceback.format_exc()}",
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                },
            }

    def run(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for the arun method.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, submit the coroutine to it
                return asyncio.run_coroutine_threadsafe(
                    self.arun(file_path, start_line, end_line), loop
                ).result()
            else:
                # If no loop is running, run the coroutine until complete
                return loop.run_until_complete(
                    self.arun(file_path, start_line, end_line)
                )
        except RuntimeError:
            # Handle cases where get_event_loop might fail (e.g., no running loop in some contexts)
            return asyncio.run(self.arun(file_path, start_line, end_line))
        except Exception as e:
            # Catch any other unexpected errors in the synchronous wrapper
            return {
                "success": False,
                "error": {
                    "reason": "Unexpected runtime error",
                    "details": f"Error in run method: {type(e).__name__}: {e}\n{traceback.format_exc()}",
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                },
            }


def get_lines_from_file_tool(file_manager) -> StructuredTool:
    """
    Returns: StructuredTool for reading specific lines from the current content
             of a file in the FileChangeManager.
    """
    tool = GetLinesFromFile(file_manager)
    return StructuredTool.from_function(
        coroutine=tool.arun,
        func=tool.run,
        name=tool.name,
        description=tool.description,
        args_schema=GetLinesFromFileInput,
    )

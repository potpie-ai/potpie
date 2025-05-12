import asyncio
import traceback
from typing import Dict, Any, List, Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.modules.intelligence.tools.file_changes_tools.file_change_manager import (
    FileChangeManager,
)


class GenerateFileDiffInput(BaseModel):
    """Input schema for GenerateFileDiff tool. No inputs are required as it operates on all managed files."""

    pass


class GenerateFileDiff:
    """
    Tool to generate a unified patch diff for all files that have been modified
    since they were loaded or last reset in the FileChangeManager.
    """

    name = "GenerateFileDiff"
    description = """Generates a unified patch diff for all files that have been modified
        since they were loaded or last reset in the FileChangeManager.
        
        This tool is typically used after you have made all necessary changes to files
        using tools like ReplaceLines, InsertLines, DeleteLines, etc., and you are ready
        to review or apply the accumulated changes.
        
        The output will be a dictionary where keys are file paths and values are the
        unified diff strings. An empty dictionary means no files have been changed.
        
        IMPORTANT:
        - Use this tool when you are confident that you have completed all modifications
          and want to see the final combined changes.
        - The generated diffs represent the differences between the *original* loaded
          content and the *current modified* content in memory.
        - If no files have been modified, an empty dictionary will be returned,
          along with a descriptive message.
        """

    def __init__(self, file_manager: FileChangeManager):
        self.file_manager = file_manager

    async def arun(self) -> Dict[str, Any]:
        """
        Asynchronous method to generate unified diffs for all changed files.
        """
        try:
            diffs = self.file_manager.generate_all_diffs()

            if not diffs:
                return {
                    "success": True,
                    "message": "No files have been modified. No diffs generated.",
                    "diffs": {},
                    "num_files_changed": 0,
                }
            else:
                return {
                    "success": True,
                    "message": f"Successfully generated diffs for {len(diffs)} modified files.",
                    "diffs": diffs,
                    "num_files_changed": len(diffs),
                }

        except Exception as e:
            # Catch any unexpected errors
            return {
                "success": False,
                "error": {
                    "reason": "Error generating diffs",
                    "details": f"Error generating unified diffs: {type(e).__name__}: {e}\n{traceback.format_exc()}",
                },
            }

    def run(self) -> Dict[str, Any]:
        """
        Synchronous wrapper for the arun method.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, submit the coroutine to it
                return asyncio.run_coroutine_threadsafe(self.arun(), loop).result()
            else:
                # If no loop is running, run the coroutine until complete
                return loop.run_until_complete(self.arun())
        except RuntimeError:
            # Handle cases where get_event_loop might fail (e.g., no running loop in some contexts)
            return asyncio.run(self.arun())
        except Exception as e:
            # Catch any other unexpected errors in the synchronous wrapper
            return {
                "success": False,
                "error": {
                    "reason": "Unexpected runtime error",
                    "details": f"Error in run method: {type(e).__name__}: {e}\n{traceback.format_exc()}",
                },
            }


def generate_file_diff_tool(file_manager) -> StructuredTool:
    """
    Returns: StructuredTool for generating patch diffs for all changed files
             in the FileChangeManager.
    """
    tool = GenerateFileDiff(file_manager)
    return StructuredTool.from_function(
        coroutine=tool.arun,
        func=tool.run,
        name=tool.name,
        description=tool.description,
        args_schema=GenerateFileDiffInput,  # No arguments needed for this tool
    )

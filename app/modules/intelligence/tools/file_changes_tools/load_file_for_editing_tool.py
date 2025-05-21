import asyncio
import traceback
from typing import Dict, Any, Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class LoadFileForEditingInput(BaseModel):
    file_path: str = Field(description="Path to the file to load for editing.")
    project_id: str = Field(description="Project ID to fetch the file from.")


class LoadFileForEditing:
    """
    Tool to load a file into FileChangeManager for editing.
    Maintains state across tool invocations.
    """

    name = "LoadFileForEditing"
    description = """Loads a file into the FileChangeManager for editing.
        This tool allows you to access files that you need to modify to fix GitHub issues.
        
        IMPORTANT: Only load files that you intend to edit. This avoids unnecessary processing.
        
        This tool is part of a workflow for making incremental changes to a repository:
        1. Load files you need to edit with this tool (LoadFileForEditing)
        2. Make changes using ReplaceLines tool
        3. Search file content with SearchInFile
        4. Generate unified diffs with GenerateFileDiff when changes are complete
        
        The FileChangeManager maintains the state of all loaded files during your session,
        so you can work on multiple files incrementally before generating final diffs.
        
        Use reset = True to load the original file again
        Note: All the changes in the file will be lost
        
        Returns the loaded file content and line count for reference.
        """

    def __init__(self, fetchfiletool: StructuredTool, file_manager):
        self.fetchfiletool = fetchfiletool
        self.file_manager = file_manager

    async def arun(
        self, file_path: str, project_id: str, reset: bool = False
    ) -> Dict[str, Any]:
        """
        Asynchronous method to load a file into FileChangeManager.
        """
        try:
            # Check if file is already loaded
            if file_path in self.file_manager.original_files and not reset:
                line_count = len(self.file_manager.get_current_content(file_path))
                return {
                    "success": True,
                    "message": f"File '{file_path}' is already loaded in FileChangeManager.",
                    "file_path": file_path,
                    "line_count": line_count,
                    "already_loaded": True,
                }

            # Fetch file content using the provided tool
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
                        "reason": "Failed to fetch file content",
                        "details": error_message,
                        "file_path": file_path,
                    },
                }

            # Get the file content
            file_content = resp["content"]

            # Load the file into FileChangeManager
            self.file_manager.load_file(file_path, content=file_content)
            line_count = len(self.file_manager.get_current_content(file_path))

            # Return success response with useful information
            return {
                "success": True,
                "message": f"File '{file_path}' loaded successfully into FileChangeManager.",
                "file_path": file_path,
                "line_count": line_count,
                "already_loaded": False,
                "preview": self._get_file_preview(file_path),
            }

        except Exception as e:
            # Catch any unexpected errors
            return {
                "success": False,
                "error": {
                    "reason": "Error during file loading",
                    "details": f"Error loading file '{file_path}': {type(e).__name__}: {e}\n{traceback.format_exc()}",
                    "file_path": file_path,
                },
            }

    def _get_file_preview(self, file_path: str, preview_lines: int = 10) -> str:
        """Get a short preview of the file content."""
        try:
            lines = self.file_manager.get_current_content(file_path)
            if len(lines) <= preview_lines:
                preview = lines
            else:
                preview = lines[:preview_lines]
                preview.append("... (file continues)")

            return "\n".join([f"{i+1}: {line}" for i, line in enumerate(preview)])
        except Exception:
            return "(Preview not available)"

    def run(
        self, file_path: str, project_id: str, reset: bool = False
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for the arun method.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, submit the coroutine to it
                return asyncio.run_coroutine_threadsafe(
                    self.arun(file_path, project_id), loop
                ).result()
            else:
                # If no loop is running, run the coroutine until complete
                return loop.run_until_complete(self.arun(file_path, project_id))
        except RuntimeError:
            # Handle cases where get_event_loop might fail
            return asyncio.run(self.arun(file_path, project_id, reset=reset))
        except Exception as e:
            # Catch any other unexpected errors in the synchronous wrapper
            return {
                "success": False,
                "error": {
                    "reason": "Unexpected runtime error",
                    "details": f"Error in run method: {type(e).__name__}: {e}\n{traceback.format_exc()}",
                    "file_path": file_path,
                },
            }


def load_file_for_editing_tool(
    fetchfiletool: StructuredTool, file_manager
) -> StructuredTool:
    """
    Returns: StructuredTool for loading files into FileChangeManager for editing.
    """
    tool = LoadFileForEditing(fetchfiletool, file_manager)
    return StructuredTool.from_function(
        coroutine=tool.arun,
        func=tool.run,
        name=tool.name,
        description=tool.description,
        args_schema=LoadFileForEditingInput,
    )

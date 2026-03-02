"""Pydantic input models for code changes tool functions."""

import json
from typing import List, Optional, Literal

from pydantic import BaseModel, Field, field_validator


class AddFileInput(BaseModel):
    file_path: str = Field(description="Path to the file to add (e.g., 'src/main.py')")
    content: str = Field(description="Full content of the file to add")
    description: Optional[str] = Field(
        default=None, description="Optional description of what this file does"
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Optional project ID (from conversation context) to write change directly to the repo worktree in non-local mode.",
    )


class UpdateFileInput(BaseModel):
    file_path: str = Field(description="Path to the file to update")
    content: str = Field(description="New content for the file")
    description: Optional[str] = Field(
        default=None, description="Optional description of the change"
    )
    preserve_previous: bool = Field(
        default=True,
        description="Whether to preserve previous content for reference",
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Optional project ID (from conversation context) to write change directly to the repo worktree in non-local mode.",
    )


class DeleteFileInput(BaseModel):
    file_path: str = Field(description="Path to the file to delete")
    description: Optional[str] = Field(
        default=None, description="Optional reason for deletion"
    )
    preserve_content: bool = Field(
        default=True,
        description="Whether to preserve file content before deletion",
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Optional project ID (from conversation context) to delete the file from the repo worktree in non-local mode.",
    )


class RevertFileInput(BaseModel):
    file_path: str = Field(
        description="Path to the file to revert (workspace-relative, e.g. 'src/main.py')"
    )
    target: Optional[Literal["saved", "HEAD"]] = Field(
        default="saved",
        description=(
            "Revert target: 'saved' = restore from disk (last saved, discard unsaved); "
            "'HEAD' = restore from git HEAD (committed version), then save."
        ),
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the revert (e.g. reason)",
    )


class GetFileInput(BaseModel):
    file_path: str = Field(description="Path to the file to retrieve")


class ListFilesInput(BaseModel):
    change_type_filter: Optional[str] = Field(
        default=None,
        description="Filter by change type: 'add', 'update', or 'delete'",
    )
    path_pattern: Optional[str] = Field(
        default=None,
        description="Optional regex pattern to filter files by path (e.g., '.*\\.py$' for Python files)",
    )


class SearchContentInput(BaseModel):
    pattern: str = Field(
        description="Regex pattern to search for in file contents (grep-like search)"
    )
    file_pattern: Optional[str] = Field(
        default=None,
        description="Optional regex pattern to filter files by path before searching",
    )
    case_sensitive: bool = Field(
        default=False, description="Whether search should be case-sensitive"
    )


class ClearFileInput(BaseModel):
    file_path: str = Field(description="Path to the file to clear from changes")


class ExportChangesInput(BaseModel):
    format: str = Field(
        default="dict",
        description="Export format: 'dict' (file_path -> content), 'list' (list of changes), 'json' (JSON string), or 'diff' (git-style diff patch)",
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Optional project ID to fetch original content from repository for accurate diff. Use project_id from conversation context.",
    )


class GetChangesForPRInput(BaseModel):
    conversation_id: str = Field(
        ...,
        description="Conversation ID where changes are stored in Redis. Pass from delegate context when creating PR.",
    )


class UpdateFileLinesInput(BaseModel):
    file_path: str = Field(description="Path to the file to update")
    start_line: int = Field(description="Starting line number (1-indexed, inclusive)")
    end_line: Optional[int] = Field(
        default=None,
        description="Ending line number (1-indexed, inclusive). Omit or set to null to update only the single line at start_line.",
    )
    new_content: str = Field(description="Content to replace the lines with")
    description: Optional[str] = Field(
        default=None, description="Optional description of the change"
    )
    project_id: str = Field(
        ...,
        description="REQUIRED: Project ID (from context) to fetch file content from repository.",
    )


class ReplaceInFileInput(BaseModel):
    file_path: str = Field(description="Path to the file to update")
    old_str: str = Field(
        description=(
            "The exact literal text to find and replace. Must match character-for-character "
            "including indentation and whitespace. Include enough surrounding lines to make it unique."
        )
    )
    new_str: str = Field(description="The replacement text. Must preserve correct indentation.")
    description: Optional[str] = Field(
        default=None, description="Optional description of the change"
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Optional project ID (from context) to fetch file content from repository.",
    )


class InsertLinesInput(BaseModel):
    file_path: str = Field(description="Path to the file to update")
    line_number: int = Field(description="Line number to insert at (1-indexed)")
    content: str = Field(description="Content to insert")
    description: Optional[str] = Field(
        default=None, description="Optional description of the change"
    )
    insert_after: bool = Field(
        default=True,
        description="If True, insert after line_number; if False, insert before",
    )
    project_id: str = Field(
        ...,
        description="REQUIRED: Project ID (from context) to fetch file content from repository.",
    )


class DeleteLinesInput(BaseModel):
    file_path: str = Field(description="Path to the file to update")
    start_line: int = Field(description="Starting line number (1-indexed, inclusive)")
    end_line: Optional[int] = Field(
        default=None,
        description="Ending line number (1-indexed, inclusive). If None, only start_line is deleted",
    )
    description: Optional[str] = Field(
        default=None, description="Optional description of the change"
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Optional project ID (from context) to fetch file content from repository.",
    )


class ShowUpdatedFileInput(BaseModel):
    file_paths: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of file paths to display. Must be an array/list of strings. "
            "If not provided (null/empty), shows ALL changed files."
        ),
    )

    @field_validator("file_paths", mode="before")
    @classmethod
    def coerce_file_paths_to_list(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            v_stripped = v.strip()
            if v_stripped.startswith("[") and v_stripped.endswith("]"):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    pass
            return [v]
        if isinstance(v, list):
            return v
        return [str(v)]


class ShowDiffInput(BaseModel):
    file_path: Optional[str] = Field(
        default=None,
        description="Optional file path to show diff for a specific file. If not provided, shows diffs for all changed files.",
    )
    context_lines: int = Field(
        default=3,
        description="Number of context lines to include around changes in the diff (default: 3)",
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Optional project ID to fetch original content from repository for accurate diffs.",
    )


class GetFileDiffInput(BaseModel):
    file_path: str = Field(description="Path to the file to get diff for")
    context_lines: int = Field(
        default=3,
        description="Number of context lines to include around changes in the diff (default: 3)",
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Optional project ID to fetch original content from repository for accurate diff.",
    )


class GetComprehensiveMetadataInput(BaseModel):
    project_id: Optional[str] = Field(
        default=None,
        description="Optional project ID for logging purposes. Use project_id from conversation context.",
    )

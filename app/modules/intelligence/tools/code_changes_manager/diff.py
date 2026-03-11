"""Diff generation for code changes."""

import difflib
from typing import Dict, Optional

from sqlalchemy.orm import Session

from app.modules.utils.logger import setup_logger

from .models import ChangeType, FileChange

logger = setup_logger(__name__)


def create_unified_diff(
    old_content: str,
    new_content: str,
    old_path: str,
    new_path: str,
    context_lines: int,
) -> str:
    """Create a unified diff string from old and new content."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=old_path,
        tofile=new_path,
        lineterm="\n",
        n=context_lines,
    )
    return "".join(diff)


def generate_git_diff_patch(
    file_path: str,
    old_content: str,
    new_content: str,
    context_lines: int = 3,
) -> str:
    """
    Generate a git-style diff patch for a single file.

    Args:
        file_path: Path to the file
        old_content: Original file content
        new_content: New file content
        context_lines: Number of context lines to include in diff

    Returns:
        Git-style diff patch string
    """
    old_lines = old_content.splitlines(keepends=True) if old_content else []
    new_lines = new_content.splitlines(keepends=True) if new_content else []

    diff_lines = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="\n",
            n=context_lines,
        )
    )

    if not diff_lines:
        return ""

    git_diff = f"diff --git a/{file_path} b/{file_path}\n"
    git_diff += "".join(diff_lines)
    return git_diff


def fetch_repo_file_content_for_diff(
    project_id: str, file_path: str, db: Session
) -> Optional[str]:
    """Fetch file content from the project repository for diff base. Returns None on failure."""
    try:
        from app.modules.code_provider.code_provider_service import CodeProviderService
        from app.modules.code_provider.git_safe import (
            safe_git_operation,
            GitOperationError,
        )
        from app.modules.projects.projects_model import Project

        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return None
        cp_service = CodeProviderService(db)

        def _fetch():
            return cp_service.get_file_content(
                repo_name=project.repo_name,
                file_path=file_path,
                branch_name=project.branch_name,
                start_line=None,
                end_line=None,
                project_id=project_id,
                commit_id=project.commit_id,
            )

        try:
            return safe_git_operation(
                _fetch,
                max_retries=1,
                timeout=20.0,
                max_total_timeout=25.0,
                operation_name=f"get_file_diff_repo_content({file_path})",
            )
        except GitOperationError:
            return None
    except Exception as e:
        logger.debug(f"diff.fetch_repo_file_content_for_diff: Failed: {e}")
        return None

"""Library-friendly project service adapter.

This module provides an HTTPException-free wrapper around the existing
ProjectService for use in the PotpieRuntime library.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from potpie.exceptions import ProjectError, ProjectNotFoundError

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class LibraryProjectService:
    """Library-friendly project service that wraps the app's ProjectService.

    Key differences from app.modules.projects.projects_service:
    - Raises library exceptions instead of HTTPException
    - Uses create_from_config for clean dependency injection
    """

    def __init__(self, db: Session):
        """Initialize the library project service.

        Args:
            db: Database session
        """
        self.db = db
        self._service = None

    def _get_service(self):
        """Get or create the underlying ProjectService."""
        if self._service is None:
            from app.modules.projects.projects_service import ProjectService

            self._service = ProjectService.create_from_config(
                self.db, raise_library_exceptions=True
            )
        return self._service

    async def get_project_name(self, project_ids: List[str]) -> str:
        """Get project name for given project IDs.

        Args:
            project_ids: List of project identifiers

        Returns:
            Project name

        Raises:
            ProjectNotFoundError: If no projects found
            ProjectError: If operation fails
        """
        try:
            return await self._get_service().get_project_name(project_ids)
        except Exception as e:
            if "not found" in str(e).lower():
                raise ProjectNotFoundError(str(e)) from e
            raise ProjectError(f"Failed to get project name: {e}") from e

    async def register_project(
        self,
        repo_name: str,
        branch_name: str,
        user_id: str,
        project_id: str,
        *,
        commit_id: Optional[str] = None,
        repo_path: Optional[str] = None,
    ) -> str:
        """Register a new project or update existing one.

        Args:
            repo_name: Repository name
            branch_name: Branch name
            user_id: User identifier
            project_id: Project identifier
            commit_id: Optional commit ID
            repo_path: Optional local repository path

        Returns:
            Project ID

        Raises:
            ProjectError: If registration fails
        """
        try:
            return await self._get_service().register_project(
                repo_name=repo_name,
                branch_name=branch_name,
                user_id=user_id,
                project_id=project_id,
                commit_id=commit_id,
                repo_path=repo_path,
            )
        except Exception as e:
            raise ProjectError(f"Failed to register project: {e}") from e

    async def list_projects(self, user_id: str) -> List[Dict[str, Any]]:
        """List all projects for a user.

        Args:
            user_id: User identifier

        Returns:
            List of project dictionaries

        Raises:
            ProjectError: If operation fails
        """
        try:
            return await self._get_service().list_projects(user_id)
        except Exception as e:
            raise ProjectError(f"Failed to list projects: {e}") from e

    async def get_project_from_db_by_id(
        self, project_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get project by ID.

        Args:
            project_id: Project identifier

        Returns:
            Project dictionary or None if not found

        Raises:
            ProjectError: If operation fails
        """
        try:
            return await self._get_service().get_project_from_db_by_id(project_id)
        except Exception as e:
            raise ProjectError(f"Failed to get project: {e}") from e

    async def get_project_from_db(
        self,
        repo_name: str,
        branch_name: str,
        user_id: str,
        *,
        repo_path: Optional[str] = None,
        commit_id: Optional[str] = None,
    ) -> Optional[Any]:
        """Get project by repository details.

        Args:
            repo_name: Repository name
            branch_name: Branch name
            user_id: User identifier
            repo_path: Optional local repository path
            commit_id: Optional commit ID

        Returns:
            Project object or None

        Raises:
            ProjectError: If operation fails
        """
        try:
            return await self._get_service().get_project_from_db(
                repo_name=repo_name,
                branch_name=branch_name,
                user_id=user_id,
                repo_path=repo_path,
                commit_id=commit_id,
            )
        except Exception as e:
            raise ProjectError(f"Failed to get project: {e}") from e

    async def update_project_status(self, project_id: str, status) -> None:
        """Update project status.

        Args:
            project_id: Project identifier
            status: New status (ProjectStatusEnum)

        Raises:
            ProjectNotFoundError: If project not found
            ProjectError: If operation fails
        """
        try:
            await self._get_service().update_project_status(project_id, status)
        except Exception as e:
            if "not found" in str(e).lower():
                raise ProjectNotFoundError(f"Project not found: {project_id}") from e
            raise ProjectError(f"Failed to update project status: {e}") from e

    async def delete_project(self, project_id: str) -> None:
        """Delete a project.

        Args:
            project_id: Project identifier

        Raises:
            ProjectNotFoundError: If project not found
            ProjectError: If operation fails
        """
        try:
            await self._get_service().delete_project(project_id)
        except ProjectNotFoundError:
            raise
        except Exception as e:
            if "not found" in str(e).lower():
                raise ProjectNotFoundError(f"Project not found: {project_id}") from e
            raise ProjectError(f"Failed to delete project: {e}") from e

    async def get_project_repo_details_from_db(
        self, project_id: str, user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get repository details for a project.

        Args:
            project_id: Project identifier
            user_id: User identifier

        Returns:
            Repository details or None

        Raises:
            ProjectError: If operation fails
        """
        try:
            return await self._get_service().get_project_repo_details_from_db(
                project_id, user_id
            )
        except Exception as e:
            raise ProjectError(f"Failed to get repo details: {e}") from e

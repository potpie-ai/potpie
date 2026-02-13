"""Project resource for PotpieRuntime library."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from uuid6 import uuid7

from potpie.core.exception_utils import ExceptionTranslator
from potpie.exceptions import ProjectError, ProjectNotFoundError
from potpie.resources.base import BaseResource
from potpie.types.project import ProjectInfo, ProjectStatus

if TYPE_CHECKING:
    from potpie.config import RuntimeConfig
    from potpie.core.database import DatabaseManager
    from potpie.core.neo4j import Neo4jManager

logger = logging.getLogger(__name__)


class ProjectResource(BaseResource):
    """Access and manage projects.

    Wraps the existing ProjectService with a clean library interface.
    Translates HTTP exceptions to library exceptions.
    User context is passed per-operation, not stored in the resource.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        db_manager: DatabaseManager,
        neo4j_manager: Neo4jManager,
    ):
        super().__init__(config, db_manager, neo4j_manager)

    def _get_service(self):
        """Get a ProjectService instance with a fresh session."""
        from app.modules.projects.projects_service import ProjectService

        session = self._db_manager.get_session()
        service = ProjectService.create_from_config(
            session, raise_library_exceptions=True
        )
        return service, session

    def _generate_project_id(
        self,
        user_id: str,
        repo_name: str,
        branch_name: str,
        repo_path: Optional[str] = None,
    ) -> str:
        """Generate a deterministic project ID."""
        return str(uuid7())

    async def register(
        self,
        repo_name: str,
        branch_name: str,
        user_id: str,
        *,
        repo_path: Optional[str] = None,
        commit_id: Optional[str] = None,
    ) -> str:
        """Register a new project for indexing.

        Args:
            repo_name: Full repository name (e.g., "owner/repo")
            branch_name: Branch to index
            user_id: User ID who owns this project
            repo_path: Local path to repository (for local repos)
            commit_id: Specific commit to index

        Returns:
            project_id: Unique identifier for the project

        Raises:
            ProjectError: If registration fails

        Example:
            project_id = await runtime.projects.register(
                repo_name="langchain-ai/langchain",
                branch_name="main",
                user_id="user-123",
            )
        """
        service, session = self._get_service()
        try:
            project_id = self._generate_project_id(
                user_id, repo_name, branch_name, repo_path
            )

            result = await service.register_project(
                repo_name=repo_name,
                branch_name=branch_name,
                user_id=user_id,
                project_id=project_id,
                commit_id=commit_id,
                repo_path=repo_path,
            )

            logger.info(f"Registered project: {result}")
            return result

        except ProjectError:
            raise
        except Exception as e:
            session.rollback()
            translated = ExceptionTranslator.translate_exception(
                e, ProjectError, ProjectNotFoundError
            )
            raise translated from e
        finally:
            session.close()

    async def get(self, project_id: str) -> Optional[ProjectInfo]:
        """Get project information by ID.

        Args:
            project_id: Project identifier

        Returns:
            ProjectInfo with id, repo_name, branch_name, status, commit_id
            None if project not found
        """
        service, session = self._get_service()
        try:
            result = await service.get_project_from_db_by_id(project_id)
            if result is None:
                return None
            return ProjectInfo.from_dict(result)

        except ProjectNotFoundError:
            return None
        except ProjectError:
            raise
        except Exception as e:
            translated = ExceptionTranslator.translate_exception(
                e, ProjectError, ProjectNotFoundError
            )
            if isinstance(translated, ProjectNotFoundError):
                return None
            raise translated from e
        finally:
            session.close()

    async def get_by_repo(
        self,
        repo_name: str,
        branch_name: str,
        user_id: str,
        *,
        repo_path: Optional[str] = None,
        commit_id: Optional[str] = None,
    ) -> Optional[ProjectInfo]:
        """Find existing project by repository details.

        Args:
            repo_name: Repository name
            branch_name: Branch name
            user_id: User ID who owns the project
            repo_path: Local repository path (optional)
            commit_id: Specific commit (optional)

        Returns:
            ProjectInfo if found, None otherwise
        """
        service, session = self._get_service()
        try:
            project = await service.get_project_from_db(
                repo_name=repo_name,
                branch_name=branch_name,
                user_id=user_id,
                repo_path=repo_path,
                commit_id=commit_id,
            )

            if project is None:
                return None

            return ProjectInfo(
                id=project.id,
                repo_name=project.repo_name,
                branch_name=project.branch_name,
                status=ProjectStatus.from_string(project.status),
                commit_id=project.commit_id,
                repo_path=project.repo_path,
                user_id=project.user_id,
                created_at=project.created_at,
                updated_at=project.updated_at,
            )

        except ProjectNotFoundError:
            return None
        except ProjectError:
            raise
        except Exception as e:
            translated = ExceptionTranslator.translate_exception(
                e, ProjectError, ProjectNotFoundError
            )
            if isinstance(translated, ProjectNotFoundError):
                return None
            raise translated from e
        finally:
            session.close()

    async def list(self, user_id: str) -> List[ProjectInfo]:
        """List all projects for a user.

        Args:
            user_id: User ID whose projects to list

        Returns:
            List of ProjectInfo objects
    async def list(self, user_id: str) -> List[str]:
        """List all project IDs for a user.

        Args:
            user_id: User ID whose projects to list

        Returns:
            List of project ID strings
        """
        service, session = self._get_service()
        try:
            projects = await service.list_projects(user_id)
            return [p["id"] for p in projects]

        except ProjectError:
            raise
    async def delete(self, project_id: str) -> None:
        """Delete a project and its associated data.

        Args:
            project_id: Project to delete

        Raises:
            ProjectNotFoundError: If project not found
            ProjectError: If deletion fails
        """
        service, session = self._get_service()
        try:
            await service.delete_project(project_id)
            logger.info(f"Deleted project: {project_id}")

        except ProjectNotFoundError:
            raise
        except ProjectError:
            raise
        except Exception as e:
            session.rollback()
            translated = ExceptionTranslator.translate_exception(
                e, ProjectError, ProjectNotFoundError
            )
            raise translated from e
        finally:
            session.close()

    async def update_status(
        self,
        project_id: str,
        status: ProjectStatus,
    ) -> None:
        """Update project status.

        Args:
            project_id: Project to update
            status: New status

        Raises:
            ProjectNotFoundError: If project not found
            ProjectError: If update fails
        """
        from app.modules.projects.projects_schema import ProjectStatusEnum

        service, session = self._get_service()
        try:
            status_enum = ProjectStatusEnum(status.value)
            await service.update_project_status(project_id, status_enum)
            logger.info(f"Updated project {project_id} status to {status.value}")

        except ProjectNotFoundError:
            raise
        except ProjectError:
            raise
        except Exception as e:
            session.rollback()
            translated = ExceptionTranslator.translate_exception(
                e, ProjectError, ProjectNotFoundError
            )
            raise translated from e
        finally:
            session.close()

    async def get_repo_details(self, project_id: str, user_id: str) -> Optional[dict]:
        """Get repository details for a project.

        Args:
            project_id: Project identifier
            user_id: User ID who owns the project

        Returns:
            Dictionary with repo_name, branch_name, repo_path, commit_id
            None if project not found
        """
        service, session = self._get_service()
        try:
            result = await service.get_project_repo_details_from_db(project_id, user_id)
            return result

        except ProjectNotFoundError:
            return None
        except ProjectError:
            raise
        except Exception as e:
            translated = ExceptionTranslator.translate_exception(
                e, ProjectError, ProjectNotFoundError
            )
            if isinstance(translated, ProjectNotFoundError):
                return None
            raise translated from e
        finally:
            session.close()

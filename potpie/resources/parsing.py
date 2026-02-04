"""Parsing resource for PotpieRuntime library."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from potpie.exceptions import ParsingError, ProjectNotFoundError
from potpie.resources.base import BaseResource
from potpie.types.parsing import ParsingResult
from potpie.types.project import ProjectStatus

if TYPE_CHECKING:
    from potpie.config import RuntimeConfig
    from potpie.core.database import DatabaseManager
    from potpie.core.neo4j import Neo4jManager

logger = logging.getLogger(__name__)


class ParsingResource(BaseResource):
    """Parse and index projects into knowledge graph.

    Wraps the existing ParsingService with a clean library interface.
    Executes synchronously (no Celery).
    User context is passed per-operation, not stored in the resource.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        db_manager: DatabaseManager,
        neo4j_manager: Neo4jManager,
    ):
        super().__init__(config, db_manager, neo4j_manager)

    def _get_neo4j_config(self) -> dict:
        """Get Neo4j configuration dictionary for services."""
        return self._neo4j_manager.get_neo4j_config()

    async def parse_project(
        self,
        project_id: str,
        user_id: str,
        user_email: str = "",
        *,
        cleanup_graph: bool = True,
    ) -> ParsingResult:
        """Parse a project and build its knowledge graph.

        This is a synchronous operation that may take several minutes
        for large repositories.

        Args:
            project_id: Project to parse
            user_id: User ID performing the parsing
            user_email: User email (optional, used for notifications)
            cleanup_graph: Whether to remove existing graph data first

        Returns:
            ParsingResult with status and any errors

        Example:
            result = await runtime.parsing.parse_project(
                project_id,
                user_id="user-123",
                user_email="user@example.com"
            )
            if result.success:
                print(f"Parsed successfully")
        """
        from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
        from app.modules.parsing.graph_construction.parsing_service import (
            ParsingService,
        )
        from app.modules.projects.projects_service import ProjectService

        session = self._db_manager.get_session()
        try:
            project_service = ProjectService(session)
            project_data = await project_service.get_project_from_db_by_id(project_id)

            if project_data is None:
                raise ProjectNotFoundError(f"Project not found: {project_id}")

            repo_name = project_data.get("project_name")
            branch_name = project_data.get("branch_name")
            repo_path = project_data.get("repo_path")
            commit_id = project_data.get("commit_id")

            repo_details = ParsingRequest(
                repo_name=repo_name,
                branch_name=branch_name,
                repo_path=repo_path,
                commit_id=commit_id,
            )

            neo4j_config = self._get_neo4j_config()
            parsing_service = ParsingService.create_from_config(
                session,
                user_id,
                neo4j_config,
                raise_library_exceptions=True,
            )

            try:
                result = await parsing_service.parse_directory(
                    repo_details=repo_details,
                    user_id=user_id,
                    user_email=user_email,
                    project_id=project_id,
                    cleanup_graph=cleanup_graph,
                )

                logger.info(f"Parsing completed for project {project_id}: {result}")

                return ParsingResult.success_result(
                    project_id=project_id,
                    node_count=None,
                )

            except ParsingError as e:
                logger.error(f"Parsing failed for project {project_id}: {e}")
                return ParsingResult.error_result(
                    project_id=project_id,
                    error_message=str(e),
                )

        except ProjectNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Parsing failed for project {project_id}: {e}")
            session.rollback()
            return ParsingResult.error_result(
                project_id=project_id,
                error_message=str(e),
            )
        finally:
            session.close()

    async def get_status(self, project_id: str) -> ProjectStatus:
        """Get current parsing status for a project.

        Args:
            project_id: Project identifier

        Returns:
            Current ProjectStatus

        Raises:
            ProjectNotFoundError: If project not found
        """
        from app.modules.projects.projects_service import ProjectService

        session = self._db_manager.get_session()
        try:
            project_service = ProjectService(session)
            project_data = await project_service.get_project_from_db_by_id(project_id)

            if project_data is None:
                raise ProjectNotFoundError(f"Project not found: {project_id}")

            return ProjectStatus.from_string(project_data.get("status", "error"))

        except ProjectNotFoundError:
            raise
        except Exception as e:
            raise ParsingError(f"Failed to get status: {e}") from e
        finally:
            session.close()

    async def duplicate_graph(
        self,
        source_project_id: str,
        target_project_id: str,
        user_id: str,
    ) -> None:
        """Duplicate knowledge graph from one project to another.

        Args:
            source_project_id: Source project ID
            target_project_id: Target project ID
            user_id: User ID performing the duplication

        Raises:
            ProjectNotFoundError: If source project not found
            ParsingError: If duplication fails
        """
        from app.modules.parsing.graph_construction.parsing_service import (
            ParsingService,
        )

        session = self._db_manager.get_session()
        try:
            parsing_service = ParsingService(session, user_id)
            await parsing_service.duplicate_graph(
                old_repo_id=source_project_id,
                new_repo_id=target_project_id,
            )

            logger.info(
                f"Duplicated graph from {source_project_id} to {target_project_id}"
            )

        except Exception as e:
            session.rollback()
            raise ParsingError(f"Failed to duplicate graph: {e}") from e
        finally:
            session.close()

    async def cleanup_graph(self, project_id: str) -> None:
        """Clean up knowledge graph for a project.

        Args:
            project_id: Project whose graph should be cleaned up

        Raises:
            ParsingError: If cleanup fails
        """
        from app.modules.parsing.graph_construction.code_graph_service import (
            CodeGraphService,
        )

        try:
            neo4j_config = self._get_neo4j_config()
            session = self._db_manager.get_session()

            try:
                code_graph_service = CodeGraphService(
                    neo4j_config["uri"],
                    neo4j_config["username"],
                    neo4j_config["password"],
                    session,
                )

                code_graph_service.cleanup_graph(project_id)
                logger.info(f"Cleaned up graph for project {project_id}")

            finally:
                code_graph_service.close()
                session.close()

        except Exception as e:
            raise ParsingError(f"Failed to cleanup graph: {e}") from e

    async def get_node_count(self, project_id: str) -> int:
        """Get the number of nodes in the knowledge graph for a project.

        Args:
            project_id: Project identifier

        Returns:
            Number of nodes in the graph

        Raises:
            ParsingError: If count fails
        """
        try:
            query = """
            MATCH (n:NODE {repoId: $project_id})
            RETURN count(n) as count
            """
            results = await self._neo4j_manager.execute_query(
                query,
                parameters={"project_id": project_id},
            )

            if results:
                return results[0].get("count", 0)
            return 0

        except Exception as e:
            raise ParsingError(f"Failed to get node count: {e}") from e

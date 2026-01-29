"""Library-friendly parsing service adapter.

This module provides a Celery-free, HTTPException-free wrapper around the
existing ParsingService for use in the PotpieRuntime library.
"""

from __future__ import annotations

import logging
import os
import shutil
import traceback
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Optional

from potpie.exceptions import ParsingError, ProjectNotFoundError

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class LibraryParsingService:
    """Library-friendly parsing service that wraps the app's ParsingService.

    Key differences from app.modules.parsing.graph_construction.parsing_service:
    - No Celery task dependencies (synchronous execution)
    - Raises library exceptions instead of HTTPException
    - Accepts injectable Neo4j config instead of using config_provider
    - No email/webhook notifications (those are web-app concerns)
    """

    def __init__(
        self,
        db: Session,
        user_id: str,
        neo4j_config: Dict[str, str],
        project_path: str = "./projects",
        development_mode: bool = False,
    ):
        """Initialize the library parsing service.

        Args:
            db: Database session
            user_id: User identifier
            neo4j_config: Dict with 'uri', 'username', 'password' keys
            project_path: Path for storing project files
            development_mode: Whether in development mode
        """
        self.db = db
        self.user_id = user_id
        self._neo4j_config = neo4j_config
        self._project_path = project_path
        self._development_mode = development_mode

        # Import app services lazily to avoid circular imports
        self._parse_helper = None
        self._project_service = None
        self._inference_service = None
        self._search_service = None

    def _get_parse_helper(self):
        """Lazily get ParseHelper instance."""
        if self._parse_helper is None:
            from app.modules.parsing.graph_construction.parsing_helper import (
                ParseHelper,
            )

            self._parse_helper = ParseHelper(self.db)
        return self._parse_helper

    def _get_project_service(self):
        """Lazily get ProjectService instance."""
        if self._project_service is None:
            from app.modules.projects.projects_service import ProjectService

            self._project_service = ProjectService.create_from_config(
                self.db, raise_library_exceptions=True
            )
        return self._project_service

    def _get_inference_service(self):
        """Lazily get InferenceService instance."""
        if self._inference_service is None:
            from app.modules.parsing.knowledge_graph.inference_service import (
                InferenceService,
            )

            self._inference_service = InferenceService(self.db, self.user_id)
        return self._inference_service

    def _get_search_service(self):
        """Lazily get SearchService instance."""
        if self._search_service is None:
            from app.modules.search.search_service import SearchService

            self._search_service = SearchService(self.db)
        return self._search_service

    @contextmanager
    def _change_dir(self, path: str):
        """Context manager for temporary directory change."""
        old_dir = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(old_dir)

    async def parse_directory(
        self,
        repo_name: str,
        branch_name: str,
        project_id: str,
        user_email: str = "",
        *,
        repo_path: Optional[str] = None,
        commit_id: Optional[str] = None,
        cleanup_graph: bool = True,
    ) -> Dict[str, Any]:
        """Parse a repository and build its knowledge graph.

        This is the library-friendly version of ParsingService.parse_directory.
        It executes synchronously (no Celery), uses injectable config, and
        raises library exceptions.

        Args:
            repo_name: Repository name (e.g., "owner/repo")
            branch_name: Branch to parse
            project_id: Project identifier
            user_email: Optional user email (unused in library mode)
            repo_path: Optional local repository path
            commit_id: Optional specific commit to parse
            cleanup_graph: Whether to clean up existing graph data first

        Returns:
            Dict with 'message' and 'id' keys on success

        Raises:
            ProjectNotFoundError: If project not found
            ParsingError: If parsing fails
        """
        from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
        from app.modules.parsing.graph_construction.code_graph_service import (
            CodeGraphService,
        )
        from app.modules.projects.projects_schema import ProjectStatusEnum
        from git import Repo

        project_service = self._get_project_service()
        parse_helper = self._get_parse_helper()
        inference_service = self._get_inference_service()
        extracted_dir = None

        try:
            if cleanup_graph:
                try:
                    code_graph_service = CodeGraphService(
                        self._neo4j_config["uri"],
                        self._neo4j_config["username"],
                        self._neo4j_config["password"],
                        self.db,
                    )
                    code_graph_service.cleanup_graph(project_id)
                except Exception as e:
                    logger.exception(
                        "Error in cleanup_graph",
                        extra={"project_id": project_id, "user_id": self.user_id},
                    )
                    raise ParsingError(f"Failed to cleanup graph: {e}") from e

            repo_details = ParsingRequest(
                repo_name=repo_name,
                branch_name=branch_name,
                repo_path=repo_path,
                commit_id=commit_id,
            )

            repo, owner, auth = await parse_helper.clone_or_copy_repository(
                repo_details, self.user_id
            )

            if self._development_mode:
                extracted_dir, project_id = await parse_helper.setup_project_directory(
                    repo,
                    branch_name,
                    auth,
                    repo_details,
                    self.user_id,
                    project_id,
                    commit_id=commit_id,
                )
            else:
                extracted_dir, project_id = await parse_helper.setup_project_directory(
                    repo,
                    branch_name,
                    auth,
                    repo,
                    self.user_id,
                    project_id,
                    commit_id=commit_id,
                )

            if isinstance(repo, Repo):
                language = parse_helper.detect_repo_language(extracted_dir)
            else:
                languages = repo.get_languages()
                if languages:
                    language = max(languages, key=languages.get).lower()
                else:
                    language = parse_helper.detect_repo_language(extracted_dir)

            await self._analyze_directory(extracted_dir, project_id, language)

            return {
                "message": "The project has been parsed successfully",
                "id": project_id,
            }

        except ProjectNotFoundError:
            raise
        except ParsingError:
            raise
        except Exception as e:
            tb_str = "".join(traceback.format_exception(None, e, e.__traceback__))
            logger.error(
                f"Error during parsing: {e}\nTraceback:\n{tb_str}",
                extra={"project_id": project_id, "user_id": self.user_id},
            )
            self.db.rollback()
            try:
                await project_service.update_project_status(
                    project_id, ProjectStatusEnum.ERROR
                )
            except Exception:
                logger.exception(
                    "Failed to update project status after error",
                    extra={"project_id": project_id, "user_id": self.user_id},
                )
            raise ParsingError(f"Parsing failed for project {project_id}: {e}") from e

        finally:
            if (
                extracted_dir
                and isinstance(extracted_dir, str)
                and os.path.exists(extracted_dir)
                and extracted_dir.startswith(self._project_path)
            ):
                shutil.rmtree(extracted_dir, ignore_errors=True)

    async def _analyze_directory(
        self,
        extracted_dir: str,
        project_id: str,
        language: str,
    ) -> None:
        """Analyze directory and build knowledge graph.

        Args:
            extracted_dir: Path to extracted repository
            project_id: Project identifier
            language: Detected programming language

        Raises:
            ParsingError: If analysis fails
        """
        from app.modules.parsing.graph_construction.code_graph_service import (
            CodeGraphService,
        )
        from app.modules.parsing.graph_construction.parsing_helper import (
            ParsingFailedError,
        )
        from app.modules.projects.projects_schema import ProjectStatusEnum

        logger.info(f"Analyzing directory: {extracted_dir} for project {project_id}")

        if not isinstance(extracted_dir, str):
            raise ParsingError(
                f"Invalid extracted_dir type: {type(extracted_dir)}, value: {extracted_dir}"
            )

        if not os.path.exists(extracted_dir):
            raise ParsingError(f"Directory not found: {extracted_dir}")

        project_service = self._get_project_service()
        inference_service = self._get_inference_service()

        project_details = await project_service.get_project_from_db_by_id(project_id)
        if not project_details:
            raise ProjectNotFoundError(f"Project with ID {project_id} not found")

        if language == "other":
            await project_service.update_project_status(
                project_id, ProjectStatusEnum.ERROR
            )
            raise ParsingError(
                "Repository doesn't consist of a language currently supported."
            )

        code_graph_service = None
        try:
            code_graph_service = CodeGraphService(
                self._neo4j_config["uri"],
                self._neo4j_config["username"],
                self._neo4j_config["password"],
                self.db,
            )

            code_graph_service.create_and_store_graph(
                extracted_dir, project_id, self.user_id
            )

            await project_service.update_project_status(
                project_id, ProjectStatusEnum.PARSED
            )

            await inference_service.run_inference(project_id)

            await project_service.update_project_status(
                project_id, ProjectStatusEnum.READY
            )

            logger.info(f"Successfully parsed project {project_id}")

        except ParsingFailedError as e:
            raise ParsingError(str(e)) from e
        except Exception as e:
            raise ParsingError(f"Failed to analyze directory: {e}") from e
        finally:
            if code_graph_service:
                code_graph_service.close()

    async def duplicate_graph(
        self,
        source_project_id: str,
        target_project_id: str,
    ) -> None:
        """Duplicate knowledge graph from one project to another.

        Args:
            source_project_id: Source project ID
            target_project_id: Target project ID

        Raises:
            ParsingError: If duplication fails
        """
        search_service = self._get_search_service()
        inference_service = self._get_inference_service()

        try:
            await search_service.clone_search_indices(
                source_project_id, target_project_id
            )

            node_batch_size = 3000
            relationship_batch_size = 3000

            with inference_service.driver.session() as session:
                offset = 0
                while True:
                    nodes_query = """
                    MATCH (n:NODE {repoId: $old_repo_id})
                    RETURN n.node_id AS node_id, n.text AS text, n.file_path AS file_path,
                           n.start_line AS start_line, n.end_line AS end_line, n.name AS name,
                           COALESCE(n.docstring, '') AS docstring,
                           COALESCE(n.embedding, []) AS embedding,
                           labels(n) AS labels
                    SKIP $offset LIMIT $limit
                    """
                    nodes_result = session.run(
                        nodes_query,
                        old_repo_id=source_project_id,
                        offset=offset,
                        limit=node_batch_size,
                    )
                    nodes = [dict(record) for record in nodes_result]

                    if not nodes:
                        break

                    create_query = """
                    UNWIND $batch AS node
                    CALL apoc.create.node(node.labels, {
                        repoId: $new_repo_id,
                        node_id: node.node_id,
                        text: node.text,
                        file_path: node.file_path,
                        start_line: node.start_line,
                        end_line: node.end_line,
                        name: node.name,
                        docstring: node.docstring,
                        embedding: node.embedding
                    }) YIELD node AS new_node
                    RETURN new_node
                    """
                    session.run(
                        create_query, new_repo_id=target_project_id, batch=nodes
                    )
                    offset += node_batch_size

            with inference_service.driver.session() as session:
                offset = 0
                while True:
                    relationships_query = """
                    MATCH (n:NODE {repoId: $old_repo_id})-[r]->(m:NODE)
                    RETURN n.node_id AS start_node_id, type(r) AS relationship_type, m.node_id AS end_node_id
                    SKIP $offset LIMIT $limit
                    """
                    relationships_result = session.run(
                        relationships_query,
                        old_repo_id=source_project_id,
                        offset=offset,
                        limit=relationship_batch_size,
                    )
                    relationships = [dict(record) for record in relationships_result]

                    if not relationships:
                        break

                    relationship_query = """
                    UNWIND $batch AS relationship
                    MATCH (a:NODE {repoId: $new_repo_id, node_id: relationship.start_node_id}),
                          (b:NODE {repoId: $new_repo_id, node_id: relationship.end_node_id})
                    CALL apoc.create.relationship(a, relationship.relationship_type, {}, b) YIELD rel
                    RETURN rel
                    """
                    session.run(
                        relationship_query,
                        new_repo_id=target_project_id,
                        batch=relationships,
                    )
                    offset += relationship_batch_size

            logger.info(
                f"Successfully duplicated graph from {source_project_id} to {target_project_id}"
            )

        except Exception as e:
            logger.exception(
                "Error duplicating graph",
                extra={
                    "source_project_id": source_project_id,
                    "target_project_id": target_project_id,
                },
            )
            raise ParsingError(f"Failed to duplicate graph: {e}") from e

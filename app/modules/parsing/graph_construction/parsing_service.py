import os
import shutil
import traceback
from asyncio import create_task
from contextlib import contextmanager

from fastapi import HTTPException
from git import Repo
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.parsing.graph_construction.code_graph_service import CodeGraphService
from app.modules.parsing.graph_construction.parsing_helper import (
    ParseHelper,
    ParsingFailedError,
    ParsingServiceError,
)
from app.modules.parsing.knowledge_graph.inference_service import InferenceService
from app.modules.parsing.graph_construction.parsing_schema import RepoDetails
from app.modules.projects.projects_schema import ProjectStatusEnum
from app.modules.projects.projects_service import ProjectService
from app.modules.search.search_service import SearchService
from app.modules.utils.email_helper import EmailHelper
from app.modules.utils.logger import log_context, setup_logger
from app.modules.utils.parse_webhook_helper import ParseWebhookHelper

from .parsing_schema import ParsingRequest

logger = setup_logger(__name__)


class ParsingService:
    def __init__(
        self,
        db: Session,
        user_id: str,
        *,
        neo4j_config: dict | None = None,
        raise_library_exceptions: bool = False,
    ):
        """Initialize ParsingService.

        Args:
            db: Database session
            user_id: User identifier
            neo4j_config: Optional Neo4j config dict for library usage.
                          If None, uses config_provider.
            raise_library_exceptions: If True, raise ParsingServiceError
                                      instead of HTTPException
        """
        self.db = db
        self.parse_helper = ParseHelper(db)
        self.project_service = ProjectService(db)
        self.inference_service = InferenceService(db, user_id)
        self.search_service = SearchService(db)
        self.github_service = CodeProviderService(db)
        self._neo4j_config = neo4j_config
        self._raise_library_exceptions = raise_library_exceptions

    @classmethod
    def create_from_config(
        cls,
        db: Session,
        user_id: str,
        neo4j_config: dict,
        raise_library_exceptions: bool = True,
    ) -> "ParsingService":
        """Factory method for library usage with explicit Neo4j config.

        Args:
            db: Database session
            user_id: User identifier
            neo4j_config: Dict with 'uri', 'username', 'password' keys
            raise_library_exceptions: Whether to raise library exceptions

        Returns:
            Configured ParsingService instance
        """
        return cls(
            db,
            user_id,
            neo4j_config=neo4j_config,
            raise_library_exceptions=raise_library_exceptions,
        )

    def _get_neo4j_config(self) -> dict:
        """Get Neo4j config, preferring injected config over config_provider."""
        if self._neo4j_config is not None:
            return self._neo4j_config
        return config_provider.get_neo4j_config()

    @contextmanager
    def change_dir(self, path):
        old_dir = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(old_dir)

    async def parse_directory(
        self,
        repo_details: ParsingRequest,
        user_id: str,
        user_email: str,
        project_id: int,
        cleanup_graph: bool = True,
    ):
        # Set up logging context with domain IDs
        with log_context(project_id=str(project_id), user_id=user_id):
            project_manager = ProjectService(self.db)
            extracted_dir = None
            try:
                # Early check: if project already exists and is READY for this commit, skip parsing
                if cleanup_graph and repo_details.commit_id:
                    existing_project = await project_manager.get_project_from_db_by_id(
                        project_id
                    )
                    if existing_project:
                        is_latest = await self.parse_helper.check_commit_status(
                            str(project_id), requested_commit_id=repo_details.commit_id
                        )
                        if is_latest:
                            logger.info(
                                "Skipping parse for project %s - already parsed at commit %s",
                                project_id,
                                existing_project.get("commit_id"),
                            )
                            await project_manager.update_project_status(
                                project_id, ProjectStatusEnum.READY
                            )
                            return {
                                "message": "Project already parsed for requested commit",
                                "id": project_id,
                            }

                if cleanup_graph:
                    neo4j_config = self._get_neo4j_config()

                    try:
                        code_graph_service = CodeGraphService(
                            neo4j_config["uri"],
                            neo4j_config["username"],
                            neo4j_config["password"],
                            self.db,
                        )

                        code_graph_service.cleanup_graph(str(project_id))
                    except Exception:
                        logger.exception(
                            "Error in cleanup_graph",
                            project_id=project_id,
                            user_id=user_id,
                        )
                        if self._raise_library_exceptions:
                            raise ParsingServiceError("Failed to cleanup graph")
                        raise HTTPException(
                            status_code=500, detail="Internal server error"
                        )

                # Convert ParsingRequest to RepoDetails
                repo_details_converted = RepoDetails(
                    repo_name=repo_details.repo_name or "",
                    branch_name=repo_details.branch_name or "",
                    repo_path=repo_details.repo_path,
                    commit_id=repo_details.commit_id,
                )
                repo, owner, auth = await self.parse_helper.clone_or_copy_repository(
                    repo_details_converted, user_id
                )
                if config_provider.get_is_development_mode():
                    (
                        extracted_dir,
                        project_id,
                    ) = await self.parse_helper.setup_project_directory(
                        repo,
                        repo_details.branch_name,
                        auth,
                        repo_details,
                        user_id,
                        project_id,
                        commit_id=repo_details.commit_id,
                    )
                else:
                    (
                        extracted_dir,
                        project_id,
                    ) = await self.parse_helper.setup_project_directory(
                        repo,
                        repo_details.branch_name,
                        auth,
                        repo,
                        user_id,
                        project_id,
                        commit_id=repo_details.commit_id,
                    )

                if isinstance(repo, Repo):
                    language = self.parse_helper.detect_repo_language(extracted_dir)
                else:
                    languages = repo.get_languages()
                    if languages:
                        language = max(languages, key=languages.get).lower()
                    else:
                        language = self.parse_helper.detect_repo_language(extracted_dir)

                await self.analyze_directory(
                    extracted_dir, project_id, user_id, self.db, language, user_email
                )
                message = "The project has been parsed successfully"
                return {"message": message, "id": project_id}

            except ParsingServiceError as e:
                message = str(f"{project_id} Failed during parsing: " + str(e))
                await project_manager.update_project_status(
                    project_id, ProjectStatusEnum.ERROR
                )
                if not self._raise_library_exceptions:
                    await ParseWebhookHelper().send_slack_notification(
                        project_id, message
                    )
                    raise HTTPException(status_code=500, detail=message)
                raise

            except Exception as e:
                # Log the full traceback server-side for debugging
                tb_str = "".join(traceback.format_exception(None, e, e.__traceback__))
                logger.exception(
                    "Error during parsing",
                    project_id=project_id,
                    user_id=user_id,
                )
                # Log the formatted traceback string explicitly for detailed debugging
                logger.error(
                    f"Full traceback:\n{tb_str}",
                    project_id=project_id,
                    user_id=user_id,
                )
                # Rollback the database session to clear any pending transactions
                self.db.rollback()
                try:
                    await project_manager.update_project_status(
                        project_id, ProjectStatusEnum.ERROR
                    )
                except Exception:
                    logger.exception(
                        "Failed to update project status after error",
                        project_id=project_id,
                        user_id=user_id,
                    )
                if self._raise_library_exceptions:
                    raise ParsingServiceError(
                        f"Parsing failed for project {project_id}: {e}"
                    ) from e
                await ParseWebhookHelper().send_slack_notification(project_id, str(e))
                # Raise generic error with correlation ID for client
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal server error. Please contact support with project ID: {project_id}",
                )

            finally:
                project_path = os.getenv("PROJECT_PATH")
                if (
                    extracted_dir
                    and isinstance(extracted_dir, str)
                    and os.path.exists(extracted_dir)
                    and project_path
                    and extracted_dir.startswith(project_path)
                ):
                    shutil.rmtree(extracted_dir, ignore_errors=True)

    def create_neo4j_indices(self, graph_manager):
        # Create existing indices from blar_graph
        graph_manager.create_entityId_index()
        graph_manager.create_node_id_index()
        graph_manager.create_function_name_index()

        with graph_manager.driver.session() as session:
            # Existing composite index for repo_id and node_id
            node_query = """
                CREATE INDEX repo_id_node_id_NODE IF NOT EXISTS FOR (n:NODE) ON (n.repoId, n.node_id)
                """
            session.run(node_query)

            # New composite index for name and repo_id to speed up node name lookups
            name_repo_query = """
                CREATE INDEX node_name_repo_id_NODE IF NOT EXISTS FOR (n:NODE) ON (n.name, n.repoId)
                """
            session.run(name_repo_query)

            # New index for relationship types - using correct Neo4j syntax
            rel_type_query = """
                CREATE LOOKUP INDEX relationship_type_lookup IF NOT EXISTS FOR ()-[r]->() ON EACH type(r)
                """
            session.run(rel_type_query)

    async def analyze_directory(
        self,
        extracted_dir: str,
        project_id: int,
        user_id: str,
        db,
        language: str,
        user_email: str,
    ):
        logger.info(
            f"ParsingService: Parsing project {project_id}: Analyzing directory: {extracted_dir}"
        )

        # Validate that extracted_dir is a valid path
        if not isinstance(extracted_dir, str):
            error_msg = f"ParsingService: Invalid extracted_dir type: {type(extracted_dir)}, value: {extracted_dir}"
            logger.bind(project_id=project_id, user_id=user_id).error(error_msg)
            raise ValueError(
                f"Expected string path, got {type(extracted_dir)}: {extracted_dir}"
            )

        if not os.path.exists(extracted_dir):
            error_msg = f"ParsingService: Directory does not exist: {extracted_dir}"
            logger.bind(project_id=project_id, user_id=user_id).error(error_msg)
            raise FileNotFoundError(f"Directory not found: {extracted_dir}")

        logger.info(
            f"ParsingService: Directory exists and is accessible: {extracted_dir}"
        )
        project_details = await self.project_service.get_project_from_db_by_id(
            project_id
        )
        if project_details:
            repo_name = project_details.get("project_name")
            branch_name = project_details.get("branch_name")
        else:
            error_msg = f"Project with ID {project_id} not found."
            logger.bind(project_id=project_id, user_id=user_id).error(error_msg)
            if self._raise_library_exceptions:
                raise ParsingServiceError(error_msg)
            raise HTTPException(status_code=404, detail="Project not found.")

        service = None
        if language != "other":
            try:
                neo4j_config = self._get_neo4j_config()
                service = CodeGraphService(
                    neo4j_config["uri"],
                    neo4j_config["username"],
                    neo4j_config["password"],
                    db,
                )

                service.create_and_store_graph(extracted_dir, project_id, user_id)

                await self.project_service.update_project_status(
                    project_id, ProjectStatusEnum.PARSED
                )
                # Generate docstrings using InferenceService
                await self.inference_service.run_inference(str(project_id))
                logger.info(f"DEBUGNEO4J: After inference project {project_id}")
                self.inference_service.log_graph_stats(project_id)
                await self.project_service.update_project_status(
                    project_id, ProjectStatusEnum.READY
                )
                if not self._raise_library_exceptions and user_email:
                    create_task(
                        EmailHelper().send_email(user_email, repo_name, branch_name)
                    )
                logger.info(f"DEBUGNEO4J: After update project status {project_id}")
                self.inference_service.log_graph_stats(project_id)
            finally:
                if service is not None:
                    service.close()
                logger.info(f"DEBUGNEO4J: After close service {project_id}")
                self.inference_service.log_graph_stats(project_id)
        else:
            await self.project_service.update_project_status(
                project_id, ProjectStatusEnum.ERROR
            )
            if not self._raise_library_exceptions:
                await ParseWebhookHelper().send_slack_notification(project_id, "Other")
            logger.info(f"DEBUGNEO4J: After update project status {project_id}")
            self.inference_service.log_graph_stats(project_id)
            raise ParsingFailedError(
                "Repository doesn't consist of a language currently supported."
            )

    async def duplicate_graph(self, old_repo_id: str, new_repo_id: str):
        await self.search_service.clone_search_indices(old_repo_id, new_repo_id)
        node_batch_size = 3000  # Fixed batch size for nodes
        relationship_batch_size = 3000  # Fixed batch size for relationships
        try:
            # Step 1: Fetch and duplicate nodes in batches
            with self.inference_service.driver.session() as session:
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
                        old_repo_id=old_repo_id,
                        offset=offset,
                        limit=node_batch_size,
                    )
                    nodes = [dict(record) for record in nodes_result]

                    if not nodes:
                        break

                    # Insert nodes under the new repo ID, preserving labels, docstring, and embedding
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
                    session.run(create_query, new_repo_id=new_repo_id, batch=nodes)
                    offset += node_batch_size

            # Step 2: Fetch and duplicate relationships in batches
            with self.inference_service.driver.session() as session:
                offset = 0
                while True:
                    relationships_query = """
                    MATCH (n:NODE {repoId: $old_repo_id})-[r]->(m:NODE)
                    RETURN n.node_id AS start_node_id, type(r) AS relationship_type, m.node_id AS end_node_id
                    SKIP $offset LIMIT $limit
                    """
                    relationships_result = session.run(
                        relationships_query,
                        old_repo_id=old_repo_id,
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
                        relationship_query, new_repo_id=new_repo_id, batch=relationships
                    )
                    offset += relationship_batch_size

            logger.info(
                f"Successfully duplicated graph from {old_repo_id} to {new_repo_id}"
            )

        except Exception:
            logger.exception(
                "Error duplicating graph",
                old_repo_id=old_repo_id,
                new_repo_id=new_repo_id,
            )

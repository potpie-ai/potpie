import logging
import os
import shutil
import traceback
import fnmatch
from asyncio import create_task
from contextlib import contextmanager
from typing import Optional

# Apply encoding patch BEFORE importing blar_graph
from app.modules.parsing.utils.encoding_patch import apply_encoding_patch

apply_encoding_patch()

from blar_graph.db_managers import Neo4jManager
from blar_graph.graph_construction.core.graph_builder import GraphConstructor
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
from app.modules.projects.projects_schema import ProjectStatusEnum
from app.modules.projects.projects_service import ProjectService
from app.modules.search.search_service import SearchService
from app.modules.utils.email_helper import EmailHelper
from app.modules.utils.parse_webhook_helper import ParseWebhookHelper
from app.modules.utils.posthog_helper import PostHogClient

from .parsing_schema import ParsingRequest, ParseFilters

logger = logging.getLogger(__name__)


class ParsingService:
    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.parse_helper = ParseHelper(db)
        self.project_service = ProjectService(db)
        self.inference_service = InferenceService(db, user_id)
        self.search_service = SearchService(db)
        self.github_service = CodeProviderService(db)

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
        project_manager = ProjectService(self.db)
        extracted_dir = None
        try:
            if cleanup_graph:
                neo4j_config = config_provider.get_neo4j_config()

                try:
                    code_graph_service = CodeGraphService(
                        neo4j_config["uri"],
                        neo4j_config["username"],
                        neo4j_config["password"],
                        self.db,
                    )

                    code_graph_service.cleanup_graph(project_id)
                except Exception as e:
                    logger.error(f"Error in cleanup_graph: {e}")
                    raise HTTPException(status_code=500, detail="Internal server error")

            repo, owner, auth = await self.parse_helper.clone_or_copy_repository(
                repo_details, user_id
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

            # Apply filters to the extracted directory
            if repo_details.filters:
                self.apply_filters_to_directory(extracted_dir, repo_details.filters)

            if isinstance(repo, Repo):
                language = self.parse_helper.detect_repo_language(extracted_dir)
            else:
                languages = repo.get_languages()
                if languages:
                    language = max(languages, key=languages.get).lower()
                else:
                    language = self.parse_helper.detect_repo_language(extracted_dir)

            await self.analyze_directory(
                extracted_dir,
                project_id,
                user_id,
                self.db,
                language,
                user_email,
                repo_details.filters,
            )
            message = "The project has been parsed successfully"
            return {"message": message, "id": project_id}

        except ParsingServiceError as e:
            message = str(f"{project_id} Failed during parsing: " + str(e))
            await project_manager.update_project_status(
                project_id, ProjectStatusEnum.ERROR
            )
            await ParseWebhookHelper().send_slack_notification(project_id, message)
            raise HTTPException(status_code=500, detail=message)

        except Exception as e:
            logger.error(f"Error during parsing for project {project_id}: {e}")
            # Rollback the database session to clear any pending transactions
            self.db.rollback()
            try:
                await project_manager.update_project_status(
                    project_id, ProjectStatusEnum.ERROR
                )
            except Exception as update_error:
                logger.error(
                    f"Failed to update project status after error: {update_error}"
                )
            await ParseWebhookHelper().send_slack_notification(project_id, str(e))
            tb_str = "".join(traceback.format_exception(None, e, e.__traceback__))
            raise HTTPException(
                status_code=500, detail=f"{str(e)}\nTraceback: {tb_str}"
            )

        finally:
            if (
                extracted_dir
                and isinstance(extracted_dir, str)
                and os.path.exists(extracted_dir)
                and extracted_dir.startswith(os.getenv("PROJECT_PATH"))
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

    def apply_filters_to_directory(self, directory: str, filters: ParseFilters):
        """
        Walks through the directory and removes files/folders that match the filters.
        """
        logger.info(f"Applying filters to directory: {directory}")

        # If no filters are set (empty lists and include_mode=False), don't skip anything
        if (
            not filters.excluded_directories
            and not filters.excluded_files
            and not filters.excluded_extensions
            and not filters.include_mode
        ):
            return

        for root, dirs, files in os.walk(directory, topdown=True):
            rel_root = os.path.relpath(root, directory)
            if rel_root == ".":
                rel_root = ""

            # Filter directories in-place to prevent walking into them
            # We iterate over a copy of dirs to safely modify it
            for d in list(dirs):
                dir_path = os.path.join(root, d)
                rel_path = os.path.join(rel_root, d)

                should_remove = False

                # Check directory exclusions
                path_parts = rel_path.split(os.sep)
                for excluded_dir in filters.excluded_directories:
                    if excluded_dir in path_parts:
                        should_remove = True
                        break

                # If include_mode is True, we might need to keep it if it's a parent of an included file?
                # But the spec says "If true, above become INCLUDE instead of EXCLUDE".
                # This is tricky for directories. If I include "src", I should keep "src".
                # If I include "src/utils", I should keep "src" and "src/utils".
                # For now, let's assume directory filtering is strict exclusion/inclusion.

                # Actually, for include_mode, we should probably only remove files that don't match.
                # Removing directories might be dangerous if they contain included files.
                # But if the directory ITSELF is excluded (in exclude mode), we remove it.

                if not filters.include_mode and should_remove:
                    shutil.rmtree(dir_path)
                    dirs.remove(d)
                    logger.info(f"Removed excluded directory: {rel_path}")

            # Filter files
            for f in files:
                file_path = os.path.join(root, f)
                rel_path = os.path.join(rel_root, f)

                matches_filter = False

                # Check directory exclusions (again, for the file's path)
                path_parts = rel_path.split(os.sep)
                dir_parts = path_parts[:-1]
                for excluded_dir in filters.excluded_directories:
                    if excluded_dir in dir_parts:
                        matches_filter = True
                        break

                if not matches_filter:
                    # Check extension exclusions
                    for ext in filters.excluded_extensions:
                        if not ext.startswith("."):
                            ext = "." + ext
                        if file_path.endswith(ext):
                            matches_filter = True
                            break

                if not matches_filter:
                    # Check file pattern exclusions (glob matching)
                    for excluded_pattern in filters.excluded_files:
                        if fnmatch.fnmatch(
                            rel_path, excluded_pattern
                        ) or fnmatch.fnmatch(f, excluded_pattern):
                            matches_filter = True
                            break

                should_remove = False
                if filters.include_mode:
                    # In include mode, remove if it DOES NOT match any filter
                    if not matches_filter:
                        should_remove = True
                else:
                    # In exclude mode, remove if it DOES match a filter
                    if matches_filter:
                        should_remove = True

                if should_remove:
                    try:
                        os.remove(file_path)
                        logger.info(f"Removed excluded file: {rel_path}")
                    except OSError as e:
                        logger.error(f"Error removing file {file_path}: {e}")

    async def analyze_directory(
        self,
        extracted_dir: str,
        project_id: int,
        user_id: str,
        db,
        language: str,
        user_email: str,
        filters: Optional[ParseFilters] = None,
    ):
        logger.info(
            f"ParsingService: Parsing project {project_id}: Analyzing directory: {extracted_dir}"
        )

        # Validate that extracted_dir is a valid path
        if not isinstance(extracted_dir, str):
            logger.error(
                f"ParsingService: Invalid extracted_dir type: {type(extracted_dir)}, value: {extracted_dir}"
            )
            raise ValueError(
                f"Expected string path, got {type(extracted_dir)}: {extracted_dir}"
            )

        if not os.path.exists(extracted_dir):
            logger.error(f"ParsingService: Directory does not exist: {extracted_dir}")
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
            logger.error(f"Project with ID {project_id} not found.")
            raise HTTPException(status_code=404, detail="Project not found.")

        if language in ["python", "javascript", "typescript"]:
            graph_manager = Neo4jManager(project_id, user_id)
            self.create_neo4j_indices(
                graph_manager
            )  # commented since indices are created already

            try:
                graph_constructor = GraphConstructor(user_id, extracted_dir)
                n, r = graph_constructor.build_graph()
                graph_manager.create_nodes(n)
                graph_manager.create_edges(r)
                await self.project_service.update_project_status(
                    project_id, ProjectStatusEnum.PARSED
                )
                PostHogClient().send_event(
                    user_id,
                    "project_status_event",
                    {"project_id": project_id, "status": "Parsed"},
                )

                # Generate docstrings using InferenceService
                await self.inference_service.run_inference(project_id)
                logger.info(f"DEBUGNEO4J: After inference project {project_id}")
                self.inference_service.log_graph_stats(project_id)
                await self.project_service.update_project_status(
                    project_id, ProjectStatusEnum.READY
                )
                create_task(
                    EmailHelper().send_email(user_email, repo_name, branch_name)
                )
                PostHogClient().send_event(
                    user_id,
                    "project_status_event",
                    {"project_id": project_id, "status": "Ready"},
                )
            except Exception as e:
                logger.error(e)
                logger.error(traceback.format_exc())
                await self.project_service.update_project_status(
                    project_id, ProjectStatusEnum.ERROR
                )
                await ParseWebhookHelper().send_slack_notification(project_id, str(e))
                PostHogClient().send_event(
                    user_id,
                    "project_status_event",
                    {"project_id": project_id, "status": "Error"},
                )
            finally:
                graph_manager.close()
        elif language != "other":
            try:
                neo4j_config = config_provider.get_neo4j_config()
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
                await self.inference_service.run_inference(project_id)
                logger.info(f"DEBUGNEO4J: After inference project {project_id}")
                self.inference_service.log_graph_stats(project_id)
                await self.project_service.update_project_status(
                    project_id, ProjectStatusEnum.READY
                )
                create_task(
                    EmailHelper().send_email(user_email, repo_name, branch_name)
                )
                logger.info(f"DEBUGNEO4J: After update project status {project_id}")
                self.inference_service.log_graph_stats(project_id)
            finally:
                service.close()
                logger.info(f"DEBUGNEO4J: After close service {project_id}")
                self.inference_service.log_graph_stats(project_id)
        else:
            await self.project_service.update_project_status(
                project_id, ProjectStatusEnum.ERROR
            )
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

        except Exception as e:
            logger.error(
                f"Error duplicating graph from {old_repo_id} to {new_repo_id}: {e}"
            )

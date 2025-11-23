import logging
import os
import shutil
import traceback
from asyncio import create_task
from contextlib import contextmanager

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

from .parsing_schema import ParsingRequest

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
        logger.info(f"ParsingService: parse_directory called for project_id={project_id}, user_id={user_id}")
        project_manager = ProjectService(self.db)
        extracted_dir = None
        try:
            if cleanup_graph:
                logger.info(f"ParsingService: Cleanup graph enabled for project {project_id}")
                neo4j_config = config_provider.get_neo4j_config()

                try:
                    logger.info(f"ParsingService: Initializing CodeGraphService for cleanup")
                    code_graph_service = CodeGraphService(
                        neo4j_config["uri"],
                        neo4j_config["username"],
                        neo4j_config["password"],
                        self.db,
                    )

                    logger.info(f"ParsingService: Cleaning up graph for project {project_id}")
                    code_graph_service.cleanup_graph(project_id)
                    logger.info(f"ParsingService: Graph cleanup completed for project {project_id}")
                except Exception as e:
                    logger.error(f"ParsingService: Error in cleanup_graph: {e}")
                    logger.exception("ParsingService: Cleanup exception details:")
                    raise HTTPException(status_code=500, detail="Internal server error")

            logger.info(f"ParsingService: Cloning or copying repository for project {project_id}")
            repo, owner, auth = await self.parse_helper.clone_or_copy_repository(
                repo_details, user_id
            )
            logger.info(f"ParsingService: Repository cloned/copied successfully")
            
            is_dev_mode = config_provider.get_is_development_mode()
            logger.info(f"ParsingService: Development mode={is_dev_mode}")
            
            if is_dev_mode:
                logger.info(f"ParsingService: Setting up project directory (dev mode)")
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
                logger.info(f"ParsingService: Project directory setup completed (dev mode): {extracted_dir}")
            else:
                logger.info(f"ParsingService: Setting up project directory (prod mode)")
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
                logger.info(f"ParsingService: Project directory setup completed (prod mode): {extracted_dir}")

            logger.info(f"ParsingService: Detecting repository language")
            logger.info(f"ParsingService: Repo type={type(repo).__name__}, extracted_dir={extracted_dir}")
            
            if isinstance(repo, Repo):
                logger.info(f"ParsingService: Repo is a Git Repo instance (local or GitBucket), detecting from files")
                try:
                    language = self.parse_helper.detect_repo_language(extracted_dir)
                    logger.info(f"ParsingService: Detected language from files: {language}")
                except Exception as lang_error:
                    logger.error(f"ParsingService: Language detection from files failed: {lang_error}")
                    logger.exception("ParsingService: Language detection exception details:")
                    language = "other"
                    logger.warning(f"ParsingService: Defaulting to language='other' due to detection failure")
            else:
                logger.info(f"ParsingService: Repo is a GitHub/API instance, attempting to get languages from API")
                try:
                    languages = repo.get_languages()
                    logger.info(f"ParsingService: Available languages from API: {languages}")
                    if languages:
                        language = max(languages, key=languages.get).lower()
                        logger.info(f"ParsingService: Primary language from API: {language}")
                    else:
                        logger.warning(f"ParsingService: No languages from API, falling back to file detection")
                        try:
                            language = self.parse_helper.detect_repo_language(extracted_dir)
                            logger.info(f"ParsingService: Detected language from files (fallback): {language}")
                        except Exception as lang_error:
                            logger.error(f"ParsingService: File-based language detection failed: {lang_error}")
                            logger.exception("ParsingService: Language detection exception details:")
                            language = "other"
                            logger.warning(f"ParsingService: Defaulting to language='other' due to detection failure")
                except Exception as api_error:
                    logger.error(f"ParsingService: API language detection failed: {api_error}")
                    logger.exception("ParsingService: API language detection exception details:")
                    logger.warning(f"ParsingService: Falling back to file-based detection")
                    try:
                        language = self.parse_helper.detect_repo_language(extracted_dir)
                        logger.info(f"ParsingService: Detected language from files (API fallback): {language}")
                    except Exception as lang_error:
                        logger.error(f"ParsingService: File-based language detection also failed: {lang_error}")
                        logger.exception("ParsingService: Language detection exception details:")
                        language = "other"
                        logger.warning(f"ParsingService: Defaulting to language='other' due to all detection failures")

            logger.info(f"ParsingService: Calling analyze_directory for project {project_id}")
            logger.info(f"ParsingService: analyze_directory parameters - extracted_dir={extracted_dir}, project_id={project_id}, user_id={user_id}, language={language}")
            await self.analyze_directory(
                extracted_dir, project_id, user_id, self.db, language, user_email
            )
            logger.info(f"ParsingService: analyze_directory completed successfully for project {project_id}")
            message = "The project has been parsed successfully"
            return {"message": message, "id": project_id}

        except ParsingServiceError as e:
            message = str(f"{project_id} Failed during parsing: " + str(e))
            logger.error(f"ParsingService: ParsingServiceError caught: {message}")
            logger.exception("ParsingService: ParsingServiceError details:")
            await project_manager.update_project_status(
                project_id, ProjectStatusEnum.ERROR
            )
            await ParseWebhookHelper().send_slack_notification(project_id, message)
            raise HTTPException(status_code=500, detail=message)

        except Exception as e:
            logger.error(f"ParsingService: Exception during parsing for project {project_id}: {e}")
            logger.exception("ParsingService: Exception details:")
            # Rollback the database session to clear any pending transactions
            self.db.rollback()
            logger.info(f"ParsingService: Database session rolled back")
            try:
                logger.info(f"ParsingService: Attempting to update project status to ERROR")
                await project_manager.update_project_status(
                    project_id, ProjectStatusEnum.ERROR
                )
                logger.info(f"ParsingService: Project status updated to ERROR")
            except Exception as update_error:
                logger.error(
                    f"ParsingService: Failed to update project status after error: {update_error}"
                )
                logger.exception("ParsingService: Status update exception details:")
            
            await ParseWebhookHelper().send_slack_notification(project_id, str(e))
            tb_str = "".join(traceback.format_exception(None, e, e.__traceback__))
            logger.error(f"ParsingService: Full traceback: {tb_str}")
            raise HTTPException(
                status_code=500, detail=f"{str(e)}\nTraceback: {tb_str}"
            )

        finally:
            logger.info(f"ParsingService: Entering finally block for project {project_id}")
            if (
                extracted_dir
                and isinstance(extracted_dir, str)
                and os.path.exists(extracted_dir)
                and extracted_dir.startswith(os.getenv("PROJECT_PATH"))
            ):
                logger.info(f"ParsingService: Cleaning up extracted directory: {extracted_dir}")
                shutil.rmtree(extracted_dir, ignore_errors=True)
                logger.info(f"ParsingService: Extracted directory cleaned up successfully")
            else:
                logger.info(f"ParsingService: No directory cleanup needed. extracted_dir={extracted_dir}")
            logger.info(f"ParsingService: parse_directory completed for project {project_id}")

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
        logger.debug(f"ParsingService: Parameters - project_id={project_id}, user_id={user_id}, language={language}")

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

        if not os.path.isdir(extracted_dir):
            logger.error(f"ParsingService: Path is not a directory: {extracted_dir}")
            raise NotADirectoryError(f"Path is not a directory: {extracted_dir}")

        # Check if directory is accessible and list contents
        try:
            contents = os.listdir(extracted_dir)
            logger.info(f"ParsingService: Directory contains {len(contents)} items")
            logger.debug(f"ParsingService: Directory contents (first 20): {contents[:20]}")
        except Exception as e:
            logger.error(f"ParsingService: Cannot list directory contents: {e}")
            raise

        logger.info(
            f"ParsingService: Directory exists and is accessible: {extracted_dir}"
        )
        
        logger.info(f"ParsingService: Fetching project details for project_id={project_id}")
        try:
            project_details = await self.project_service.get_project_from_db_by_id(
                project_id
            )
            logger.info(f"ParsingService: Project details retrieved successfully")
        except Exception as e:
            logger.error(f"ParsingService: Failed to fetch project details: {e}")
            logger.exception("ParsingService: Exception details:")
            raise
        
        if project_details:
            repo_name = project_details.get("project_name")
            branch_name = project_details.get("branch_name")
            logger.info(f"ParsingService: Project name={repo_name}, branch={branch_name}")
        else:
            logger.error(f"ParsingService: Project with ID {project_id} not found.")
            raise HTTPException(status_code=404, detail="Project not found.")

        logger.info(f"ParsingService: Detected language={language}, determining parsing path")
        if language in ["python", "javascript", "typescript"]:
            logger.info(f"ParsingService: Processing python/javascript/typescript project with language={language}")
            logger.info(f"ParsingService: Initializing Neo4jManager for project_id={project_id}, user_id={user_id}")
            try:
                graph_manager = Neo4jManager(project_id, user_id)
                logger.info(f"ParsingService: Neo4jManager initialized successfully")
            except Exception as e:
                logger.error(f"ParsingService: Failed to initialize Neo4jManager: {e}")
                logger.exception("ParsingService: Exception details:")
                raise
            
            logger.info(f"ParsingService: Creating Neo4j indices")
            try:
                self.create_neo4j_indices(
                    graph_manager
                )  # commented since indices are created already
                logger.info(f"ParsingService: Neo4j indices created successfully")
            except Exception as e:
                logger.error(f"ParsingService: Failed to create Neo4j indices: {e}")
                logger.exception("ParsingService: Exception details:")
                raise

            try:
                logger.info(f"ParsingService: Initializing GraphConstructor with extracted_dir={extracted_dir}")
                graph_constructor = GraphConstructor(user_id, extracted_dir)
                logger.info(f"ParsingService: GraphConstructor initialized, building graph")
                
                n, r = graph_constructor.build_graph()
                logger.info(f"ParsingService: Graph built successfully with {len(n) if n else 0} nodes and {len(r) if r else 0} relationships")
                
                logger.info(f"ParsingService: Creating nodes in Neo4j")
                graph_manager.create_nodes(n)
                logger.info(f"ParsingService: Nodes created successfully")
                
                logger.info(f"ParsingService: Creating edges in Neo4j")
                graph_manager.create_edges(r)
                logger.info(f"ParsingService: Edges created successfully")
                
                logger.info(f"ParsingService: Updating project status to PARSED")
                await self.project_service.update_project_status(
                    project_id, ProjectStatusEnum.PARSED
                )
                logger.info(f"ParsingService: Project status updated to PARSED")
                
                PostHogClient().send_event(
                    user_id,
                    "project_status_event",
                    {"project_id": project_id, "status": "Parsed"},
                )

                # Check if inference is enabled via environment variable
                enable_inference = os.getenv("ENABLE_INFERENCE", "false").lower() == "true"

                if enable_inference:
                    # Generate docstrings using InferenceService
                    logger.info(f"ParsingService: Starting inference for project {project_id}")
                    await self.inference_service.run_inference(project_id)
                    logger.info(f"ParsingService: Inference completed for project {project_id}")
                    logger.info(f"DEBUGNEO4J: After inference project {project_id}")
                    self.inference_service.log_graph_stats(project_id)
                else:
                    logger.info(f"ParsingService: Skipping inference for project {project_id} (ENABLE_INFERENCE=false)")

                logger.info(f"ParsingService: Updating project status to READY")
                await self.project_service.update_project_status(
                    project_id, ProjectStatusEnum.READY
                )
                logger.info(f"ParsingService: Project status updated to READY")
                
                create_task(
                    EmailHelper().send_email(user_email, repo_name, branch_name)
                )
                PostHogClient().send_event(
                    user_id,
                    "project_status_event",
                    {"project_id": project_id, "status": "Ready"},
                )
                logger.info(f"ParsingService: Python/JS/TS parsing flow completed successfully for project {project_id}")
            except Exception as e:
                logger.error(f"ParsingService: Error during python/js/ts parsing for project {project_id}: {e}")
                logger.error(traceback.format_exc())
                logger.info(f"ParsingService: Updating project status to ERROR due to exception")
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
                logger.info(f"ParsingService: Closing graph_manager connection")
                graph_manager.close()
                logger.info(f"ParsingService: graph_manager closed successfully")
        elif language != "other":
            logger.info(f"ParsingService: Processing non-python/js/ts project with language: {language}")
            try:
                logger.info(f"ParsingService: Retrieving Neo4j config")
                neo4j_config = config_provider.get_neo4j_config()
                logger.debug(f"ParsingService: Neo4j config retrieved: uri={neo4j_config.get('uri', 'N/A')}")
                
                logger.info(f"ParsingService: Initializing CodeGraphService")
                service = CodeGraphService(
                    neo4j_config["uri"],
                    neo4j_config["username"],
                    neo4j_config["password"],
                    db,
                )
                logger.info(f"ParsingService: CodeGraphService initialized successfully")

                logger.info(f"ParsingService: Calling create_and_store_graph for project {project_id}")
                logger.info(f"ParsingService: Arguments - extracted_dir={extracted_dir}, project_id={project_id}, user_id={user_id}")
                
                try:
                    service.create_and_store_graph(extracted_dir, project_id, user_id)
                    logger.info(f"ParsingService: create_and_store_graph completed successfully for project {project_id}")
                except Exception as graph_error:
                    logger.error(f"ParsingService: create_and_store_graph failed for project {project_id}: {graph_error}")
                    logger.exception("ParsingService: create_and_store_graph exception details:")
                    raise

                logger.info(f"ParsingService: Updating project status to PARSED")
                await self.project_service.update_project_status(
                    project_id, ProjectStatusEnum.PARSED
                )
                logger.info(f"ParsingService: Project status updated to PARSED")

                # Check if inference is enabled via environment variable
                enable_inference = os.getenv("ENABLE_INFERENCE", "false").lower() == "true"

                if enable_inference:
                    # Generate docstrings using InferenceService
                    logger.info(f"ParsingService: Starting inference for project {project_id}")
                    await self.inference_service.run_inference(project_id)
                    logger.info(f"ParsingService: Inference completed for project {project_id}")
                    logger.info(f"DEBUGNEO4J: After inference project {project_id}")
                    self.inference_service.log_graph_stats(project_id)
                else:
                    logger.info(f"ParsingService: Skipping inference for project {project_id} (ENABLE_INFERENCE=false)")

                logger.info(f"ParsingService: Updating project status to READY")
                await self.project_service.update_project_status(
                    project_id, ProjectStatusEnum.READY
                )
                logger.info(f"ParsingService: Project status updated to READY")
                
                create_task(
                    EmailHelper().send_email(user_email, repo_name, branch_name)
                )
                logger.info(f"DEBUGNEO4J: After update project status {project_id}")
                self.inference_service.log_graph_stats(project_id)
                logger.info(f"ParsingService: Non-python/js/ts parsing flow completed successfully for project {project_id}")
            except Exception as e:
                logger.error(f"ParsingService: Error during non-python/js/ts parsing for project {project_id}: {e}")
                logger.exception("ParsingService: Exception details:")
                raise
            finally:
                logger.info(f"ParsingService: Closing CodeGraphService connection")
                service.close()
                logger.info(f"ParsingService: CodeGraphService closed successfully")
                logger.info(f"DEBUGNEO4J: After close service {project_id}")
                self.inference_service.log_graph_stats(project_id)
        else:
            logger.warning(f"ParsingService: Language 'other' detected for project {project_id}, marking as ERROR")
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

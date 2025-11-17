import logging
import os
import shutil
import traceback
from asyncio import create_task
from contextlib import contextmanager

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

            supported_languages = {
                "c_sharp",
                "c",
                "cpp",
                "elisp",
                "elixir",
                "elm",
                "go",
                "java",
                "javascript",
                "ocaml",
                "php",
                "python",
                "ql",
                "ruby",
                "rust",
                "typescript",
            }

            # Always use file-based detection as the primary method (counts files, excludes "other")
            # This is more reliable than percentage-based detection from GitHub API
            file_based_language = self.parse_helper.detect_repo_language(extracted_dir)
            logger.info(f"File-based language detection result: {file_based_language}")

            # Collect all detected languages from GitHub API if available (for LSP indexing)
            detected_languages = []
            if not isinstance(repo, Repo):
                languages = repo.get_languages()
                logger.info(f"GitHub API languages: {languages}")
                if languages:
                    # Get all languages with significant code (e.g., > 5% of codebase)
                    # Filter out "other" and unsupported languages
                    detected_languages = [
                        lang.lower()
                        for lang, percentage in languages.items()
                        if percentage > 5 and lang.lower() in supported_languages
                    ]
                    logger.info(
                        f"Filtered detected languages from API: {detected_languages}"
                    )

            # For local repos or when API doesn't provide languages, use file-based detection
            # Add file-based language to detected_languages if it's valid and not already included
            if (
                file_based_language
                and file_based_language != "other"
                and file_based_language in supported_languages
            ):
                if file_based_language not in detected_languages:
                    detected_languages.append(file_based_language)
                    logger.info(
                        f"Added file-based language '{file_based_language}' to detected_languages for LSP indexing"
                    )

            # Use file-based language detection result, which already excludes "other"
            if (
                file_based_language
                and file_based_language != "other"
                and file_based_language in supported_languages
            ):
                language = file_based_language
                logger.info(f"Using file-based language detection: {language}")
            elif detected_languages:
                # Fallback to first detected language from GitHub API if file-based detection failed
                language = detected_languages[0]
                logger.info(
                    f"Using first detected language from API as fallback: {language}"
                )
            else:
                # Last resort: try to find any supported language in the repo
                language = file_based_language if file_based_language else "other"
                logger.warning(
                    f"Could not detect a supported language, using: {language}"
                )
                logger.warning(
                    f"File-based result: {file_based_language}, API languages: {detected_languages}"
                )

            # Run graph construction and inference first
            await self.analyze_directory(
                extracted_dir, project_id, user_id, self.db, language, user_email
            )

            # Get the worktree path from RepoManager for LSP indexing
            # This ensures we index the persistent worktree, not a temporary directory
            # LSP indexing happens after inference to ensure the worktree is fully set up
            worktree_path_for_indexing = extracted_dir
            if self.parse_helper.repo_manager:
                import asyncio

                # Wait a bit for worktree to be fully created and registered
                await asyncio.sleep(0.5)

                try:
                    details = await self.project_service.get_project_from_db_by_id(
                        project_id
                    )
                    if details:
                        repo_name = details.get("project_name")
                        branch = details.get("branch_name")
                        commit_id = details.get("commit_id")

                        # Try to get worktree path with retries
                        worktree_path = None
                        for attempt in range(3):
                            worktree_path = (
                                self.parse_helper.repo_manager.get_repo_path(
                                    repo_name, branch=branch, commit_id=commit_id
                                )
                            )
                            if worktree_path and os.path.exists(worktree_path):
                                break
                            if attempt < 2:
                                await asyncio.sleep(0.5)  # Wait and retry

                        if worktree_path and os.path.exists(worktree_path):
                            worktree_path_for_indexing = worktree_path
                            logger.info(
                                f"[LSP] Using worktree path from RepoManager for indexing: {worktree_path}"
                            )
                        else:
                            logger.warning(
                                f"[LSP] Worktree not found in RepoManager after retries, "
                                f"using extracted_dir: {extracted_dir}. "
                                f"Repo: {repo_name}, Branch: {branch}, Commit: {commit_id}"
                            )
                except Exception as exc:
                    logger.warning(
                        f"[LSP] Failed to get worktree path from RepoManager, using extracted_dir: {exc}",
                        exc_info=True,
                    )

            # Index workspace with LSP servers after inference
            # This must succeed for parsing to be considered successful
            logger.info(
                f"[LSP] Starting LSP indexing for project {project_id} "
                f"at workspace {worktree_path_for_indexing} with languages {detected_languages}"
            )
            logger.info(
                f"[LSP] Waiting for LSP indexing to complete for project {project_id}..."
            )
            indexing_success = await self._index_workspace_with_lsp(
                project_id, worktree_path_for_indexing, detected_languages
            )
            logger.info(
                f"[LSP] LSP indexing completed for project {project_id}: success={indexing_success}"
            )
            if not indexing_success:
                error_msg = (
                    f"LSP indexing failed for project {project_id}. "
                    "Parsing cannot complete without successful indexing."
                )
                await project_manager.update_project_status(
                    project_id, ProjectStatusEnum.ERROR
                )
                await ParseWebhookHelper().send_slack_notification(
                    project_id, error_msg
                )
                raise HTTPException(status_code=500, detail=error_msg)

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

    async def _index_workspace_with_lsp(
        self, project_id: int, workspace_root: str, languages: list
    ) -> bool:
        """
        Index the workspace with LSP servers for the detected languages.

        This pre-indexes the workspace so that LSP queries are fast when users
        start chatting. This is a required step - parsing will fail if indexing fails.

        Returns:
            True if indexing succeeded (or was skipped for valid reasons), False otherwise
        """
        try:
            # Check if repo manager is enabled
            repo_manager_enabled = (
                os.getenv("REPO_MANAGER_ENABLED", "false").lower() == "true"
            )
            if not repo_manager_enabled:
                logger.info(
                    f"[LSP] Skipping LSP indexing for project {project_id}: "
                    "REPO_MANAGER_ENABLED is not set"
                )
                # If repo manager is disabled, we can't index, but this is acceptable
                return True

            # Import here to avoid circular dependencies
            from app.modules.intelligence.tools.code_query_tools.lsp_server_manager import (
                get_lsp_server_manager,
            )
            from app.modules.intelligence.tools.code_query_tools.lsp_query_tool import (
                LspQueryTool,
            )

            # Initialize LSP query tool to configure language servers
            # This ensures language servers are registered
            lsp_tool = LspQueryTool(self.db, "")
            if not lsp_tool.repo_manager:
                logger.warning(
                    f"[LSP] LSP indexing failed for project {project_id}: "
                    "RepoManager not available"
                )
                return False

            # Get the LSP server manager
            server_manager = get_lsp_server_manager()

            # Map parsing language names to LSP language identifiers
            # Some languages use different names in parsing vs LSP (e.g., c_sharp -> csharp)
            language_name_map = {
                "c_sharp": "csharp",
            }

            # Convert language names to LSP identifiers
            # Filter out None values to ensure type safety
            lsp_languages = [
                language_name_map.get(lang, lang)
                for lang in languages
                if lang is not None
            ]

            # Filter to only languages that have LSP servers configured
            supported_languages = [
                lang
                for lang in lsp_languages
                if lang and server_manager.is_language_registered(lang)
            ]

            if not supported_languages:
                logger.info(
                    f"[LSP] No supported LSP languages found for project {project_id} "
                    f"(detected: {languages}). Skipping LSP indexing."
                )
                # If no supported languages, this is acceptable (e.g., unsupported language)
                return True

            logger.info(
                f"[LSP] Starting LSP indexing for project {project_id} "
                f"with languages: {supported_languages} at workspace: {workspace_root}"
            )

            # Index the workspace
            logger.info(f"[LSP] Calling index_workspace for project {project_id}...")
            results = await server_manager.index_workspace(
                project_id=str(project_id),
                workspace_root=workspace_root,
                languages=supported_languages,
            )

            # Check if all languages were indexed successfully
            all_succeeded = True
            failed_languages = []

            for lang, result in results.items():
                status_messages = result.get("status_messages", [])
                if result.get("success"):
                    logger.info(
                        f"[LSP] Successfully indexed {lang} for project {project_id}"
                    )
                    # Log all status messages for detailed progress
                    for msg in status_messages:
                        logger.info(f"[LSP] [{lang}] {msg}")
                else:
                    all_succeeded = False
                    error_msg = result.get("error", "Unknown error")
                    failed_languages.append(f"{lang}: {error_msg}")
                    logger.error(
                        f"[LSP] Failed to index {lang} for project {project_id}: {error_msg}"
                    )
                    # Log status messages even for failures to help debug
                    for msg in status_messages:
                        logger.warning(f"[LSP] [{lang}] {msg}")

            if not all_succeeded:
                logger.error(
                    f"[LSP] LSP indexing failed for project {project_id}. "
                    f"Failed languages: {', '.join(failed_languages)}"
                )
                return False

            logger.info(
                f"[LSP] Successfully completed LSP indexing for project {project_id} "
                f"with all languages: {supported_languages}"
            )
            return True

        except Exception as exc:
            logger.error(
                f"[LSP] LSP indexing failed for project {project_id}: {exc}",
                exc_info=True,
            )
            return False

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

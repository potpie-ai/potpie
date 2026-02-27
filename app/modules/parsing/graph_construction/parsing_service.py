import asyncio
import os
import time
import traceback
from asyncio import create_task
from contextlib import contextmanager

from fastapi import HTTPException
from git import Repo
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.code_provider.github.github_service import GithubService
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
from app.modules.repo_manager import RepoManager
from app.modules.utils.email_helper import EmailHelper
from app.modules.utils.logger import log_context, setup_logger
from app.modules.utils.parse_webhook_helper import ParseWebhookHelper

from .parsing_schema import ParsingRequest, RepoDetails

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
        self.repo_manager = RepoManager()

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
        project_id: str,
        cleanup_graph: bool = True,
    ):
        # Set up logging context with domain IDs
        with log_context(project_id=str(project_id), user_id=user_id):
            project_manager = ProjectService(self.db)
            extracted_dir = None
            try:
                # Early check: if project is already inferring, return without re-running (avoids duplicate work and status update errors)
                existing_project = await project_manager.get_project_from_db_by_id(
                    project_id
                )
                if (
                    existing_project
                    and existing_project.get("status")
                    == ProjectStatusEnum.INFERRING.value
                ):
                    logger.info(
                        "Skipping parse for project %s - already in inferring state",
                        project_id,
                    )
                    return {
                        "message": "Project already inferring",
                        "id": project_id,
                        "status": ProjectStatusEnum.INFERRING.value,
                    }

                # Early check: if project already exists and is READY for this commit, skip parsing
                if cleanup_graph and repo_details.commit_id and existing_project:
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
                        # Ensure worktree exists in repo manager when enabled
                        if self.repo_manager and os.getenv(
                            "REPO_MANAGER_ENABLED", "false"
                        ).lower() == "true":
                            repo_name = existing_project.get("project_name")
                            branch = existing_project.get("branch_name")
                            commit_id_val = existing_project.get("commit_id")
                            repo_path = existing_project.get("repo_path")
                            if repo_name and not repo_path:
                                ref = commit_id_val if commit_id_val else branch
                                if ref:
                                    try:
                                        github_service = GithubService(self.db)
                                        user_token = github_service.get_github_oauth_token(user_id)
                                        loop = asyncio.get_event_loop()
                                        await loop.run_in_executor(
                                            None,
                                            lambda: self.repo_manager.prepare_for_parsing(
                                                repo_name,
                                                ref,
                                                auth_token=user_token,
                                                is_commit=bool(commit_id_val),
                                                user_id=user_id,
                                            ),
                                        )
                                        logger.info(
                                            "Ensured worktree for already-parsed project %s (%s@%s)",
                                            project_id,
                                            repo_name,
                                            ref,
                                        )
                                    except Exception:
                                        logger.warning(
                                            "Failed to ensure worktree for project %s",
                                            project_id,
                                            exc_info=True,
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
                logger.info(
                    "ParsingService: About to call clone_or_copy_repository",
                    repo_name=repo_details.repo_name,
                    repo_path=repo_details.repo_path,
                    project_id=project_id,
                )
                # Fetch user's GitHub OAuth token from user_auth_providers table
                user_token = None
                try:
                    github_service = GithubService(self.db)
                    user_token = github_service.get_github_oauth_token(user_id)
                    if user_token:
                        logger.info(
                            "Using user's GitHub OAuth token for cloning",
                            user_id=user_id,
                            repo_name=repo_details.repo_name,
                            token_prefix=user_token[:8] if len(user_token) > 8 else "short",
                        )
                    else:
                        logger.warning(
                            "No user GitHub OAuth token found - will use environment tokens",
                            user_id=user_id,
                            repo_name=repo_details.repo_name,
                            reason="User may not have linked GitHub account or token expired",
                        )
                except Exception as e:
                    logger.exception(
                        "Failed to fetch user GitHub token - falling back to environment tokens",
                        user_id=user_id,
                        repo_name=repo_details.repo_name,
                        error=str(e),
                    )
                (
                    repo,
                    _owner,
                    auth,
                    repo_manager_path,
                ) = await self.parse_helper.clone_or_copy_repository(
                    repo_details_converted, user_id, auth_token=user_token, project_id=str(project_id)
                )
                logger.info(
                    "ParsingService: clone_or_copy_repository completed",
                    repo_name=repo_details.repo_name,
                    project_id=project_id,
                    repo_manager_path=repo_manager_path,
                )
                if config_provider.get_is_development_mode():
                    (
                        extracted_dir,
                        returned_project_id,
                    ) = await self.parse_helper.setup_project_directory(
                        repo,
                        repo_details.branch_name,
                        auth,
                        repo_details,
                        user_id,
                        str(project_id),
                        commit_id=repo_details.commit_id,
                        repo_manager_path=repo_manager_path,
                        auth_token=user_token,
                    )
                else:
                    (
                        extracted_dir,
                        returned_project_id,
                    ) = await self.parse_helper.setup_project_directory(
                        repo,
                        repo_details.branch_name,
                        auth,
                        repo_details,
                        user_id,
                        str(project_id),
                        commit_id=repo_details.commit_id,
                        repo_manager_path=repo_manager_path,
                        auth_token=user_token,
                    )

                # setup_project_directory returns str | None, but project_id is int
                # Keep original project_id since it's already set and used as int throughout
                # The returned value is only used for logging/debugging

                # Ensure extracted_dir is a string
                if extracted_dir is None:
                    if self._raise_library_exceptions:
                        raise ParsingServiceError(
                            "Failed to set up project directory"
                        )
                    raise HTTPException(
                        status_code=500, detail="Failed to set up project directory"
                    )
                extracted_dir = str(extracted_dir)

                if repo is None or isinstance(repo, Repo):
                    # Local repo or cached repo without GitHub API access
                    # Use local language detection
                    # Always use extracted_dir as it's the canonical path for parsing
                    # (repo_manager_path is set to extracted_dir when using RepoManager)
                    language_dir = extracted_dir
                    logger.info(
                        f"Using extracted_dir for language detection: {language_dir} "
                        f"(repo_manager_path: {repo_manager_path}, "
                        f"repo.working_tree_dir: {repo.working_tree_dir if isinstance(repo, Repo) else 'N/A'})"
                    )
                    language = self.parse_helper.detect_repo_language(language_dir)
                    logger.info(f"Detected language: {language}")
                else:
                    # PyGithub Repository object - use API for language detection
                    languages = repo.get_languages()
                    if languages:
                        language = max(languages, key=languages.get).lower()
                        logger.info(f"Detected language from GitHub API: {language}")
                    else:
                        logger.info(
                            "GitHub API returned no languages, using local detection"
                        )
                        language = self.parse_helper.detect_repo_language(extracted_dir)
                        logger.info(
                            f"Detected language from local detection: {language}"
                        )

                logger.info(
                    "ParsingService: About to analyze directory",
                    extra={
                        "extracted_dir": extracted_dir,
                        "repo_manager_path": repo_manager_path,
                        "repo_type": type(repo).__name__ if repo else "None",
                        "repo_working_tree_dir": (
                            repo.working_tree_dir if isinstance(repo, Repo) else "N/A"
                        ),
                        "language": language,
                    },
                )
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
                logger.exception(
                    "Error during parsing",
                    project_id=project_id,
                    user_id=user_id,
                )
                # Log the formatted traceback as extra to avoid format-placeholder issues in message
                logger.error(
                    "Full traceback (see full_traceback extra)",
                    full_traceback=traceback.format_exc(),
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
        project_id: str,
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
            "ParsingService: Directory exists and is accessible", dir=extracted_dir
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

        analysis_start_time = time.time()
        logger.info(
            f"[PARSING] Starting analysis for project {project_id} (language: {language})",
            project_id=project_id,
            user_id=user_id,
            language=language,
        )

        service = None
        if language != "other":
            try:
                # Step 1: Graph Generation
                graph_gen_start = time.time()
                logger.info(
                    "[PARSING] Step 1/3: Graph generation",
                    project_id=project_id,
                )
                neo4j_config = self._get_neo4j_config()
                service = CodeGraphService(
                    neo4j_config["uri"],
                    neo4j_config["username"],
                    neo4j_config["password"],
                    db,
                )

                service.create_and_store_graph(extracted_dir, project_id, user_id)
                graph_gen_time = time.time() - graph_gen_start
                logger.info(
                    f"[PARSING] Graph generation completed in {graph_gen_time:.2f}s",
                    project_id=project_id,
                    graph_gen_time_seconds=graph_gen_time,
                )

                status_update_start = time.time()
                await self.project_service.update_project_status(
                    project_id, ProjectStatusEnum.PARSED
                )
                status_update_time = time.time() - status_update_start

                # Step 2: Inference
                inference_start = time.time()
                logger.info(
                    "[PARSING] Step 2/3: Running inference",
                    project_id=project_id,
                )
                cache_stats = await self.inference_service.run_inference(
                    str(project_id)
                )
                inference_time = time.time() - inference_start
                logger.info(
                    f"[PARSING] Inference completed in {inference_time:.2f}s",
                    project_id=project_id,
                    inference_time_seconds=inference_time,
                )
                self.inference_service.log_graph_stats(project_id)

                # Step 3: Final status update
                final_status_start = time.time()
                await self.project_service.update_project_status(
                    project_id, ProjectStatusEnum.READY
                )
                final_status_time = time.time() - final_status_start

                if not self._raise_library_exceptions and user_email:
                    task = create_task(
                        EmailHelper().send_email(user_email, repo_name, branch_name)
                    )

                    def _on_email_done(t: asyncio.Task) -> None:
                        if t.cancelled():
                            return
                        try:
                            exc = t.exception()
                        except asyncio.CancelledError:
                            return
                        if exc is not None:
                            logger.exception("Failed to send email", exc_info=exc)

                    task.add_done_callback(_on_email_done)

                total_analysis_time = time.time() - analysis_start_time

                # Log cache statistics if available
                cache_hit_rate = 0.0
                if cache_stats and isinstance(cache_stats, dict):
                    total_cacheable = cache_stats.get(
                        "cache_hits", 0
                    ) + cache_stats.get("cache_misses", 0)
                    if total_cacheable > 0:
                        cache_hit_rate = (
                            cache_stats.get("cache_hits", 0) / total_cacheable
                        ) * 100

                    logger.info(
                        f"[PARSING] Cache statistics - "
                        f"Hits: {cache_stats.get('cache_hits', 0)} ({cache_stats.get('cache_hit_rate', 0):.1f}%), "
                        f"Misses: {cache_stats.get('cache_misses', 0)} ({cache_stats.get('cache_miss_rate', 0):.1f}%), "
                        f"Uncacheable: {cache_stats.get('uncacheable_nodes', 0)}, "
                        f"Stored: {cache_stats.get('cache_stored', 0)}, "
                        f"Cache hit rate (cacheable only): {cache_hit_rate:.1f}%",
                        project_id=project_id,
                        cache_hits=cache_stats.get("cache_hits", 0),
                        cache_misses=cache_stats.get("cache_misses", 0),
                        uncacheable_nodes=cache_stats.get("uncacheable_nodes", 0),
                        cache_stored=cache_stats.get("cache_stored", 0),
                        cache_hit_rate=cache_stats.get("cache_hit_rate", 0),
                        cache_miss_rate=cache_stats.get("cache_miss_rate", 0),
                        cache_hit_rate_cacheable_only=cache_hit_rate,
                    )

                logger.info(
                    f"[PARSING] Analysis completed in {total_analysis_time:.2f}s: "
                    f"Graph gen: {graph_gen_time:.2f}s, "
                    f"Inference: {inference_time:.2f}s, "
                    f"Status updates: {status_update_time + final_status_time:.2f}s",
                    project_id=project_id,
                    total_analysis_time_seconds=total_analysis_time,
                    graph_gen_time_seconds=graph_gen_time,
                    inference_time_seconds=inference_time,
                    status_update_time_seconds=status_update_time + final_status_time,
                )
                logger.info(f"DEBUGNEO4J: After update project status {project_id}")
                self.inference_service.log_graph_stats(project_id)
            finally:
                if service is not None:
                    service.close()
                logger.info(
                    "[PARSING] Cleaned up graph service",
                    project_id=project_id,
                )
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

        except Exception as e:
            logger.exception(
                "Error duplicating graph",
                old_repo_id=old_repo_id,
                new_repo_id=new_repo_id,
            )
            raise ParsingServiceError(
                f"Failed to duplicate graph from {old_repo_id} to {new_repo_id}: {e}"
            ) from e

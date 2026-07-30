import asyncio
import os
import time
import traceback
from asyncio import create_task
from contextlib import contextmanager
from typing import TYPE_CHECKING

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.code_provider.github.github_service import GithubService

# Lazy import for GitPython - import at module level causes SIGSEGV in forked workers
if TYPE_CHECKING:
    pass


def _get_repo_class():
    """Lazy import git.Repo to avoid fork-safety issues."""
    from git import Repo

    return Repo


from app.modules.intelligence.tools.sandbox.project_sandbox import (
    ProjectRef,
    ProjectSandbox,
    get_project_sandbox,
)
from app.modules.parsing.graph_construction.parsing_helper import (
    ParseHelper,
    ParsingFailedError,
    ParsingServiceError,
)
from app.modules.projects.projects_schema import ProjectStatusEnum
from app.modules.projects.projects_service import ProjectService
from app.modules.search.search_service import SearchService
from app.modules.utils.email_helper import EmailHelper
from app.modules.utils.logger import log_context, setup_logger
from app.modules.utils.parse_webhook_helper import ParseWebhookHelper
from sandbox import WorkspaceHandle
from sandbox.api.parser_wire import ParseArtifacts

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
        project_sandbox: ProjectSandbox | None = None,
    ):
        """Initialize ParsingService.

        Args:
            db: Database session
            user_id: User identifier
            neo4j_config: Optional Neo4j config dict for library usage.
                          If None, uses config_provider.
            raise_library_exceptions: If True, raise ParsingServiceError
                                      instead of HTTPException
            project_sandbox: Override the process-wide project sandbox
                             facade. Passed by tests; production uses
                             :func:`get_project_sandbox` (one per worker).
        """
        self.db = db
        self.parse_helper = ParseHelper(db)
        self.project_service = ProjectService(db)
        # Neo4j bypassed: InferenceService (neo4j driver + embedding model)
        # is intentionally not constructed anymore.
        self.search_service = SearchService(db)
        self.github_service = CodeProviderService(db)
        self._neo4j_config = neo4j_config
        self._raise_library_exceptions = raise_library_exceptions
        self._project_sandbox = project_sandbox or get_project_sandbox()

    def close(self) -> None:
        """Release parsing service resources (Neo4j-backed services bypassed)."""

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
                        # Re-up the project sandbox so the next agent run
                        # / conversation message lands on a warm workspace.
                        # ``ensure`` is idempotent and self-healing: no-op
                        # if already alive, recreate-and-clone if not.
                        # Best-effort — observability shouldn't fail the
                        # already-parsed early return.
                        await self._warm_project_sandbox_safe(
                            user_id=user_id,
                            project_id=str(project_id),
                            repo_name=existing_project.get("project_name"),
                            base_ref=(
                                existing_project.get("commit_id")
                                or existing_project.get("branch_name")
                            ),
                            repo_url=existing_project.get("repo_path"),
                        )
                        return {
                            "message": "Project already parsed for requested commit",
                            "id": project_id,
                        }

                if cleanup_graph:
                    # Neo4j is bypassed: no graph is written during parsing
                    # anymore, so there is nothing to clean up either. Keeping
                    # the branch (rather than deleting the parameter) preserves
                    # the public API for library callers.
                    logger.info(
                        "[PARSING] Neo4j graph cleanup permanently bypassed",
                        project_id=project_id,
                        user_id=user_id,
                        graph_cleaned=False,
                    )

                # Resolve the user's GitHub OAuth token; ProjectSandbox
                # picks an env / GitHub-App fallback if this is None.
                user_token = self._resolve_user_github_token(
                    user_id, repo_details.repo_name
                )

                # Provision the project sandbox: clones the repo at
                # base_ref into a long-lived ANALYSIS workspace. One
                # sandbox per (user, project); subsequent calls
                # (conversations, agent tools) reuse the same backing
                # workspace via the same key.
                base_ref = (
                    repo_details.commit_id or repo_details.branch_name or ""
                )
                logger.info(
                    "ParsingService: ensuring project sandbox",
                    repo_name=repo_details.repo_name,
                    base_ref=base_ref,
                    project_id=project_id,
                )
                handle = await self._project_sandbox.ensure(
                    user_id=user_id,
                    project_id=str(project_id),
                    repo=ProjectRef(
                        repo_name=repo_details.repo_name or "",
                        base_ref=base_ref,
                        repo_url=repo_details.repo_path,
                    ),
                    auth_token=user_token,
                )
                logger.info(
                    "ParsingService: project sandbox ready",
                    workspace_id=handle.workspace_id,
                    backend_kind=handle.backend_kind,
                    project_id=project_id,
                )

                # Run the in-sandbox parser and persist the resulting
                # graph. Replaces the host-FS analyze_directory path —
                # the host process never walks the repo tree itself.
                await self.analyze_workspace(
                    handle=handle,
                    project_id=project_id,
                    user_id=user_id,
                    user_email=user_email,
                    repo_details=repo_details,
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

    # ------------------------------------------------------------------
    # Sandbox-based parsing helpers (Phase 3 of the migration).
    # ------------------------------------------------------------------
    def _resolve_user_github_token(
        self, user_id: str, repo_name: str | None
    ) -> str | None:
        """Best-effort GitHub token lookup with the same logging shape
        as the legacy clone path.

        Falls back to ``None`` on any failure — ``ProjectSandbox.ensure``
        threads it through to the local adapter's resolver chain (env
        token → GitHub App → user OAuth), so a missing user token here
        is recoverable rather than fatal.
        """
        try:
            github_service = GithubService(self.db)
            user_token = github_service.get_github_oauth_token(user_id)
            if user_token:
                logger.info(
                    "Using user's GitHub OAuth token for sandbox provisioning",
                    user_id=user_id,
                    repo_name=repo_name,
                    token_prefix=(
                        user_token[:8] if len(user_token) > 8 else "short"
                    ),
                )
            else:
                logger.warning(
                    "No user GitHub OAuth token found - sandbox will fall back "
                    "to environment / app tokens",
                    user_id=user_id,
                    repo_name=repo_name,
                )
            return user_token
        except Exception as e:
            logger.exception(
                "Failed to fetch user GitHub token; sandbox will fall back",
                user_id=user_id,
                repo_name=repo_name,
                error=str(e),
            )
            return None

    async def _warm_project_sandbox_safe(
        self,
        *,
        user_id: str,
        project_id: str,
        repo_name: str | None,
        base_ref: str | None,
        repo_url: str | None,
    ) -> None:
        """Re-up the project sandbox without failing the caller.

        Used from the early-return branch in :meth:`parse_directory`
        when the project is already parsed at the requested commit —
        we still want to make sure the next agent run lands on a warm
        sandbox, but that's a nice-to-have, not a precondition for the
        early return.
        """
        if not repo_name or not base_ref:
            return
        try:
            await self._project_sandbox.ensure(
                user_id=user_id,
                project_id=project_id,
                repo=ProjectRef(
                    repo_name=repo_name,
                    base_ref=base_ref,
                    repo_url=repo_url,
                ),
            )
        except Exception:
            logger.warning(
                "Failed to warm project sandbox after early-return parse",
                project_id=project_id,
                user_id=user_id,
                repo_name=repo_name,
                base_ref=base_ref,
                exc_info=True,
            )

    async def analyze_workspace(
        self,
        *,
        handle: WorkspaceHandle,
        project_id: str,
        user_id: str,
        user_email: str,
        repo_details,
    ) -> None:
        """Run the in-sandbox parser and persist the resulting graph.

        Phase-3 replacement for :meth:`analyze_directory`. The host
        process never walks the repo tree itself — ``ProjectSandbox.parse``
        invokes ``potpie-parse`` inside the workspace, streams NDJSON
        back, and we feed the reconstructed graph straight into neo4j
        and qdrant via :meth:`CodeGraphService.store_graph_from_artifacts`.

        Refuses to proceed if the parser produced only FILE nodes —
        that's the new "language not supported" gate, replacing the
        host-side ``detect_repo_language``. parsing_rs handles every
        language we care about, so empty output ⇒ no parseable code.
        """
        analysis_start_time = time.time()
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

        logger.info(
            "[PARSING] Step 1/3: in-sandbox parse",
            project_id=project_id,
            workspace_id=handle.workspace_id,
            backend_kind=handle.backend_kind,
        )
        parse_start = time.time()
        try:
            artifacts: ParseArtifacts = await self._project_sandbox.parse(
                handle, timeout_s=900
            )
        except Exception as exc:
            logger.exception(
                "[PARSING] In-sandbox parser failed",
                project_id=project_id,
                workspace_id=handle.workspace_id,
            )
            if self._raise_library_exceptions:
                raise ParsingServiceError(
                    f"Sandbox parse failed for project {project_id}: {exc}"
                ) from exc
            raise

        parse_time = time.time() - parse_start
        non_file_nodes = sum(
            1 for n in artifacts.nodes if getattr(n, "node_type", None) != "FILE"
        )
        logger.info(
            "[PARSING] In-sandbox parse complete: %d nodes (%d non-FILE), "
            "%d edges in %.2fs",
            len(artifacts.nodes),
            non_file_nodes,
            len(artifacts.relationships),
            parse_time,
            project_id=project_id,
        )

        # Replaces the legacy "if language != 'other'" gate. parsing_rs
        # handles every language we ship a tree-sitter grammar for, so
        # an empty non-FILE node count means the repo doesn't carry
        # parseable code and the inference / search pipeline has
        # nothing to chew on.
        if non_file_nodes == 0:
            await self.project_service.update_project_status(
                project_id, ProjectStatusEnum.ERROR
            )
            if not self._raise_library_exceptions:
                await ParseWebhookHelper().send_slack_notification(
                    project_id, "Other"
                )
            raise ParsingFailedError(
                "Repository doesn't consist of a language currently supported."
            )

        # ------------------------------------------------------------------
        # Neo4j permanently bypassed: the in-sandbox parse above already
        # validated the repository and produced the structure counts, so we
        # skip the neo4j/qdrant graph write and the inference pass entirely
        # and mark the project READY directly.
        # ------------------------------------------------------------------
        await self.project_service.update_project_status(
            project_id, ProjectStatusEnum.READY
        )

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

        total_time = time.time() - analysis_start_time
        logger.info(
            "[PARSING] Analysis completed with Neo4j permanently bypassed",
            project_id=project_id,
            user_id=user_id,
            node_count=len(artifacts.nodes),
            relationship_count=len(artifacts.relationships),
            graph_created=False,
            inference_completed=False,
            parse_time_seconds=parse_time,
            total_analysis_time_seconds=total_time,
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


async def duplicate_graph(self, old_repo_id: str, new_repo_id: str):
    """Clone search indices while bypassing Neo4j graph duplication."""
    await self.search_service.clone_search_indices(old_repo_id, new_repo_id)
    logger.info(
        "[PARSING] Neo4j graph duplication permanently bypassed",
        old_repo_id=old_repo_id,
        new_repo_id=new_repo_id,
        graph_duplicated=False,
    )

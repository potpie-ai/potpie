import asyncio
import logging
import os
from asyncio import create_task
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from uuid6 import uuid7

from app.celery.tasks.parsing_tasks import process_parsing
from app.core.config_provider import config_provider
from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.parsing.graph_construction.parsing_helper import ParseHelper
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.parsing.graph_construction.parsing_service import ParsingService
from app.modules.parsing.graph_construction.parsing_validator import (
    validate_parsing_input,
)
from app.modules.parsing.utils.repo_name_normalizer import normalize_repo_name
from app.modules.projects.projects_schema import ProjectStatusEnum
from app.modules.projects.projects_service import ProjectService
from app.modules.utils.email_helper import EmailHelper
from app.modules.utils.posthog_helper import PostHogClient
from app.modules.conversations.conversation.conversation_model import Conversation
from app.modules.conversations.conversation.conversation_model import Visibility
from app.modules.projects.projects_model import Project

logger = logging.getLogger(__name__)

load_dotenv(override=True)


class ParsingController:
    @staticmethod
    @validate_parsing_input
    async def parse_directory(
        repo_details: ParsingRequest, db: AsyncSession, user: Dict[str, Any]
    ):
        if "email" not in user:
            user_email = None
        else:
            user_email = user["email"]

        user_id = user["user_id"]
        project_manager = ProjectService(db)
        parse_helper = ParseHelper(db)
        parsing_service = ParsingService(db, user_id)

        # Auto-detect if repo_name is actually a filesystem path
        if repo_details.repo_name and not repo_details.repo_path:
            is_path = (
                os.path.isabs(repo_details.repo_name)
                or repo_details.repo_name.startswith(("~", "./", "../"))
                or os.path.isdir(os.path.expanduser(repo_details.repo_name))
            )
            if is_path:
                # Move from repo_name to repo_path
                repo_details.repo_path = repo_details.repo_name
                repo_details.repo_name = repo_details.repo_path.split("/")[-1]
                logger.info(
                    f"Auto-detected filesystem path: repo_path={repo_details.repo_path}, repo_name={repo_details.repo_name}"
                )

        if config_provider.get_is_development_mode():
            # In dev mode: if both repo_path and repo_name are provided, prioritize repo_path (local)
            if repo_details.repo_path and repo_details.repo_name:
                repo_details.repo_name = None
            # Otherwise keep whichever one is provided as-is
        else:
            # In non-dev mode: if repo_name is None but repo_path exists, extract repo_name from repo_path
            if not repo_details.repo_name and repo_details.repo_path:
                repo_details.repo_name = repo_details.repo_path.split("/")[-1]

        # For later use in the code
        repo_name = repo_details.repo_name or (
            repo_details.repo_path.split("/")[-1] if repo_details.repo_path else None
        )
        repo_path = repo_details.repo_path
        if repo_path:
            if os.getenv("isDevelopmentMode") != "enabled":
                raise HTTPException(
                    status_code=400,
                    detail="Parsing local repositories is only supported in development mode",
                )
            else:
                new_project_id = str(uuid7())
                return await ParsingController.handle_new_project(
                    repo_details,
                    user_id,
                    user_email,
                    new_project_id,
                    project_manager,
                    db,
                )

        demo_repos = [
            "Portkey-AI/gateway",
            "crewAIInc/crewAI",
            "AgentOps-AI/agentops",
            "calcom/cal.com",
            "langchain-ai/langchain",
            "AgentOps-AI/AgentStack",
            "formbricks/formbricks",
        ]

        try:
            # Normalize repository name for consistent database lookups
            normalized_repo_name = normalize_repo_name(repo_name)
            logger.info(
                f"Original repo_name: {repo_name}, Normalized: {normalized_repo_name}"
            )

            project = await project_manager.get_project_from_db(
                normalized_repo_name,
                repo_details.branch_name,
                user_id,
                repo_path=repo_details.repo_path,
                commit_id=repo_details.commit_id,
            )

            # First check if this is a demo project that hasn't been accessed by this user yet
            if not project and repo_details.repo_name in demo_repos:
                existing_project = await project_manager.get_global_project_from_db(
                    normalized_repo_name,
                    repo_details.branch_name,
                    repo_details.commit_id,
                )

                new_project_id = str(uuid7())

                if existing_project:
                    await project_manager.duplicate_project(
                        repo_name,
                        repo_details.branch_name,
                        user_id,
                        new_project_id,
                        existing_project.properties,
                        existing_project.commit_id,
                    )
                    await project_manager.update_project_status(
                        new_project_id, ProjectStatusEnum.SUBMITTED
                    )

                    old_project_id = await project_manager.get_demo_project_id(
                        repo_name
                    )

                    asyncio.create_task(
                        CodeProviderService(db).get_project_structure_async(
                            new_project_id
                        )
                    )
                    # Duplicate the graph under the new repo ID
                    await parsing_service.duplicate_graph(
                        old_project_id, new_project_id
                    )

                    # Update the project status to READY after copying
                    await project_manager.update_project_status(
                        new_project_id, ProjectStatusEnum.READY
                    )
                    create_task(
                        EmailHelper().send_email(
                            user_email, repo_name, repo_details.branch_name
                        )
                    )

                    return {
                        "project_id": new_project_id,
                        "status": ProjectStatusEnum.READY.value,
                    }
                else:
                    return await ParsingController.handle_new_project(
                        repo_details,
                        user_id,
                        user_email,
                        new_project_id,
                        project_manager,
                        db,
                    )

            # Handle existing projects (including previously duplicated demo projects)
            if project:
                project_id = project.id
                is_latest = await parse_helper.check_commit_status(
                    project_id, requested_commit_id=repo_details.commit_id
                )

                if not is_latest or project.status != ProjectStatusEnum.READY.value:
                    cleanup_graph = True
                    logger.info(
                        f"Submitting parsing task for existing project {project_id}"
                    )
                    process_parsing.delay(
                        repo_details.model_dump(),
                        user_id,
                        user_email,
                        project_id,
                        cleanup_graph,
                    )

                    await project_manager.update_project_status(
                        project_id, ProjectStatusEnum.SUBMITTED
                    )
                    PostHogClient().send_event(
                        user_id,
                        "parsed_repo_event",
                        {
                            "repo_name": repo_details.repo_name,
                            "branch": repo_details.branch_name,
                            "project_id": project_id,
                        },
                    )
                    return {
                        "project_id": project_id,
                        "status": ProjectStatusEnum.SUBMITTED.value,
                    }

                # Check if repo exists in .repos and is LSP indexed
                repo_exists_and_indexed = await ParsingController._check_repo_indexed(
                    project_id,
                    repo_details.repo_name,
                    repo_details.branch_name,
                    repo_details.commit_id,
                    db,
                )

                if not repo_exists_and_indexed:
                    logger.info(
                        f"Project {project_id} exists but repo is not indexed. "
                        "Re-parsing to create LSP index."
                    )
                    cleanup_graph = False  # Don't cleanup graph, just re-index
                    process_parsing.delay(
                        repo_details.model_dump(),
                        user_id,
                        user_email,
                        project_id,
                        cleanup_graph,
                    )

                    await project_manager.update_project_status(
                        project_id, ProjectStatusEnum.SUBMITTED
                    )
                    PostHogClient().send_event(
                        user_id,
                        "parsed_repo_event",
                        {
                            "repo_name": repo_details.repo_name,
                            "branch": repo_details.branch_name,
                            "project_id": project_id,
                        },
                    )
                    return {
                        "project_id": project_id,
                        "status": ProjectStatusEnum.SUBMITTED.value,
                    }

                return {"project_id": project_id, "status": project.status}
            else:
                # Handle new non-demo projects
                new_project_id = str(uuid7())
                return await ParsingController.handle_new_project(
                    repo_details,
                    user_id,
                    user_email,
                    new_project_id,
                    project_manager,
                    db,
                )

        except Exception as e:
            logger.error(f"Error in parse_directory: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @staticmethod
    async def _check_repo_indexed(
        project_id: str,
        repo_name: str,
        branch: Optional[str],
        commit_id: Optional[str],
        db: AsyncSession,
    ) -> bool:
        """
        Check if a repository exists in .repos and is LSP indexed.

        Returns:
            True if repo exists and is indexed, False otherwise
        """
        try:
            # Check if repo manager is enabled
            repo_manager_enabled = (
                os.getenv("REPO_MANAGER_ENABLED", "false").lower() == "true"
            )
            if not repo_manager_enabled:
                # If repo manager is disabled, we can't check, so assume not indexed
                return False

            from app.modules.repo_manager import RepoManager
            from app.modules.intelligence.tools.code_query_tools.lsp_server_manager import (
                get_lsp_server_manager,
            )
            from app.modules.projects.projects_service import ProjectService

            repo_manager = RepoManager()
            worktree_path = repo_manager.get_repo_path(
                repo_name, branch=branch, commit_id=commit_id
            )

            if not worktree_path or not os.path.exists(worktree_path):
                logger.debug(
                    f"Repo {repo_name}@{commit_id} not found in .repos for project {project_id}"
                )
                return False

            # Get project details to detect languages
            project_service = ProjectService(db)
            project_details = await project_service.get_project_from_db_by_id(
                project_id
            )
            if not project_details:
                logger.debug(f"Project {project_id} not found in database")
                return False

            # Detect languages from the worktree
            parse_helper = ParseHelper(db)
            detected_language = parse_helper.detect_repo_language(worktree_path)
            detected_languages = [detected_language] if detected_language else []

            # Check if workspace is indexed
            server_manager = get_lsp_server_manager()
            is_indexed = server_manager.is_workspace_indexed(
                project_id, worktree_path, detected_languages
            )

            if not is_indexed:
                logger.debug(
                    f"Repo {repo_name}@{commit_id} exists but is not LSP indexed "
                    f"for project {project_id}"
                )

            return is_indexed

        except Exception as exc:
            logger.warning(
                f"Error checking if repo is indexed for project {project_id}: {exc}",
                exc_info=True,
            )
            # On error, assume not indexed to be safe
            return False

    @staticmethod
    async def handle_new_project(
        repo_details: ParsingRequest,
        user_id: str,
        user_email: str,
        new_project_id: str,
        project_manager: ProjectService,
        db: AsyncSession,
    ):
        response = {
            "project_id": new_project_id,
            "status": ProjectStatusEnum.SUBMITTED.value,
        }

        logger.info(f"Submitting parsing task for new project {new_project_id}")
        repo_name = repo_details.repo_name or repo_details.repo_path.split("/")[-1]
        await project_manager.register_project(
            repo_name,
            repo_details.branch_name,
            user_id,
            new_project_id,
            repo_details.commit_id,
            repo_details.repo_path,
        )
        asyncio.create_task(
            CodeProviderService(db).get_project_structure_async(new_project_id)
        )
        if not user_email:
            user_email = None

        process_parsing.delay(
            repo_details.model_dump(),
            user_id,
            user_email,
            new_project_id,
            False,
        )
        PostHogClient().send_event(
            user_id,
            "repo_parsed_event",
            {
                "repo_name": repo_details.repo_name,
                "branch": repo_details.branch_name,
                "commit_id": repo_details.commit_id,
                "project_id": new_project_id,
            },
        )
        return response

    @staticmethod
    async def fetch_parsing_status(
        project_id: str, db: AsyncSession, user: Dict[str, Any]
    ):
        try:
            project_query = (
                select(Project.status)
                .join(
                    Conversation, Conversation.project_ids.any(Project.id), isouter=True
                )
                .where(
                    Project.id == project_id,
                    or_(
                        Project.user_id == user["user_id"],
                        Conversation.visibility == Visibility.PUBLIC,
                        Conversation.shared_with_emails.any(user["email"]),
                    ),
                )
                .limit(1)  # Since we only need one result
            )

            result = db.execute(project_query)
            project_status = result.scalars().first()

            if not project_status:
                raise HTTPException(
                    status_code=404, detail="Project not found or access denied"
                )
            parse_helper = ParseHelper(db)
            is_latest = await parse_helper.check_commit_status(project_id)

            return {"status": project_status, "latest": is_latest}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in fetch_parsing_status: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

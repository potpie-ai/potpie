import asyncio
import os
from asyncio import create_task
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from uuid6 import uuid7

from app.celery.tasks.parsing_tasks import process_parsing
from app.core.config_provider import config_provider
from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.parsing.graph_construction.parsing_helper import ParseHelper
from app.modules.parsing.graph_construction.parsing_schema import (
    ParsingRequest,
    ParsingStatusRequest,
)
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
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

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

                # Check if this project is already parsed for the requested commit
                # Only check commit status if commit_id is provided
                if repo_details.commit_id:
                    is_latest = await parse_helper.check_commit_status(
                        project_id, requested_commit_id=repo_details.commit_id
                    )
                else:
                    # If no commit_id provided, check if project is READY (assume it's for the branch)
                    is_latest = project.status == ProjectStatusEnum.READY.value

                # If project exists with this commit_id and is READY, return it immediately
                if is_latest and project.status == ProjectStatusEnum.READY.value:
                    logger.info(
                        f"Project {project_id} already exists and is READY for commit {repo_details.commit_id or 'branch'}. "
                        "Returning existing project."
                    )
                    return {"project_id": project_id, "status": project.status}

                # If project exists but commit doesn't match or status is not READY, reparse
                cleanup_graph = True
                logger.info(
                    f"Submitting parsing task for existing project {project_id} "
                    f"(is_latest={is_latest}, status={project.status})"
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
                        "commit_id": repo_details.commit_id,
                        "project_id": project_id,
                    },
                )
                return {
                    "project_id": project_id,
                    "status": ProjectStatusEnum.SUBMITTED.value,
                }
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

    @staticmethod
    async def fetch_parsing_status_by_repo(
        request: ParsingStatusRequest, db: AsyncSession, user: Dict[str, Any]
    ):
        try:
            user_id = user["user_id"]
            project_manager = ProjectService(db)

            # Use ProjectService to find project by repo_name and commit_id/branch_name
            normalized_repo_name = normalize_repo_name(request.repo_name)
            project = await project_manager.get_project_from_db(
                normalized_repo_name,
                request.branch_name,
                user_id,
                repo_path=None,
                commit_id=request.commit_id,
            )

            if not project:
                raise HTTPException(
                    status_code=404,
                    detail="Project not found for the given repo_name and commit_id/branch_name",
                )

            parse_helper = ParseHelper(db)
            is_latest = await parse_helper.check_commit_status(project.id)

            return {
                "project_id": project.id,
                "repo_name": project.repo_name,
                "status": project.status,
                "latest": is_latest,
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in fetch_parsing_status_by_repo: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

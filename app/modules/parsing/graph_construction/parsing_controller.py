import asyncio
import logging
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
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.parsing.graph_construction.parsing_service import ParsingService
from app.modules.parsing.graph_construction.parsing_validator import (
    validate_parsing_input,
)
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
        logger.info(f"=== PARSING CONTROLLER ENTRY ===")
        logger.info(f"Raw repo_details: {repo_details.model_dump()}")
        logger.info(f"User: {user}")
        
        user_email = user["email"]
        user_id = user["user_id"]
        project_manager = ProjectService(db)
        parse_helper = ParseHelper(db)
        parsing_service = ParsingService(db, user_id)
        
        # Store original values for logging
        original_repo_name = repo_details.repo_name
        original_repo_path = repo_details.repo_path
        
        logger.info(f"=== PARSING CONTROLLER DEBUG ===")
        logger.info(f"Development mode: {config_provider.get_is_development_mode()}")
        logger.info(f"GitHub configured: {config_provider.is_github_configured()}")
        logger.info(f"Original repo_name: {original_repo_name}")
        logger.info(f"Original repo_path: {original_repo_path}")
        
        # Apply development mode logic for data manipulation
        if config_provider.get_is_development_mode():
            # In dev mode: if repo_name exists and no repo_path, check if it looks like a local path
            if repo_details.repo_name and not repo_details.repo_path:
                # Check if repo_name looks like a local path (starts with / or ./ or contains more than one /)
                # vs a GitHub repo name (format: owner/repo)
                repo_name_parts = repo_details.repo_name.split("/")
                is_github_format = len(repo_name_parts) == 2 and not repo_details.repo_name.startswith("/")
                is_local_path = (repo_details.repo_name.startswith("/") or 
                               repo_details.repo_name.startswith("./") or 
                               repo_details.repo_name.startswith("../") or
                               len(repo_name_parts) > 2)
                
                if is_local_path:
                    logger.info(f"Dev mode: Treating repo_name '{repo_details.repo_name}' as local path")
                    repo_details.repo_path = repo_details.repo_name
                    repo_details.repo_name = None
                    logger.info(f"Dev mode: After manipulation - repo_name: {repo_details.repo_name}, repo_path: {repo_details.repo_path}")
                elif is_github_format:
                    logger.info(f"Dev mode: Keeping repo_name '{repo_details.repo_name}' as remote repository (GitHub format)")
                else:
                    logger.info(f"Dev mode: Ambiguous repo_name '{repo_details.repo_name}', treating as local path")
                    repo_details.repo_path = repo_details.repo_name
                    repo_details.repo_name = None
                    logger.info(f"Dev mode: After manipulation - repo_name: {repo_details.repo_name}, repo_path: {repo_details.repo_path}")
            else:
                logger.info(f"Dev mode: No manipulation needed - repo_name: {repo_details.repo_name}, repo_path: {repo_details.repo_path}")
        else:
            # In non-dev mode: if repo_name is None but repo_path exists, extract repo_name from repo_path
            if not repo_details.repo_name and repo_details.repo_path:
                extracted_name = repo_details.repo_path.split("/")[-1]
                logger.info(f"Non-dev mode: Extracting repo_name '{extracted_name}' from repo_path '{repo_details.repo_path}'")
                repo_details.repo_name = extracted_name
                logger.info(f"Non-dev mode: After manipulation - repo_name: {repo_details.repo_name}, repo_path: {repo_details.repo_path}")
            else:
                logger.info(f"Non-dev mode: No manipulation needed - repo_name: {repo_details.repo_name}, repo_path: {repo_details.repo_path}")

        # For later use in the code
        repo_name = repo_details.repo_name or (
            repo_details.repo_path.split("/")[-1] if repo_details.repo_path else None
        )
        repo_path = repo_details.repo_path
        
        logger.info(f"After manipulation - repo_name: {repo_details.repo_name}, repo_path: {repo_details.repo_path}")
        logger.info(f"Computed repo_name: {repo_name}, repo_path: {repo_path}")
        
        # Check if repo_path is provided
        if repo_path:
            # Check if development mode is enabled for local repositories
            if os.getenv("isDevelopmentMode") != "enabled":
                raise HTTPException(
                    status_code=400,
                    detail="Parsing local repositories is only supported in development mode",
                )
            # Check if the local path actually exists
            elif os.path.exists(repo_path):
                logger.info(f"Local repository detected: {repo_path}")
                new_project_id = str(uuid7())
                return await ParsingController.handle_new_project(
                    repo_details,
                    user_id,
                    user_email,
                    new_project_id,
                    project_manager,
                    db,
                )
            else:
                # repo_path provided but doesn't exist
                logger.error(f"Local repository path does not exist: {repo_path}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Local repository does not exist at path: {repo_path}",
                )
        elif repo_details.repo_name and not config_provider.is_github_configured():
            # Remote repository requested but GitHub not configured
            logger.error(f"Remote repository '{repo_details.repo_name}' requested but GitHub not configured")
            raise HTTPException(
                status_code=400,
                detail="GitHub is not configured, cannot parse remote repositories. Use repo_path for local repositories.",
            )
        elif repo_details.repo_name:
            # Remote repository with GitHub configured - proceed
            logger.info(f"Remote repository detected: {repo_details.repo_name}")
        else:
            # Neither local path nor remote repo name provided
            logger.error(f"Neither local path nor remote repo name provided after manipulation")
            logger.error(f"Final repo_name: {repo_details.repo_name}, repo_path: {repo_details.repo_path}")
            raise HTTPException(
                status_code=400,
                detail="Either a valid local repository path or remote repository name must be provided.",
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
            project = await project_manager.get_project_from_db(
                repo_name,
                repo_details.branch_name,
                user_id,
                repo_path=repo_details.repo_path,
                commit_id=repo_details.commit_id,
            )

            # First check if this is a demo project that hasn't been accessed by this user yet
            if not project and repo_details.repo_name in demo_repos:
                existing_project = await project_manager.get_global_project_from_db(
                    repo_name, repo_details.branch_name, repo_details.commit_id
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
                is_latest = await parse_helper.check_commit_status(project_id)

                if not is_latest or project.status != ProjectStatusEnum.READY.value:
                    cleanup_graph = True
                    logger.info(
                        f"Submitting parsing task for existing project {project_id}"
                    )
                    logger.info(f"Repo details for Celery: {repo_details.model_dump(exclude_none=True)}")
                    process_parsing.delay(
                        repo_details.model_dump(exclude_none=True),
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
        logger.info(f"Repo details for Celery: {repo_details.model_dump(exclude_none=True)}")
        repo_name = repo_details.repo_name or (
            repo_details.repo_path.split("/")[-1] if repo_details.repo_path else None
        )
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
            repo_details.model_dump(exclude_none=True),
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

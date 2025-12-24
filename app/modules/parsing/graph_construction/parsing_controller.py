import asyncio
import logging
import os
from asyncio import create_task
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
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
from app.modules.projects.projects_model import Project

logger = logging.getLogger(__name__)

load_dotenv(override=True)


class ParsingController:
    @staticmethod
    @validate_parsing_input
    async def parse_directory(
        repo_details: ParsingRequest, db: AsyncSession, user: Dict[str, Any]
    ):
        # Extract user_email from user object (Firebase tokens may have 'email' field)
        user_email = user.get("email") or user.get("user_email") or None
        
        # Extract user_id from user object (Firebase tokens use 'uid', but we also support 'user_id')
        user_id = user.get("user_id") or user.get("uid")
        
        if not user_id:
            logger.error(f"User ID not found in user object: {user.keys()}")
            raise HTTPException(
                status_code=400,
                detail="User ID not found in authentication token"
            )
        
        # Create a sync session for services that need it (ParsingService, InferenceService, etc.)
        # These services use .query() which is only available on sync sessions
        from app.core.database import SessionLocal
        sync_db = SessionLocal()
        
        try:
            # ProjectService can work with both sync and async, but we'll use sync for consistency
            # with other services that require it
            project_manager = ProjectService(sync_db)
            parse_helper = ParseHelper(sync_db)
            parsing_service = ParsingService(sync_db, user_id)

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
                    result = await ParsingController.handle_new_project(
                        repo_details,
                        user_id,
                        user_email,
                        new_project_id,
                        project_manager,
                        db,
                    )
                    return result

            demo_repos = [
                "Portkey-AI/gateway",
                "crewAIInc/crewAI",
                "AgentOps-AI/agentops",
                "calcom/cal.com",
                "langchain-ai/langchain",
                "AgentOps-AI/AgentStack",
                "formbricks/formbricks",
            ]
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
                    result = await ParsingController.handle_new_project(
                        repo_details,
                        user_id,
                        user_email,
                        new_project_id,
                        project_manager,
                        db,
                    )
                    return result

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

                return {"project_id": project_id, "status": project.status}
            else:
                # Handle new non-demo projects
                new_project_id = str(uuid7())
                result = await ParsingController.handle_new_project(
                    repo_details,
                    user_id,
                    user_email,
                    new_project_id,
                    project_manager,
                    db,
                )
                return result

        except Exception as e:
            logger.error(f"Error in parse_directory: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            # Close the sync session
            sync_db.close()

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
            user_email,
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
        # Create a sync session for ParseHelper and ProjectService which need sync Session
        from app.core.database import SessionLocal
        sync_db = SessionLocal()
        
        try:
            # Extract user_id and email from user object (handle both uid and user_id)
            user_id = user.get("user_id") or user.get("uid")
            user_email = user.get("email") or user.get("user_email") or None
            
            if not user_id:
                raise HTTPException(
                    status_code=400,
                    detail="User ID not found in authentication token"
                )
            
            # Resolve actual user_id (may differ if user exists by email)
            from app.modules.projects.projects_service import ProjectService
            project_service = ProjectService(sync_db)
            user_obj = project_service._ensure_user_exists(user_id, user_email)
            actual_user_id = user_obj.uid
            
            # Query project directly by ID and user_id
            project_query = select(Project).where(
                Project.id == project_id,
                Project.user_id == actual_user_id,
            )

            result = await db.execute(project_query)
            project = result.scalars().first()

            if not project:
                raise HTTPException(
                    status_code=404, detail="Project not found or access denied"
                )
            
            parse_helper = ParseHelper(sync_db)
            is_latest = await parse_helper.check_commit_status(project_id)

            return {"status": project.status, "latest": is_latest}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in fetch_parsing_status: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            sync_db.close()

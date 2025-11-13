import asyncio
import logging
import os
import time
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
        start_time = time.perf_counter()
        logger.info(
            f"[TIMING] parsing_controller.parse_directory: START | "
            f"repo_name={repo_details.repo_name}, branch={repo_details.branch_name}"
        )
        
        if "email" not in user:
            user_email = None
        else:
            user_email = user["email"]

        user_id = user["user_id"]
        
        init_start = time.perf_counter()
        project_manager = ProjectService(db)
        parse_helper = ParseHelper(db)
        parsing_service = ParsingService(db, user_id)
        init_elapsed = time.perf_counter() - init_start
        logger.info(
            f"[TIMING] parsing_controller.parse_directory: Service initialization | "
            f"elapsed={init_elapsed:.4f}s"
        )

        # Auto-detect if repo_name is actually a filesystem path
        path_detect_start = time.perf_counter()
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
        path_detect_elapsed = time.perf_counter() - path_detect_start
        if path_detect_elapsed > 0.001:
            logger.info(
                f"[TIMING] parsing_controller.parse_directory: Path detection | "
                f"elapsed={path_detect_elapsed:.4f}s"
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
            normalize_start = time.perf_counter()
            normalized_repo_name = normalize_repo_name(repo_name)
            normalize_elapsed = time.perf_counter() - normalize_start
            logger.info(
                f"Original repo_name: {repo_name}, Normalized: {normalized_repo_name}"
            )
            logger.info(
                f"[TIMING] parsing_controller.parse_directory: Repo name normalization | "
                f"elapsed={normalize_elapsed:.4f}s"
            )

            db_query_start = time.perf_counter()
            project = await project_manager.get_project_from_db(
                normalized_repo_name,
                repo_details.branch_name,
                user_id,
                repo_path=repo_details.repo_path,
                commit_id=repo_details.commit_id,
            )
            db_query_elapsed = time.perf_counter() - db_query_start
            logger.info(
                f"[TIMING] parsing_controller.parse_directory: Database project lookup | "
                f"elapsed={db_query_elapsed:.4f}s | found={project is not None}"
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
                commit_check_start = time.perf_counter()
                is_latest = await parse_helper.check_commit_status(
                    project_id, requested_commit_id=repo_details.commit_id
                )
                commit_check_elapsed = time.perf_counter() - commit_check_start
                logger.info(
                    f"[TIMING] parsing_controller.parse_directory: Commit status check | "
                    f"elapsed={commit_check_elapsed:.4f}s | is_latest={is_latest}"
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
                return await ParsingController.handle_new_project(
                    repo_details,
                    user_id,
                    user_email,
                    new_project_id,
                    project_manager,
                    db,
                )

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(
                f"[TIMING] parsing_controller.parse_directory: ERROR | "
                f"elapsed={elapsed:.4f}s | error={str(e)}"
            )
            raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            elapsed = time.perf_counter() - start_time
            logger.info(
                f"[TIMING] parsing_controller.parse_directory: COMPLETE | "
                f"total_elapsed={elapsed:.4f}s"
            )

    @staticmethod
    async def handle_new_project(
        repo_details: ParsingRequest,
        user_id: str,
        user_email: str,
        new_project_id: str,
        project_manager: ProjectService,
        db: AsyncSession,
    ):
        start_time = time.perf_counter()
        logger.info(
            f"[TIMING] parsing_controller.handle_new_project: START | "
            f"project_id={new_project_id}"
        )
        
        response = {
            "project_id": new_project_id,
            "status": ProjectStatusEnum.SUBMITTED.value,
        }

        logger.info(f"Submitting parsing task for new project {new_project_id}")
        repo_name = repo_details.repo_name or repo_details.repo_path.split("/")[-1]
        
        register_start = time.perf_counter()
        await project_manager.register_project(
            repo_name,
            repo_details.branch_name,
            user_id,
            new_project_id,
            repo_details.commit_id,
            repo_details.repo_path,
        )
        register_elapsed = time.perf_counter() - register_start
        logger.info(
            f"[TIMING] parsing_controller.handle_new_project: Project registration | "
            f"elapsed={register_elapsed:.4f}s"
        )
        structure_start = time.perf_counter()
        asyncio.create_task(
            CodeProviderService(db).get_project_structure_async(new_project_id)
        )
        structure_elapsed = time.perf_counter() - structure_start
        logger.info(
            f"[TIMING] parsing_controller.handle_new_project: Project structure task | "
            f"elapsed={structure_elapsed:.4f}s"
        )
        
        if not user_email:
            user_email = None

        task_submit_start = time.perf_counter()
        process_parsing.delay(
            repo_details.model_dump(),
            user_id,
            user_email,
            new_project_id,
            False,
        )
        task_submit_elapsed = time.perf_counter() - task_submit_start
        logger.info(
            f"[TIMING] parsing_controller.handle_new_project: Celery task submission | "
            f"elapsed={task_submit_elapsed:.4f}s"
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
        
        elapsed = time.perf_counter() - start_time
        logger.info(
            f"[TIMING] parsing_controller.handle_new_project: COMPLETE | "
            f"total_elapsed={elapsed:.4f}s | project_id={new_project_id}"
        )
        return response

    @staticmethod
    async def fetch_parsing_status(
        project_id: str, db: AsyncSession, user: Dict[str, Any]
    ):
        start_time = time.perf_counter()
        logger.info(
            f"[TIMING] parsing_controller.fetch_parsing_status: START | "
            f"project_id={project_id}"
        )
        
        try:
            query_start = time.perf_counter()
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
            query_elapsed = time.perf_counter() - query_start
            logger.info(
                f"[TIMING] parsing_controller.fetch_parsing_status: Database query | "
                f"elapsed={query_elapsed:.4f}s"
            )

            if not project_status:
                raise HTTPException(
                    status_code=404, detail="Project not found or access denied"
                )
            
            commit_check_start = time.perf_counter()
            parse_helper = ParseHelper(db)
            is_latest = await parse_helper.check_commit_status(project_id)
            commit_check_elapsed = time.perf_counter() - commit_check_start
            logger.info(
                f"[TIMING] parsing_controller.fetch_parsing_status: Commit status check | "
                f"elapsed={commit_check_elapsed:.4f}s"
            )

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"[TIMING] parsing_controller.fetch_parsing_status: COMPLETE | "
                f"total_elapsed={elapsed:.4f}s"
            )
            return {"status": project_status, "latest": is_latest}

        except HTTPException:
            raise
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(
                f"[TIMING] parsing_controller.fetch_parsing_status: ERROR | "
                f"elapsed={elapsed:.4f}s | error={str(e)}"
            )
            raise HTTPException(status_code=500, detail="Internal server error")

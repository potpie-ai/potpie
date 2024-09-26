import logging
from typing import Any, Dict

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from uuid6 import uuid7

from app.celery.tasks.parsing_tasks import process_parsing
from app.modules.parsing.graph_construction.parsing_helper import ParseHelper
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.parsing.graph_construction.parsing_validator import (
    validate_parsing_input,
)
from app.modules.projects.projects_schema import ProjectStatusEnum
from app.modules.projects.projects_service import ProjectService
from app.modules.utils.posthog_helper import PostHogClient

logger = logging.getLogger(__name__)


class ParsingController:
    @staticmethod
    @validate_parsing_input
    async def parse_directory(
        repo_details: ParsingRequest, db: AsyncSession, user: Dict[str, Any]
    ):
        user_id = user["user_id"]
        user_email = user["email"]
        project_manager = ProjectService(db)
        parse_helper = ParseHelper(db)
        repo_name = repo_details.repo_name or repo_details.repo_path.split("/")[-1]

        try:
            project = await project_manager.get_project_from_db(
                repo_name, repo_details.branch_name, user_id
            )

            if not project:
                new_project_id = str(uuid7())
                response = {
                    "project_id": new_project_id,
                    "status": ProjectStatusEnum.SUBMITTED.value,
                }

                logger.info(f"Submitting parsing task for new project {new_project_id}")

                await project_manager.register_project(repo_name,
                    repo_details.branch_name,
                    user_id,
                    new_project_id)

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
                        "project_id": new_project_id,
                    },
                )

                return response

            project_id = project.id
            project_status = project.status
            response = {"project_id": project_id, "status": project_status}
            # TODO: seems buggy, check and fix
            is_latest = await parse_helper.check_commit_status(project_id)

            if not is_latest or project_status != ProjectStatusEnum.READY.value:
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

                response["status"] = ProjectStatusEnum.SUBMITTED.value
                PostHogClient().send_event(
                    user_id,
                    "parsed_repo_event",
                    {
                        "repo_name": repo_details.repo_name,
                        "branch": repo_details.branch_name,
                        "project_id": project_id,
                    },
                )

            return response
        except Exception as e:
            logger.error(f"Error in parse_directory: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @staticmethod
    async def fetch_parsing_status(
        project_id: str, db: AsyncSession, user: Dict[str, Any]
    ):
        try:
            project_service = ProjectService(db)
            parse_helper = ParseHelper(db)
            project = await project_service.get_project_from_db_by_id_and_user_id(
                project_id, user["user_id"]
            )
            if project:
                is_latest = await parse_helper.check_commit_status(project_id)
                return {"status": project["status"], "latest": is_latest}
            else:
                raise HTTPException(status_code=404, detail="Project not found")
        except Exception as e:
            logger.error(f"Error in fetch_parsing_status: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

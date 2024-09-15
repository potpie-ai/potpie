import logging
from typing import Any, Dict

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from uuid6 import uuid7

from app.celery.tasks.parsing_tasks import process_parsing
from app.core.config_provider import config_provider
from app.modules.parsing.graph_construction.code_graph_service import CodeGraphService
from app.modules.parsing.graph_construction.parsing_helper import ParseHelper
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.parsing.graph_construction.parsing_validator import (
    validate_parsing_input,
)
from app.modules.projects.projects_schema import ProjectStatusEnum
from app.modules.projects.projects_service import ProjectService

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
                process_parsing.delay(
                    repo_details.model_dump(), user_id, user_email, new_project_id
                )

                return response

            project_id = project.id
            project_status = project.status
            response = {"project_id": project_id, "status": project_status}
            # TODO: seems buggy, check and fix
            is_latest = await parse_helper.check_commit_status(project_id)

            if not is_latest or project_status != ProjectStatusEnum.READY.value:
                neo4j_config = config_provider.get_neo4j_config()

                try:
                    code_graph_service = CodeGraphService(
                        neo4j_config["uri"],
                        neo4j_config["username"],
                        neo4j_config["password"],
                        db,
                    )

                    await code_graph_service.cleanup_graph(project_id)
                except Exception as e:
                    logger.error(f"Error in cleanup_graph: {e}")
                    raise HTTPException(status_code=500, detail="Internal server error")

                logger.info(
                    f"Submitting parsing task for existing project {project_id}"
                )
                process_parsing.delay(
                    repo_details.model_dump(), user_id, user_email, project_id
                )

                response["status"] = ProjectStatusEnum.SUBMITTED.value

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


logger.info("Parsing controller module loaded")

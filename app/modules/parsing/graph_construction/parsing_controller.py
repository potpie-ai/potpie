from fastapi import HTTPException
from requests import Session
from uuid6 import uuid7

from app.core.config_provider import config_provider
from app.modules.parsing.graph_construction.code_graph_service import CodeGraphService
from app.modules.parsing.graph_construction.parsing_helper import ParseHelper
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.parsing.graph_construction.parsing_tasks import process_parsing
from app.modules.parsing.graph_construction.parsing_validator import (
    validate_parsing_input,
)
from app.modules.projects.projects_schema import ProjectStatusEnum
from app.modules.projects.projects_service import ProjectService


class ParsingController:
    @staticmethod
    @validate_parsing_input
    async def parse_directory(repo_details: ParsingRequest, db: Session, user: dict):
        user_id = user["user_id"]
        user_email = user["email"]
        project_manager = ProjectService(db)
        parse_helper = ParseHelper(db)
        repo_name = repo_details.repo_name or repo_details.repo_path.split("/")[-1]
        project = await project_manager.get_project_from_db(repo_name, user_id)

        if not project:
            new_project_id = str(uuid7())
            # Prepare the response
            response = {
                "project_id": new_project_id,
                "status": ProjectStatusEnum.SUBMITTED.value,
            }

            # Enqueue the parsing task to be processed asynchronously
            process_parsing.apply_async(
                args=[repo_details.model_dump(), user_id, user_email, new_project_id],
            )

            # Return the response immediately
            return response

        project_id = project.id
        project_status = project.status
        response = {"project_id": project_id, "status": project_status}

        is_latest = await parse_helper.check_commit_status(project_id)

        if not is_latest or project_status != ProjectStatusEnum.READY.value:
            neo4j_config = config_provider.get_neo4j_config()
            code_graph_service = CodeGraphService(
                neo4j_config["uri"],
                neo4j_config["username"],
                neo4j_config["password"],
                db,
            )
            await code_graph_service.cleanup_graph(project_id)
            code_graph_service.close()

            # Enqueue the parsing task to be processed asynchronously
            process_parsing.apply_async(
                args=[repo_details.model_dump(), user_id, user_email, project_id],
            )

            # Update the response status
            response["status"] = ProjectStatusEnum.SUBMITTED.value

        # Return the response immediately
        return response

    @staticmethod
    async def fetch_parsing_status(project_id: str, db: Session, user: dict):
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

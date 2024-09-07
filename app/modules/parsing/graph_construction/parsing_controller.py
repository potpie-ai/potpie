import asyncio

from fastapi import HTTPException
from requests import Session
from uuid6 import uuid7

from app.core.config_provider import config_provider
from app.modules.parsing.graph_construction.code_graph_service import CodeGraphService
from app.modules.parsing.graph_construction.parsing_helper import ParseHelper
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.parsing.graph_construction.parsing_service import ParsingService
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
        repo_name = (
            repo_details.repo_name
            if repo_details.repo_name
            else repo_details.repo_path.split("/")[-1]
        )
        project = await project_manager.get_project_from_db(repo_name, user_id)

        if not project:
            new_project_id = str(uuid7())
            response = {
                "project_id": new_project_id,
                "status": ProjectStatusEnum.SUBMITTED.value,
            }

            asyncio.create_task(
                ParsingController._process_parsing(
                    repo_details, user_id, user_email, new_project_id, db
                )
            )

            return response

        else:
            project_id = project.id
            project_status = project.status
            response = {"project_id": project_id, "status": project_status}

            # Check if the project is on the latest commit
            is_latest = await parse_helper.check_commit_status(project_id)

            if not is_latest or project_status != ProjectStatusEnum.READY.value:
                neo4j_config = config_provider.get_neo4j_config()
                # Cleanup existing graph and search index
                code_graph_service = CodeGraphService(
                    neo4j_config["uri"],
                    neo4j_config["username"],
                    neo4j_config["password"],
                    db,
                )
                await code_graph_service.cleanup_graph(project_id)
                code_graph_service.close()
                asyncio.create_task(
                    ParsingController._process_parsing(
                        repo_details, user_id, user_email, project_id, db
                    )
                )
                response["status"] = ProjectStatusEnum.SUBMITTED.value

            return response

    @staticmethod
    async def _process_parsing(
        repo_details: ParsingRequest, user_id: str, user_email: str, project_id: str, db
    ):
        parsing_service = ParsingService(db)
        await parsing_service.parse_directory(
            repo_details, user_id, user_email, project_id
        )

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

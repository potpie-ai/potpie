import asyncio
from uuid6 import uuid7

from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.parsing.graph_construction.parsing_service import ParsingService
from app.modules.parsing.graph_construction.parsing_validator import validate_parsing_input
from app.modules.projects.projects_service import ProjectService
from app.modules.projects.projects_schema import ProjectStatusEnum

class ParsingController:
    @staticmethod
    @validate_parsing_input
    async def parse_directory(
        repo_details: ParsingRequest,
        db,
        user,
    ):
        user_id = user["user_id"]
        user_email = user["email"]

        project_manager = ProjectService(db)

        project = await project_manager.get_project_from_db(
            repo_details.repo_name, user_id
        )

        if not project:
            new_project_id = str(uuid7())
            response = {"project_id": new_project_id, "status": ProjectStatusEnum.SUBMITTED.value}

            asyncio.create_task(ParsingController._process_parsing(repo_details, user_id, user_email, new_project_id, db))
            
            return response
        
        else:
            project_id = project.id
            project_status = project.status

            response = {"project_id": project_id, "status": project_status}

            if project_status == ProjectStatusEnum.READY.value:
                return response
            
            asyncio.create_task(ParsingController._process_parsing(repo_details, user_id, user_email, project_id, db))

            return response

    @staticmethod
    async def _process_parsing(repo_details: ParsingRequest, user_id: str, user_email: str, project_id: str, db):
        print("parsing service was called")
        parsing_service = ParsingService(db)
        await parsing_service.parse_directory(
            repo_details, user_id, user_email, project_id
        )

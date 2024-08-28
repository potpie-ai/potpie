from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.parsing.graph_construction.parsing_service import ParsingService
from app.modules.parsing.graph_construction.parsing_validator import (
    validate_parsing_input,
)
from app.modules.projects.projects_service import ProjectService


class ParsingController:
    @staticmethod
    @validate_parsing_input
    async def parse_directory(
        repo_details: ParsingRequest,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        project_manager = ProjectService(db)
        project = await project_manager.get_project_from_db(
            repo_details.repo_name, user_id
        )
        project_id = project.id if project else None

        parsing_service = ParsingService(db)
        return await parsing_service.parse_directory(repo_details, user_id, project_id)

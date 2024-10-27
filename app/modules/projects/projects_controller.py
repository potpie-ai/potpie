from fastapi import Depends, HTTPException
from starlette.responses import JSONResponse

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.projects.projects_service import ProjectService


class ProjectController:
    @staticmethod
    async def get_project_list(
        user=Depends(AuthService.check_auth), db=Depends(get_db)
    ):
        user_id = user["user_id"]
        try:
            project_service = ProjectService(db)
            project_list = await project_service.list_projects(user_id)
            return project_list
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def delete_project(
        project_id: str, user=Depends(AuthService.check_auth), db=Depends(get_db)
    ):
        project_service = ProjectService(db)
        try:
            await project_service.delete_project(project_id)
            return JSONResponse(
                status_code=200,
                content={"message": "Project deleted successfully.", "id": project_id},
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{str(e)}")

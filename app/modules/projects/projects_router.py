from fastapi import Depends

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.utils.APIRouter import APIRouter

from .projects_controller import ProjectController

router = APIRouter()


@router.get("/projects/list")
async def get_project_list(user=Depends(AuthService.check_auth), db=Depends(get_db)):
    return await ProjectController.get_project_list(user=user, db=db)


@router.delete("/projects")
async def delete_project(
    project_id: str, user=Depends(AuthService.check_auth), db=Depends(get_db)
):
    return await ProjectController.delete_project(
        project_id=project_id, user=user, db=db
    )

from fastapi import Depends

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.utils.APIRouter import APIRouter

from .projects_controller import ProjectController
from .projects_schema import (
    CreateProjectRequest,
    CreateProjectResponse,
    SelectRepositoryRequest,
    SelectRepositoryResponse,
)

router = APIRouter()


@router.get("/projects/list")
async def get_project_list(user=Depends(AuthService.check_auth), db=Depends(get_db)):
    return await ProjectController.get_project_list(user=user, db=db)


@router.post("/projects/", response_model=CreateProjectResponse)
async def create_project(
    request: CreateProjectRequest,
    user=Depends(AuthService.check_auth),
    db=Depends(get_db),
):
    return await ProjectController.create_project(request=request, user=user, db=db)


@router.post("/repos/select", response_model=SelectRepositoryResponse)
async def select_repository(
    request: SelectRepositoryRequest,
    user=Depends(AuthService.check_auth),
    db=Depends(get_db),
):
    return await ProjectController.select_repository(request=request, user=user, db=db)


@router.delete("/projects")
async def delete_project(
    project_id: str, user=Depends(AuthService.check_auth), db=Depends(get_db)
):
    return await ProjectController.delete_project(
        project_id=project_id, user=user, db=db
    )
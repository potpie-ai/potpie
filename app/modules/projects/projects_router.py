from fastapi import Body, Depends

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.projects.projects_schema import WorkspaceCreateBody
from app.modules.utils.APIRouter import APIRouter

from .projects_controller import ProjectController

router = APIRouter()


@router.get("/projects/list")
async def get_project_list(user=Depends(AuthService.check_auth), db=Depends(get_db)):
    return await ProjectController.get_project_list(user=user, db=db)


@router.post("/projects/workspace")
async def create_workspace_project(
    user=Depends(AuthService.check_auth),
    db=Depends(get_db),
    body: WorkspaceCreateBody = Body(default_factory=WorkspaceCreateBody),
):
    """Create a minimal project when the user has no parsed GitHub repo yet (e.g. Linear-only)."""
    return await ProjectController.create_workspace_project(
        user=user, db=db, body=body
    )


@router.delete("/projects")
async def delete_project(
    project_id: str, user=Depends(AuthService.check_auth), db=Depends(get_db)
):
    return await ProjectController.delete_project(
        project_id=project_id, user=user, db=db
    )

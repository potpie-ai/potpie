from fastapi import Depends
import uuid
from typing import Optional
from pydantic import BaseModel

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.utils.APIRouter import APIRouter
from app.modules.projects.projects_controller import ProjectController
from app.modules.projects.projects_service import ProjectService

# Create a Pydantic model for the request body
class ProjectRegistration(BaseModel):
    repo_name: str
    branch_name: str
    repo_path: Optional[str] = None

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

@router.post("/projects/register")
async def register_project(
    project_data: ProjectRegistration,
    user=Depends(AuthService.check_auth),
    db=Depends(get_db)
):
    project_service = ProjectService(db)
    return await project_service.register_project(
        repo_name=project_data.repo_name,
        branch_name=project_data.branch_name,
        user_id=user["user_id"],
        project_id=str(uuid.uuid4()),
        repo_path=project_data.repo_path
    )
from fastapi import Depends, HTTPException
from starlette.responses import JSONResponse
from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.projects.projects_service import ProjectService
from app.modules.utils.APIRouter import APIRouter

router = APIRouter()

class ProjectAPI:
    @router.get("/projects/list")
    async def get_project_list(
        user=Depends(AuthService.check_auth), db=Depends(get_db)
    ):
        user_id = user["user_id"]
        project_service = ProjectService(db)
        try:
            branch_list = []
            project_details = await project_service.get_parsed_project_branches( user_id )
            branch_list.extend(
                {
                    "project_id": branch[0],
                    "branch_name": branch[1],
                    "repo_name": branch[2],
                    "last_updated_at": branch[3],
                    "is_default": branch[4],
                    "project_status": branch[5],
                }
                for branch in project_details
            )
            return branch_list
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"{str(e)}")

    @router.delete("/projects")
    def delete_project(project_id: int, user=Depends(AuthService.check_auth), db=Depends(get_db)):
        
        project_service = ProjectService(db)
        try:
            project_service.delete_project(project_id)
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Project deleted successfully.",
                    "id": project_id
                }
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{str(e)}")

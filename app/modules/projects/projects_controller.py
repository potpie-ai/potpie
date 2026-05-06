from datetime import datetime

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from starlette.responses import JSONResponse

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.projects.projects_schema import WorkspaceCreateBody
from app.modules.projects.projects_service import ProjectService
from app.modules.users.user_schema import CreateUser
from app.modules.users.user_service import UserService


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

    @staticmethod
    async def create_workspace_project(
        user: dict,
        db: Session,
        body: WorkspaceCreateBody,
    ):
        """Ensure `users` row exists, then create a minimal `projects` row for sources/workflows."""
        user_id = user["user_id"]
        us = UserService(db)
        if not us.get_user_by_uid(user_id):
            cu = CreateUser(
                uid=user_id,
                email=user.get("email") or f"{user_id}@users.potpie.local",
                display_name=(
                    user.get("name")
                    or (user.get("email") or "").split("@")[0]
                    or "User"
                ),
                email_verified=bool(user.get("email_verified", True)),
                created_at=datetime.utcnow(),
                last_login_at=datetime.utcnow(),
                provider_info={"source": "workspace_bootstrap"},
                provider_username=None,
            )
            _, msg, err = us.create_user(cu)
            if err:
                raise HTTPException(
                    status_code=500,
                    detail=msg or "Could not create user record",
                )
        try:
            project_service = ProjectService(db)
            return await project_service.create_workspace_project(
                user_id, body.display_name
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

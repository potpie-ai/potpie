import uuid
import json
from datetime import datetime
from fastapi import Depends, HTTPException
from starlette.responses import JSONResponse

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.projects.projects_service import ProjectService
from app.modules.projects.projects_schema import (
    CreateProjectRequest,
    CreateProjectResponse,
    SelectRepositoryRequest,
    SelectRepositoryResponse,
    ProjectStatusEnum,
)
from app.modules.projects.projects_model import Project


class ProjectController:
    @staticmethod
    async def get_project_list(
        user=Depends(AuthService.check_auth), db=Depends(get_db)
    ):
        # Extract user_id and email from user object (handle both uid and user_id)
        token_user_id = user.get("user_id") or user.get("uid")
        user_email = user.get("email") or user.get("user_email") or None
        
        if not token_user_id:
            raise HTTPException(
                status_code=400,
                detail="User ID not found in authentication token"
            )
        
        try:
            project_service = ProjectService(db)
            # Resolve actual user_id from database (may differ if user exists by email)
            if hasattr(project_service, '_ensure_user_exists'):
                user_obj = project_service._ensure_user_exists(token_user_id, user_email)
                actual_user_id = user_obj.uid
            else:
                actual_user_id = token_user_id
            
            project_list = await project_service.list_projects(actual_user_id)
            return project_list
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def create_project(
        request: CreateProjectRequest,
        user=Depends(AuthService.check_auth),
        db=Depends(get_db),
    ):
        user_id = user.get("user_id") or user.get("uid")
        if not user_id:
            raise HTTPException(
                status_code=400,
                detail="User ID not found in authentication token"
            )
        
        try:
            # Generate project ID
            project_id = f"proj-{uuid.uuid4().hex[:12]}"
            
            # Store idea in properties field as JSON
            properties = json.dumps({"idea": request.idea}).encode("utf-8")
            
            # Create project
            project = Project(
                id=project_id,
                user_id=user_id,
                properties=properties,
                status=ProjectStatusEnum.SUBMITTED.value,
                repo_name="",  # Will be set when repository is selected
                branch_name="",  # Will be set when repository is selected
            )
            
            project_service = ProjectService(db)
            created_project = ProjectService.create_project(db, project)
            
            return CreateProjectResponse(
                id=created_project.id,
                idea=request.idea,
                status=created_project.status,
                created_at=created_project.created_at.isoformat() if created_project.created_at else datetime.utcnow().isoformat(),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")

    @staticmethod
    async def select_repository(
        request: SelectRepositoryRequest,
        user=Depends(AuthService.check_auth),
        db=Depends(get_db),
    ):
        user_id = user.get("user_id") or user.get("uid")
        if not user_id:
            raise HTTPException(
                status_code=400,
                detail="User ID not found in authentication token"
            )
        
        try:
            project_service = ProjectService(db)
            
            # Verify project exists and belongs to user
            project = await project_service.get_project_from_db_by_id_and_user_id(
                request.project_id, user_id
            )
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
            
            # TODO: Implement repository selection logic
            # For now, return success
            # This will be extended to actually link the repository and trigger parsing
            
            return SelectRepositoryResponse(
                status="success",
                analysis_id=None,  # Will be set when parsing is implemented
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to select repository: {str(e)}")

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

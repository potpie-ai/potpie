import json
from typing import Any, List
from fastapi import Depends, HTTPException, Query
from app.modules.utils.APIRouter import APIRouter
from fastapi.responses import StreamingResponse
from app.modules.auth.auth_service import AuthService
from sqlalchemy.orm import Session
from app.core.database import get_db
import asyncio
from app.modules.projects.projects_schema import ProjectStatusEnum
from .parsing_schema import (
    ParsingRequest,
    RepoDetails
    )
from app.modules.projects.projects_service import ProjectService
import os
import shutil
import logging
import traceback
from contextlib import contextmanager
from typing import Dict, Tuple
from app.modules.github.github_service import GithubService
from github import Github
from app.modules.parsing.graph_construction.parsing_helper import ParseHelper, ParsingServiceError
from app.modules.parsing.graph_construction.parsing_service import ParsingService
from uuid6 import uuid7

router = APIRouter()
from git import Repo, GitCommandError


class ParsingAPI:
    @contextmanager
    def change_dir(path):
        old_dir = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(old_dir)
            
    async def clone_or_copy_repository(repo_details: RepoDetails, db: Session, user_id: str) -> Tuple[Any, str, Any]:
        if repo_details.repo_path:
            if not os.path.exists(repo_details.repo_path):
                raise HTTPException(status_code=400, detail="Local repository does not exist on given path")
            repo = Repo(repo_details.repo_path)
            owner = None
            auth = None
        else:
            github_service = GithubService(db)
            response, auth, owner = github_service.get_github_repo_details(repo_details.repo_name)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to get installation ID")
            app_auth = auth.get_installation_auth(response.json()["id"])
            github = Github(auth=app_auth)
            try:
                repo = github.get_repo(repo_details.repo_name)
            except Exception as e:
                raise HTTPException(status_code=400, detail="Repository not found on GitHub")
        
        return repo, owner, auth
    @router.post("/parse")
    async def parse_directory(
        repo_details: ParsingRequest,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth)
    ):
        user_id = user["user_id"]
        project_manager = ProjectService(db)
        project_id = None
        parse_helper = ParseHelper(db)
        project = await project_manager.get_project_from_db(repo_details.repo_name, user_id)
        extracted_dir = None
        if project:
            project_id = project.id
            
        try:
            # Step 1: Validate input
            ParsingAPI.validate_input(repo_details, user_id)
            repo, owner, auth = await ParsingAPI.clone_or_copy_repository(repo_details, db, user_id)

            extracted_dir, project_id = await parse_helper.setup_project_directory(
                    repo, repo_details.branch_name, auth, repo, user_id, project_id
                )
            
            await ParsingService.analyze_directory(extracted_dir, project_id, user_id, db)
            shutil.rmtree(extracted_dir, ignore_errors=True)
            message = "The project has been parsed successfully"
            await project_manager.update_project_status(project_id, ProjectStatusEnum.READY)
             

            return {"message": message, "id": project_id}

        except ParsingServiceError as e:
            message = str(f"{project_id} Failed during parsing: " + e.message)
            await project_manager.update_project_status(project_id, ProjectStatusEnum.ERROR)
            raise HTTPException(status_code=500, detail=message)
        except HTTPException as http_ex:
            if project_id:
                await project_manager.update_project_status(project_id, ProjectStatusEnum.ERROR)
            raise http_ex
        except Exception as e:
            if project_id:
                await project_manager.update_project_status(project_id, ProjectStatusEnum.ERROR)
            tb_str = "".join(traceback.format_exception(None, e, e.__traceback__))
            raise HTTPException(status_code=500, detail=f"{str(e)}\nTraceback: {tb_str}")
        finally:
            if extracted_dir:
                shutil.rmtree(extracted_dir, ignore_errors=True)

    def validate_input(repo_details: ParsingRequest, user_id: str):
        if os.getenv("isDevelopmentMode") != "enabled" and repo_details.repo_path:
            raise HTTPException(status_code=403, detail="Development mode is not enabled, cannot parse local repository.")
        if user_id == os.getenv("defaultUsername") and repo_details.repo_name:
            raise HTTPException(status_code=403, detail="Cannot parse remote repository without auth token")





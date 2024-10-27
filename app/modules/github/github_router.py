from fastapi import Depends, Query
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.utils.APIRouter import APIRouter

from .github_controller import GithubController

router = APIRouter()


@router.get("/github/user-repos")
async def get_user_repos(
    user=Depends(AuthService.check_auth), db: Session = Depends(get_db)
):
    user_repo_list = await GithubController(db).get_user_repos(user=user)
    user_repo_list["repositories"].extend(config_provider.get_demo_repo_list())
    user_repo_list["repositories"] = list(reversed(user_repo_list["repositories"]))
    return user_repo_list


@router.get("/github/get-branch-list")
async def get_branch_list(
    repo_name: str = Query(..., description="Repository name"),
    user=Depends(AuthService.check_auth),
    db: Session = Depends(get_db),
):
    return await GithubController(db).get_branch_list(repo_name=repo_name)

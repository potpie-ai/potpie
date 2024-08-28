from fastapi import Depends, Query

from app.modules.auth.auth_service import AuthService
from app.modules.utils.APIRouter import APIRouter

from .github_controller import GithubController

router = APIRouter()


@router.get("/github/user-repos")
def get_user_repos(user=Depends(AuthService.check_auth)):
    return GithubController.get_user_repos(user=user)


@router.get("/github/get-branch-list")
def get_branch_list(
    repo_name: str = Query(..., description="Repository name"),
    user=Depends(AuthService.check_auth),
):
    return GithubController.get_branch_list(repo_name=repo_name, user=user)

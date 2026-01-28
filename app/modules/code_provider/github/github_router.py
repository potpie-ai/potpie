from fastapi import Depends, Query
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.code_provider.code_provider_controller import CodeProviderController
from app.modules.utils.APIRouter import APIRouter

router = APIRouter()


@router.get("/github/user-repos")
async def get_user_repos(
    user=Depends(AuthService.check_auth), db: Session = Depends(get_db)
):
    user_repo_list = await CodeProviderController(db).get_user_repos(user=user)
    if not config_provider.get_is_development_mode():
        user_repo_list["repositories"].extend(config_provider.get_demo_repo_list())

    # Remove duplicates while preserving order
    seen = set()
    deduped_repos = []
    for repo in reversed(user_repo_list["repositories"]):
        # Create tuple of values to use as hash key
        repo_key = repo["full_name"]

        if repo_key not in seen:
            seen.add(repo_key)
            deduped_repos.append(repo)

    user_repo_list["repositories"] = deduped_repos
    return user_repo_list


@router.get("/github/get-branch-list")
async def get_branch_list(
    repo_name: str = Query(..., description="Repository name"),
    limit: int = Query(None, description="Number of branches to return"),
    offset: int = Query(0, description="Number of branches to skip"),
    search: str = Query(None, description="Search query to filter branches"),
    user=Depends(AuthService.check_auth),
    db: Session = Depends(get_db),
):
    return await CodeProviderController(db).get_branch_list(
        repo_name=repo_name, 
        limit=limit, 
        offset=offset,
        search=search
    )


@router.get("/github/check-public-repo")
async def check_public_repo(
    repo_name: str = Query(..., description="Repository name"),
    user=Depends(AuthService.check_auth),
    db: Session = Depends(get_db),
):
    return await CodeProviderController(db).check_public_repo(repo_name=repo_name)

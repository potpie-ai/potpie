from fastapi import Depends, Query, HTTPException
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.code_provider.code_provider_controller import CodeProviderController
from app.modules.utils.APIRouter import APIRouter

router = APIRouter()


@router.get("/github/user-repos")
async def get_user_repos(
    search: str = Query(None, description="Search query to filter repositories"),
    user=Depends(AuthService.check_auth), 
    db: Session = Depends(get_db)
):
    controller = CodeProviderController(db)
    
    # Get repos (controller handles search filtering internally)
    # We pass search=None initially to get all repos, then filter after adding demo repos
    user_repo_list = await controller.get_user_repos(user=user, search=None)
    
    # Add demo repos if not in development mode
    if not config_provider.get_is_development_mode():
        demo_repos = config_provider.get_demo_repo_list()
        if demo_repos:
            user_repo_list["repositories"].extend(demo_repos)

    # Remove duplicates while preserving order
    seen = set()
    deduped_repos = []
    for repo in reversed(user_repo_list.get("repositories", [])):
        if not isinstance(repo, dict):
            continue
        # Create tuple of values to use as hash key
        repo_key = repo.get("full_name")
        if not repo_key:
            continue

        if repo_key not in seen:
            seen.add(repo_key)
            deduped_repos.append(repo)

    user_repo_list["repositories"] = deduped_repos
    
    # Apply search filter after deduplication and demo repo addition
    # This ensures search works on the complete, deduplicated list
    if search:
        try:
            search_query = controller._normalize_search_query(search)
            if search_query:
                user_repo_list["repositories"] = controller._filter_repositories(
                    user_repo_list["repositories"], search_query
                )
        except HTTPException:
            # Re-raise HTTP exceptions (e.g., query too long)
            raise
        except Exception as e:
            # Log but don't fail - return unfiltered results if filtering fails
            from app.modules.utils.logger import setup_logger
            logger = setup_logger(__name__)
            logger.warning(f"Error filtering repositories: {str(e)}")
    
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
        repo_name=repo_name, limit=limit, offset=offset, search=search
    )


@router.get("/github/check-public-repo")
async def check_public_repo(
    repo_name: str = Query(..., description="Repository name"),
    user=Depends(AuthService.check_auth),
    db: Session = Depends(get_db),
):
    return await CodeProviderController(db).check_public_repo(repo_name=repo_name)

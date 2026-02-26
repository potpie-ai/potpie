"""
Generic provider-agnostic repository API endpoints.

These endpoints mirror the /github/* endpoints but work with any configured
code provider (GitHub, GitLab, GitBucket, etc.) based on the CODE_PROVIDER
environment variable.

The UI should prefer these endpoints over /github/* endpoints to ensure
compatibility with all providers.
"""
import os

from fastapi import Depends, Query, HTTPException
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.code_provider.code_provider_controller import CodeProviderController
from app.modules.utils.APIRouter import APIRouter

router = APIRouter()


@router.get("/repos/provider-info")
async def get_provider_info(
    user=Depends(AuthService.check_auth),
):
    """
    Return the currently configured code provider type and its capabilities.

    This allows the UI to adapt its behavior (icons, labels, auth flow) based
    on the configured provider.
    """
    provider_type = os.getenv("CODE_PROVIDER", "github").lower()
    base_url = os.getenv("CODE_PROVIDER_BASE_URL", "")

    provider_info = {
        "provider": provider_type,
        "display_name": _get_provider_display_name(provider_type),
        "supports_oauth": provider_type == "github",
        "supports_pat": True,
        "base_url": base_url,
        "is_self_hosted": bool(base_url and provider_type != "github"),
    }

    return provider_info


def _get_provider_display_name(provider_type: str) -> str:
    names = {
        "github": "GitHub",
        "gitlab": "GitLab",
        "gitbucket": "GitBucket",
        "bitbucket": "Bitbucket",
        "local": "Local",
    }
    return names.get(provider_type, provider_type.title())


@router.get("/repos/user-repos")
async def get_user_repos(
    search: str = Query(None, description="Search query to filter repositories"),
    limit: int | None = Query(None, ge=1, description="Number of repositories to return"),
    offset: int = Query(0, ge=0, description="Number of repositories to skip"),
    user=Depends(AuthService.check_auth),
    db: Session = Depends(get_db),
):
    """
    Get user repositories from the configured code provider.
    Provider-agnostic equivalent of /github/user-repos.
    """
    controller = CodeProviderController(db)
    user_repo_list = await controller.get_user_repos(user=user, search=None)

    # Add demo repos if not in development mode
    if not config_provider.get_is_development_mode():
        demo_repos = config_provider.get_demo_repo_list()
        if demo_repos:
            user_repo_list["repositories"].extend(demo_repos)

    # Remove duplicates while preserving order
    seen = set()
    deduped_repos = []
    for repo in user_repo_list.get("repositories", []):
        if not isinstance(repo, dict):
            continue
        repo_key = repo.get("full_name")
        if not repo_key:
            continue
        if repo_key not in seen:
            seen.add(repo_key)
            deduped_repos.append(repo)

    user_repo_list["repositories"] = deduped_repos

    # Apply search filter
    if search:
        try:
            search_query = controller._normalize_search_query(search)
            if search_query:
                user_repo_list["repositories"] = controller._filter_repositories(
                    user_repo_list["repositories"], search_query
                )
        except HTTPException:
            raise
        except Exception as e:
            from app.modules.utils.logger import setup_logger
            logger = setup_logger(__name__)
            logger.warning(f"Error filtering repositories: {str(e)}")

    # Pagination
    repos = user_repo_list["repositories"]
    total_count = len(repos)
    paginated_repos = repos[offset : offset + limit] if limit is not None else repos[offset:]
    has_next_page = (offset + (limit or total_count)) < total_count
    return {
        "repositories": paginated_repos,
        "has_next_page": has_next_page,
        "total_count": total_count,
    }


@router.get("/repos/get-branch-list")
async def get_branch_list(
    repo_name: str = Query(..., description="Repository name"),
    limit: int | None = Query(None, ge=1, description="Number of branches to return"),
    offset: int = Query(0, ge=0, description="Number of branches to skip"),
    search: str = Query(None, description="Search query to filter branches"),
    user=Depends(AuthService.check_auth),
    db: Session = Depends(get_db),
):
    """
    Get branch list for a repository.
    Provider-agnostic equivalent of /github/get-branch-list.
    """
    return await CodeProviderController(db).get_branch_list(
        repo_name=repo_name, limit=limit, offset=offset, search=search
    )


@router.get("/repos/check-public-repo")
async def check_public_repo(
    repo_name: str = Query(..., description="Repository name"),
    user=Depends(AuthService.check_auth),
    db: Session = Depends(get_db),
):
    """
    Check if a repository is publicly accessible.
    Provider-agnostic equivalent of /github/check-public-repo.
    """
    return await CodeProviderController(db).check_public_repo(repo_name=repo_name)


@router.get("/repos/repo-structure")
async def get_repo_structure(
    repo_name: str = Query(..., description="Repository name"),
    branch_name: str = Query(..., description="Branch name"),
    user=Depends(AuthService.check_auth),
    db: Session = Depends(get_db),
):
    """
    Get the file/directory structure for a repository.
    Provider-agnostic equivalent of /github/repo-structure.
    """
    return await CodeProviderController(db).get_repo_structure(
        repo_name=repo_name, branch_name=branch_name
    )

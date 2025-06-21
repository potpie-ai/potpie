import os
from functools import wraps

from fastapi import HTTPException
from app.core.config_provider import config_provider


def validate_parsing_input(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract the required arguments from *args or **kwargs
        repo_details = kwargs.get("repo_details")
        user_id = kwargs.get("user_id")

        if repo_details and user_id:
            # If GitHub is not configured, only allow local repositories
            if not config_provider.is_github_configured() and repo_details.repo_name and not repo_details.repo_path:
                raise HTTPException(
                    status_code=403,
                    detail="GitHub is not configured, cannot parse remote repositories. Use repo_path for local repositories.",
                )
            if user_id == os.getenv("defaultUsername") and repo_details.repo_name:
                raise HTTPException(
                    status_code=403,
                    detail="Cannot parse remote repository without auth token",
                )
        return await func(*args, **kwargs)

    return wrapper

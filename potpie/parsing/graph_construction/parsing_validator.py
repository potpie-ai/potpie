import os
from functools import wraps

from fastapi import HTTPException


def validate_parsing_input(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract the required arguments from *args or **kwargs
        repo_details = kwargs.get("repo_details")
        user_id = kwargs.get("user_id")

        if repo_details and user_id:
            if os.getenv("isDevelopmentMode") != "enabled" and repo_details.repo_path:
                raise HTTPException(
                    status_code=403,
                    detail="Development mode is not enabled, cannot parse local repository.",
                )
            if user_id == os.getenv("defaultUsername") and repo_details.repo_name:
                raise HTTPException(
                    status_code=403,
                    detail="Cannot parse remote repository without auth token",
                )
        return await func(*args, **kwargs)

    return wrapper

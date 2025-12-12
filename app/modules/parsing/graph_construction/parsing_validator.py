import os
from functools import wraps

from fastapi import HTTPException


def validate_parsing_input(func):
    """
    Validator for parsing input. Note: Most validation is now handled by
    RepositoryResolver. This validator now only handles auth-related checks.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract the required arguments from *args or **kwargs
        repo_details = kwargs.get("repo_details")
        user = kwargs.get("user")

        if repo_details and user:
            user_id = user.get("user_id") if isinstance(user, dict) else getattr(user, "user_id", None)
            
            # Check if user is trying to parse remote repository without proper auth
            if user_id == os.getenv("defaultUsername"):
                raise HTTPException(
                    status_code=403,
                    detail="Cannot parse remote repository without auth token",
                )
        return await func(*args, **kwargs)

    return wrapper

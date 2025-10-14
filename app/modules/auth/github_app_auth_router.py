"""
GitHub App Authentication Router

Handles GitHub App user token lifecycle endpoints.
"""

import logging
from fastapi import Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.code_provider.github.github_service import GithubService
from app.modules.users.token_service import TokenService
from app.modules.utils.APIRouter import APIRouter

logger = logging.getLogger(__name__)

github_app_auth_router = APIRouter()


class GitHubAppAuthAPI:
    @github_app_auth_router.post("/github-app/generate-token")
    async def generate_github_app_token(
        request: Request,
        response: Response,
        db: Session = Depends(get_db)
    ):
        """
        Generate a GitHub App user token for the authenticated user.

        This endpoint creates a new GitHub App user token that can be used
        for GitHub API calls with higher rate limits and better security.
        """
        try:
            # Get authenticated user
            user_data = await AuthService.check_auth(request, response)
            user_id = user_data["user_id"]

            github_service = GithubService(db)

            # Generate GitHub App user token (TokenService persists info internally)
            _, _ = github_service.generate_github_app_user_token(user_id)

            token_service = TokenService(db)
            token_info = token_service.get_token_info(user_id)

            return JSONResponse(
                content={
                    "success": True,
                    "message": "GitHub App user token generated successfully",
                    "expires_at": token_info.get("expires_at"),
                    "token_type": token_info.get("token_type"),
                },
                status_code=201,
            )

        except HTTPException as he:
            logger.error(
                f"HTTP error generating GitHub App token for user: {he.detail}"
            )
            return JSONResponse(
                content={"success": False, "error": he.detail},
                status_code=he.status_code,
            )
        except Exception as e:
            logger.error(f"Unexpected error generating GitHub App token: {str(e)}")
            return JSONResponse(
                content={
                    "success": False,
                    "error": "Failed to generate GitHub App token",
                },
                status_code=500,
            )

    @github_app_auth_router.get("/github-app/status")
    async def get_github_app_token_status(
        request: Request,
        response: Response,
        db: Session = Depends(get_db)
    ):
        """
        Get the GitHub App token status for the authenticated user.

        Returns information about the user's current token type and migration status.
        """
        try:
            # Get authenticated user
            user_data = await AuthService.check_auth(request, response)
            user_id = user_data["user_id"]

            token_service = TokenService(db)
            github_service = GithubService(db)

            token_info = token_service.get_token_info(user_id)
            token, token_type = github_service.get_best_github_token(user_id)

            return JSONResponse(
                content={
                    "user_id": user_id,
                    "current_token_type": token_type,
                    "token_info": token_info,
                },
                status_code=200,
            )

        except HTTPException as he:
            logger.error(f"HTTP error getting token status: {he.detail}")
            return JSONResponse(
                content={"error": he.detail}, status_code=he.status_code
            )
        except Exception as e:
            logger.error(f"Unexpected error getting token status: {str(e)}")
            return JSONResponse(
                content={"error": "Failed to get token status"}, status_code=500
            )

    @github_app_auth_router.post("/github-app/refresh")
    async def refresh_github_app_token(
        request: Request,
        response: Response,
        db: Session = Depends(get_db)
    ):
        """
        Refresh the GitHub App user token for the authenticated user.

        This endpoint refreshes an existing GitHub App user token if it's
        expired or about to expire.
        """
        try:
            # Get authenticated user
            user_data = await AuthService.check_auth(request, response)
            user_id = user_data["user_id"]

            token_service = TokenService(db)
            refreshed_token = token_service.refresh_token(user_id, "github")

            if refreshed_token:
                token_info = token_service.get_token_info(user_id)
                return JSONResponse(
                    content={
                        "success": True,
                        "message": "Token refreshed successfully",
                        "expires_at": token_info.get("expires_at"),
                        "token_type": token_info.get("token_type"),
                    },
                    status_code=200,
                )

            return JSONResponse(
                content={
                    "success": False,
                    "error": "Failed to refresh token. User may need to reinstall the GitHub App.",
                },
                status_code=400,
            )

        except HTTPException as he:
            logger.error(f"HTTP error refreshing token: {he.detail}")
            return JSONResponse(
                content={"success": False, "error": he.detail},
                status_code=he.status_code,
            )
        except Exception as e:
            logger.error(f"Unexpected error refreshing token: {str(e)}")
            return JSONResponse(
                content={"success": False, "error": "Failed to refresh token"},
                status_code=500,
            )

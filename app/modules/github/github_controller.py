from fastapi import Depends

from app.modules.auth.auth_service import AuthService
from app.modules.github.github_service import GithubService


class GithubController:
    @staticmethod
    def get_user_repos(user=Depends(AuthService.check_auth)):
        return GithubService.get_repos_for_user(user)

    @staticmethod
    def get_branch_list(repo_name: str, user=Depends(AuthService.check_auth)):
        return GithubService.get_branch_list(repo_name)

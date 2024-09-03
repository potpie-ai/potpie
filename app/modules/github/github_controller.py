from fastapi import Depends
from sqlalchemy.orm import Session

from app.modules.auth.auth_service import AuthService
from app.modules.github.github_service import GithubService


class GithubController:
    def __init__(self, db: Session):
        self.github_service = GithubService(db)

    def get_user_repos(self, user):
        user_id = user["user_id"]
        return self.github_service.get_repos_for_user(user_id)

    @staticmethod
    def get_branch_list(repo_name: str, user=Depends(AuthService.check_auth)):
        return GithubService.get_branch_list(repo_name)

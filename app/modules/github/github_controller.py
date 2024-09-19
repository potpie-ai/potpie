from sqlalchemy.orm import Session

from app.modules.github.github_service import GithubService


class GithubController:
    def __init__(self, db: Session):
        self.github_service = GithubService(db)

    def get_user_repos(self, user):
        user_id = user["user_id"]
        return self.github_service.get_repos_for_user(user_id)

    def get_branch_list(self, repo_name: str):
        return self.github_service.get_branch_list(repo_name)

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.modules.code_provider.github.github_service import GithubService


class GithubController:
    def __init__(self, db: Session):
        self.github_service = GithubService(db)

    async def get_user_repos(self, user):
        user_id = user["user_id"]
        return await self.github_service.get_combined_user_repos(user_id)

    async def get_branch_list(self, repo_name: str):
        return await self.github_service.get_branch_list(repo_name)

    async def check_public_repo(self, repo_name: str):
        is_public = await self.github_service.check_public_repo(repo_name)
        if not is_public:
            raise HTTPException(status_code=403, detail="Repository is not found")
        return {"is_public": is_public}

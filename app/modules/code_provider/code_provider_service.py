import os
from typing import Optional

from app.modules.code_provider.github.github_service import GithubService
from app.modules.code_provider.local_repo.local_repo_service import LocalRepoService


class CodeProviderService:
    def __init__(self, sql_db):
        self.sql_db = sql_db
        self.service_instance = self._get_service_instance()

    def _get_service_instance(self):
        if os.getenv("isDevelopmentMode") == "enabled":
            return LocalRepoService(self.sql_db)
        else:
            return GithubService(self.sql_db)

    def get_repo(self, repo_name):
        return self.service_instance.get_repo(repo_name)

    async def get_project_structure_async(self, project_id, path: Optional[str] = None):
        return await self.service_instance.get_project_structure_async(project_id, path)

    def get_file_content(
        self, repo_name, file_path, start_line, end_line, branch_name, project_id
    ):
        return self.service_instance.get_file_content(
            repo_name, file_path, start_line, end_line, branch_name, project_id
        )

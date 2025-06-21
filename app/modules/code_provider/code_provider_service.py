import os
from typing import Optional

from app.core.config_provider import config_provider
from app.modules.code_provider.github.github_service import GithubService
from app.modules.code_provider.local_repo.local_repo_service import LocalRepoService


class CodeProviderService:
    def __init__(self, sql_db):
        self.sql_db = sql_db
        self.github_service = GithubService(sql_db) if config_provider.is_github_configured() else None
        self.local_service = LocalRepoService(sql_db)

    def get_repo(self, repo_name):
        # If it's a local path, use local service
        if os.path.exists(repo_name) and os.path.isdir(repo_name):
            return self.local_service.get_repo(repo_name)
        # If GitHub is configured, use GitHub service for remote repos
        elif self.github_service:
            return self.github_service.get_repo(repo_name)
        else:
            raise Exception("GitHub is not configured and no local repository found")

    async def get_project_structure_async(self, project_id, path: Optional[str] = None):
        # Try GitHub service first if available, fallback to local service
        if self.github_service:
            try:
                return await self.github_service.get_project_structure_async(project_id, path)
            except Exception:
                # Fallback to local service if GitHub service fails
                pass
        return await self.local_service.get_project_structure_async(project_id, path)

    def get_file_content(
        self,
        repo_name,
        file_path,
        start_line,
        end_line,
        branch_name,
        project_id,
        commit_id,
    ):
        # If it's a local path, use local service
        if os.path.exists(repo_name) and os.path.isdir(repo_name):
            return self.local_service.get_file_content(
                repo_name, file_path, start_line, end_line, branch_name, project_id, commit_id
            )
        # If GitHub is configured, use GitHub service for remote repos
        elif self.github_service:
            return self.github_service.get_file_content(
                repo_name, file_path, start_line, end_line, branch_name, project_id, commit_id
            )
        else:
            raise Exception("GitHub is not configured and no local repository found")

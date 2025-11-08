import os
from typing import Optional

from app.core.config_provider import config_provider
from app.modules.code_provider.github.github_service import GithubService
from app.modules.code_provider.local_repo.local_repo_service import LocalRepoService
from app.modules.projects.projects_service import ProjectService


class CodeProviderService:
    def __init__(self, sql_db):
        self.sql_db = sql_db
        # In dev mode with GitHub configured, support both local and GitHub repos
        self.github_service = (
            GithubService(sql_db) if config_provider.is_github_configured() else None
        )
        self.local_service = LocalRepoService(sql_db)

    def get_repo(self, repo_name):
        # If it looks like a file path, use local service
        if os.path.exists(repo_name):
            return self.local_service.get_repo(repo_name)
        # Otherwise, use GitHub service if configured
        elif self.github_service:
            return self.github_service.get_repo(repo_name)
        else:
            raise Exception(
                f"GitHub not configured and {repo_name} is not a local path"
            )

    async def get_project_structure_async(self, project_id, path: Optional[str] = None):
        # Check if project has a local repo_path
        project_service = ProjectService(self.sql_db)
        project = await project_service.get_project_from_db_by_id(project_id)

        if project and project.get("repo_path"):
            # Local repository
            return await self.local_service.get_project_structure_async(
                project_id, path
            )
        elif self.github_service:
            # GitHub repository
            return await self.github_service.get_project_structure_async(
                project_id, path
            )
        else:
            raise Exception(
                f"GitHub not configured and project {project_id} has no local repo_path"
            )

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
        # Check if project has a local repo_path
        project_service = ProjectService(self.sql_db)
        project = project_service.get_project_from_db_by_id_sync(project_id)

        if project and project.get("repo_path"):
            # Local repository
            return self.local_service.get_file_content(
                repo_name,
                file_path,
                start_line,
                end_line,
                branch_name,
                project_id,
                commit_id,
            )
        elif self.github_service:
            # GitHub repository
            return self.github_service.get_file_content(
                repo_name,
                file_path,
                start_line,
                end_line,
                branch_name,
                project_id,
                commit_id,
            )
        else:
            raise Exception(
                f"GitHub not configured and project {project_id} has no local repo_path"
            )

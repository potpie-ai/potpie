import logging
import os
from typing import Any, Dict, Tuple

import chardet
import requests
from fastapi import HTTPException
from github import Github
from github.Auth import AppAuth
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.projects.projects_service import ProjectService
from app.modules.users.user_service import UserService

logger = logging.getLogger(__name__)


class GithubService:
    def __init__(self, db: Session):
        self.db = db
        self.project_manager = ProjectService(db)

    def get_github_repo_details(self, repo_name: str) -> Tuple[Github, Dict, str]:
        private_key = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            + config_provider.get_github_key()
            + "\n-----END RSA PRIVATE KEY-----\n"
        )
        app_id = os.environ["GITHUB_APP_ID"]
        auth = AppAuth(app_id=app_id, private_key=private_key)
        jwt = auth.create_jwt()
        owner = repo_name.split("/")[0]

        url = f"https://api.github.com/repos/{owner}/{repo_name.split('/')[1]}/installation"
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {jwt}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to get installation ID")

        app_auth = auth.get_installation_auth(response.json()["id"])
        github = Github(auth=app_auth)

        return github, response.json(), owner

    def get_public_github_repo(self, repo_name: str) -> Tuple[Dict[str, Any], str]:
        owner, repo = repo_name.split("/")
        url = f"https://api.github.com/repos/{owner}/{repo}"
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail="Failed to fetch public repository",
            )

        return response.json(), owner

    def get_file_content(
        self, repo_name: str, file_path: str, start_line: int, end_line: int
    ) -> str:
        logger.info(f"Attempting to access file: {file_path} in repo: {repo_name}")

        # Clean up the file path
        path_parts = file_path.split("/")
        if len(path_parts) > 1 and "-" in path_parts[0]:
            # Remove the first part if it contains a dash (likely a commit hash or branch name)
            path_parts = path_parts[1:]
        clean_file_path = "/".join(path_parts)

        logger.info(f"Cleaned file path: {clean_file_path}")

        try:
            # Try public access first
            github = self.get_public_github_instance()
            repo = github.get_repo(repo_name)
            try:
                file_contents = repo.get_contents(clean_file_path)
            except Exception as file_error:
                logger.info(f"Failed to access file in public repo: {str(file_error)}")
                raise  # Re-raise to be caught by the outer try-except
        except Exception as public_error:
            logger.info(f"Failed to access public repo: {str(public_error)}")
            # If public access fails, try authenticated access
            try:
                github, repo = self.get_repo(repo_name)
                try:
                    file_contents = repo.get_contents(clean_file_path)
                except Exception as file_error:
                    logger.error(
                        f"Failed to access file in private repo: {str(file_error)}"
                    )
                    raise HTTPException(
                        status_code=404,
                        detail=f"File not found or inaccessible: {clean_file_path}",
                    )
            except Exception as private_error:
                logger.error(f"Failed to access private repo: {str(private_error)}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Repository not found or inaccessible: {repo_name}",
                )

        if isinstance(file_contents, list):
            raise HTTPException(
                status_code=400, detail="Provided path is a directory, not a file"
            )

        try:
            content_bytes = file_contents.decoded_content
            encoding = self._detect_encoding(content_bytes)
            decoded_content = content_bytes.decode(encoding)
            lines = decoded_content.splitlines()

            if start_line == end_line == 0:
                return decoded_content
            
            selected_lines = lines[start_line:end_line]
            return "\n".join(selected_lines)
        except Exception as e:
            logger.error(
                f"Error processing file content for {repo_name}/{clean_file_path}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file content: {str(e)}",
            )

    def _get_repo(self, repo_name: str) -> Tuple[Github, Any]:
        github, _, _ = self.get_github_repo_details(repo_name)
        return github, github.get_repo(repo_name)

    @staticmethod
    def _detect_encoding(content_bytes: bytes) -> str:
        detection = chardet.detect(content_bytes)
        encoding = detection["encoding"]
        confidence = detection["confidence"]

        if not encoding or confidence < 0.5:
            raise HTTPException(
                status_code=400,
                detail="Unable to determine file encoding or low confidence",
            )

        return encoding

    def get_repos_for_user(self, user_id: str):
        try:
            user_service = UserService(self.db)
            user = user_service.get_user_by_uid(user_id)

            if user is None:
                raise HTTPException(status_code=404, detail="User not found")

            github_username = user.provider_username

            if not github_username:
                raise HTTPException(
                    status_code=400, detail="GitHub username not found for this user"
                )

            # Use GitHub App authentication
            github, _, _ = self.get_github_repo_details(github_username)

            repos = []
            for repo in github.get_user(github_username).get_repos():
                repos.append(
                    {
                        "id": repo.id,
                        "name": repo.name,
                        "full_name": repo.full_name,
                        "private": repo.private,
                        "url": repo.html_url,
                        "owner": repo.owner.login,
                    }
                )

            return {"repositories": repos}

        except Exception as e:
            logger.error(f"Failed to fetch repositories: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Failed to fetch repositories: {str(e)}"
            )

    def get_branch_list(self, repo_name: str):
        try:
            github, repo = self.get_repo(repo_name)
            branches = repo.get_branches()
            branch_list = [branch.name for branch in branches]
            return {"branches": branch_list}
        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(
                f"Error fetching branches for repo {repo_name}: {str(e)}", exc_info=True
            )
            raise HTTPException(
                status_code=404,
                detail=f"Repository not found or error fetching branches: {str(e)}",
            )

    @staticmethod
    def get_public_github_instance():
        return Github()

    def get_repo(self, repo_name: str) -> Tuple[Github, Any]:
        try:
            # Try public access first
            github = self.get_public_github_instance()
            repo = github.get_repo(repo_name)
            return github, repo
        except Exception as public_error:
            logger.info(f"Failed to access public repo: {str(public_error)}")
            # If public access fails, try authenticated access
            try:
                github, _, _ = self.get_github_repo_details(repo_name)
                repo = github.get_repo(repo_name)
                return github, repo
            except Exception as private_error:
                logger.error(f"Failed to access private repo: {str(private_error)}")
                raise HTTPException(
                    status_code=404,
                    detail="Repository not found or inaccessible on GitHub",
                )

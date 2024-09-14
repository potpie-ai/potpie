import logging
import os
from typing import Any, Dict, Tuple

import chardet
import requests
from fastapi import HTTPException
from github import Github
from github.Auth import AppAuth
from github.GithubException import GithubException, UnknownObjectException
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
        logger.info(f"Function: get_github_repo_details, Repo: {repo_name}")
        private_key = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            + config_provider.get_github_key()
            + "\n-----END RSA PRIVATE KEY-----\n"
        )
        app_id = os.environ["GITHUB_APP_ID"]
        auth = AppAuth(app_id=app_id, private_key=private_key)
        jwt = auth.create_jwt()

        url = f"https://api.github.com/repos/{repo_name}/installation"
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

        owner = repo_name.split("/")[0]  # Extract owner from repo_name if needed

        return github, response.json(), owner

    def get_public_github_repo(self, repo_name: str) -> Tuple[Dict[str, Any], str]:
        logger.info(f"Function: get_public_github_repo, Repo: {repo_name}")
        url = f"https://api.github.com/repos/{repo_name}"
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

        owner = repo_name.split("/")[0]
        return response.json(), owner

    def get_file_content(
        self, repo_name: str, file_path: str, start_line: int, end_line: int
    ) -> Dict[str, Any]:
        logger.info(f"Function: get_file_content, Repo: {repo_name}")
        logger.info(f"Attempting to access file: {file_path} in repo: {repo_name}")

        # Clean up the file path
        path_parts = file_path.split("/")
        if len(path_parts) > 1 and "-" in path_parts[0]:
            # Remove the first part if it contains a dash (likely a commit hash or branch name)
            path_parts = path_parts[1:]
        clean_file_path = "/".join(path_parts)

        logger.info(f"Cleaned file path: {clean_file_path}")

        try:
            github, repo, error = self.get_repo(repo_name)
            if error:
                return {"content": "", "error": error}

            try:
                file_contents = repo.get_contents(clean_file_path)
            except UnknownObjectException:
                return {"content": "", "error": f"File not found: {clean_file_path}"}
            except GithubException as ge:
                return {"content": "", "error": f"GitHub error: {str(ge)}"}

            if isinstance(file_contents, list):
                return {"content": "", "error": "Provided path is a directory, not a file"}

            try:
                content_bytes = file_contents.decoded_content
                encoding = self._detect_encoding(content_bytes)
                decoded_content = content_bytes.decode(encoding)
                lines = decoded_content.splitlines()

                selected_lines = lines[start_line:end_line]
                return {"content": "\n".join(selected_lines)}
            except Exception as e:
                logger.error(
                    f"Error processing file content for {repo_name}/{clean_file_path}: {e}",
                    exc_info=True,
                )
                return {"content": "", "error": f"Error processing file content: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error in get_file_content: {str(e)}", exc_info=True)
            return {"content": "", "error": f"Unexpected error: {str(e)}"}

    def _get_repo(self, repo_name: str) -> Tuple[Github, Any]:
        logger.info(f"Function: _get_repo, Repo: {repo_name}")
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

    def get_repos_for_user(self, user_id: str) -> Dict[str, Any]:
        try:
            user_service = UserService(self.db)
            user = user_service.get_user_by_uid(user_id)

            if user is None:
                return {"repositories": [], "error": "User not found"}

            github_username = user.provider_username

            if not github_username:
                return {"repositories": [], "error": "GitHub username not found for this user"}

            # Use GitHub API directly
            url = f"https://api.github.com/users/{github_username}/repos"
            headers = {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                return {"repositories": [], "error": f"Failed to fetch repositories: HTTP {response.status_code}"}

            repos = []
            for repo in response.json():
                repos.append(
                    {
                        "id": repo["id"],
                        "name": repo["name"],
                        "full_name": repo["full_name"],
                        "private": repo["private"],
                        "url": repo["html_url"],
                        "owner": repo["owner"]["login"],
                    }
                )

            return {"repositories": repos}

        except Exception as e:
            logger.error(f"Failed to fetch repositories: {str(e)}", exc_info=True)
            return {"repositories": [], "error": f"Failed to fetch repositories: {str(e)}"}

    def get_branch_list(self, repo_name: str):
        logger.info(f"Function: get_branch_list, Repo: {repo_name}")
        try:
            github, repo = self.get_repo(repo_name)
            if github is None or repo is None:
                return {"branches": [], "error": "Repository not found or inaccessible"}
            
            branches = repo.get_branches()
            branch_list = [branch.name for branch in branches]
            return {"branches": branch_list}
        except Exception as e:
            logger.error(
                f"Error fetching branches for repo {repo_name}: {str(e)}", exc_info=True
            )
            return {"branches": [], "error": f"Error fetching branches: {str(e)}"}

    @staticmethod
    def get_public_github_instance():
        return Github()

    def get_repo(self, repo_name: str) -> Tuple[Github, Any, str]:
        logger.info(f"Function: get_repo, Repo: {repo_name}")
        try:
            # Try authenticated access first
            github, _, _ = self.get_github_repo_details(repo_name)
            repo = github.get_repo(repo_name)
            return github, repo, None
        except Exception as private_error:
            logger.info(f"Failed to access private repo: {str(private_error)}")
            # If authenticated access fails, try public access
            try:
                github = self.get_public_github_instance()
                repo = github.get_repo(repo_name)
                return github, repo, None
            except Exception as public_error:
                error_msg = f"Repository not found or inaccessible: {str(public_error)}"
                logger.error(error_msg)
                return None, None, error_msg

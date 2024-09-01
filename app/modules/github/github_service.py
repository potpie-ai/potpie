import base64
import logging
import os

import requests
from fastapi import HTTPException
from github import Github
from github.Auth import AppAuth
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.projects.projects_schema import ProjectStatusEnum
from app.modules.projects.projects_service import ProjectService

logger = logging.getLogger(__name__)


class GithubService:
    # Start Generation Here
    def __init__(self, db: Session):
        self.project_manager = ProjectService(db)

    @staticmethod
    def get_github_repo_details(repo_name: str):
        private_key = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            + config_provider.get_github_key()
            + "\n-----END RSA PRIVATE KEY-----\n"
        )
        app_id = os.environ["GITHUB_APP_ID"]
        auth = AppAuth(app_id=app_id, private_key=private_key)
        jwt = auth.create_jwt()
        owner = repo_name.split("/")[0]
        repo = repo_name.split("/")[1]
        url = f"https://api.github.com/repos/{owner}/{repo}/installation"
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

        return github, response, auth, owner

    @staticmethod
    def check_is_commit_added(repo_details, project_details, branch_name):
        branch = repo_details.get_branch(branch_name)
        latest_commit_sha = branch.commit.sha
        if (
            latest_commit_sha == project_details[3]
            and project_details[4] == ProjectStatusEnum.READY
        ):
            return False
        else:
            return True

    @staticmethod
    def fetch_method_from_repo(node, db):
        method_content = None
        github = None
        try:
            project_id = node["project_id"]
            project_manager = ProjectService(db)
            repo_details = project_manager.get_repo_and_branch_name(
                project_id=project_id
            )
            repo_name = repo_details[0]
            branch_name = repo_details[1]

            file_path = node["id"].split(":")[0].lstrip("/")
            start_line = node["start"]
            end_line = node["end"]

            _,response, auth, _ = GithubService.get_github_repo_details(repo_name)

            if response.status_code != 200:
                raise HTTPException(
                    status_code=400, detail="Failed to get installation ID"
                )

            app_auth = auth.get_installation_auth(response.json()["id"])
            github = Github(auth=app_auth)
            repo = github.get_repo(repo_name)
            file_contents = repo.get_contents(
                file_path.replace("\\", "/"), ref=branch_name
            )
            decoded_content = base64.b64decode(file_contents.content).decode("utf-8")
            lines = decoded_content.split("\n")
            method_lines = lines[start_line - 1 : end_line]
            method_content = "\n".join(method_lines)

        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)

        return method_content

    @staticmethod
    def get_repos_for_user(user):
        try:
            private_key = (
                "-----BEGIN RSA PRIVATE KEY-----\n"
                + config_provider.get_github_key()
                + "\n-----END RSA PRIVATE KEY-----\n"
            )
            app_id = os.environ["GITHUB_APP_ID"]
            auth = AppAuth(app_id=app_id, private_key=private_key)
            jwt = auth.create_jwt()

            url = "https://api.github.com/app/installations"
            headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {jwt}",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400, detail="Failed to get installations"
                )

            installations = response.json()
            repos = []

            for installation in installations:
                app_auth = auth.get_installation_auth(installation["id"])
                github = Github(auth=app_auth)
                repos_url = installation["repositories_url"]
                repos_response = requests.get(
                    repos_url, headers={"Authorization": f"Bearer {app_auth.token}"}
                )
                if repos_response.status_code == 200:
                    repos.extend(repos_response.json().get("repositories", []))
                else:
                    logger.error(
                        f"Failed to fetch repositories for installation ID {installation['id']}"
                    )
            repo_list = [
                {
                    "id": repo["id"],
                    "name": repo["name"],
                    "full_name": repo["full_name"],
                    "private": repo["private"],
                    "url": repo["html_url"],
                    "owner": repo["owner"]["login"],
                }
                for repo in repos
            ]
            return {"repositories": repo_list}
        except Exception as e:
            logger.error(f"Failed to fetch repositories: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Failed to fetch repositories: {str(e)}"
            )

    @staticmethod
    def get_branch_list(repo_name: str):
        try:
            github_client, _, _, _ = GithubService.get_github_repo_details(repo_name)
            repo = github_client.get_repo(repo_name)
            branches = repo.get_branches()
            branch_list = [branch.name for branch in branches]
            return {"branches": branch_list}
        except Exception as e:
            logger.error(
                f"Error fetching branches for repo {repo_name}: {str(e)}", exc_info=True
            )
            raise HTTPException(
                status_code=404,
                detail=f"Repository not found or error fetching branches: {str(e)}",
            )

    @staticmethod
    def get_public_github_repo(repo_name: str):
        owner = repo_name.split("/")[0]
        repo = repo_name.split("/")[1]
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

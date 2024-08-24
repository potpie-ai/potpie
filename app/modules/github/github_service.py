import base64
import logging
import os

import requests
from fastapi import HTTPException
from github import Github
from github.Auth import AppAuth
from sqlalchemy.orm import Session

from app.core.config import config_provider
from app.modules.projects.projects_schema import ProjectStatusEnum
from app.modules.projects.projects_service import ProjectService

logger = logging.getLogger(__name__)


class GithubService:
    # Start Generation Here
    def __init__(self, db: Session):
        self.project_manager = ProjectService(db)

    @staticmethod
    def get_github_repo_details(repo_name):
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

        return requests.get(url, headers=headers), auth, owner

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

            response, auth, _ = GithubService.get_github_repo_details(repo_name)

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

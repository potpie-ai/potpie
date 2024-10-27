import logging
import os
import random
from typing import Any, Dict, List, Tuple

import chardet
import requests
from fastapi import HTTPException
from github import Github
from github.Auth import AppAuth
from redis import Redis
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.projects.projects_service import ProjectService
from app.modules.users.user_model import User

logger = logging.getLogger(__name__)


class GithubService:
    gh_token_list: List[str] = []

    @classmethod
    def initialize_tokens(cls):
        token_string = os.getenv("GH_TOKEN_LIST", "")
        cls.gh_token_list = [
            token.strip() for token in token_string.split(",") if token.strip()
        ]
        if not cls.gh_token_list:
            raise ValueError(
                "GitHub token list is empty or not set in environment variables"
            )
        logger.info(f"Initialized {len(cls.gh_token_list)} GitHub tokens")

    def __init__(self, db: Session):
        self.db = db
        self.project_manager = ProjectService(db)
        if not GithubService.gh_token_list:
            GithubService.initialize_tokens()
        self.redis = Redis.from_url(config_provider.get_redis_url())

    def get_github_repo_details(self, repo_name: str) -> Tuple[Github, Dict, str]:
        logger.info(f"Getting GitHub repo details for: {repo_name}")
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
            logger.error(
                f"Failed to get installation ID for {repo_name}. Status code: {response.status_code}, Response: {response.text}"
            )
            raise HTTPException(
                status_code=400, detail=f"Failed to get installation ID for {repo_name}"
            )

        logger.info(f"Successfully got installation ID for {repo_name}")
        app_auth = auth.get_installation_auth(response.json()["id"])
        github = Github(auth=app_auth)

        return github, response.json(), owner

    def get_file_content(
        self,
        repo_name: str,
        file_path: str,
        start_line: int,
        end_line: int,
        branch_name: str,
    ) -> str:
        logger.info(f"Attempting to access file: {file_path} in repo: {repo_name}")

        # Clean up the file path
        path_parts = file_path.split("/")
        if len(path_parts) > 1 and "-" in path_parts[0]:
            path_parts = path_parts[1:]
        clean_file_path = "/".join(path_parts)

        logger.info(f"Cleaned file path: {clean_file_path}")

        try:
            # Try authenticated access first
            github, repo = self.get_repo(repo_name)
            file_contents = repo.get_contents(clean_file_path, ref=branch_name)
        except Exception as private_error:
            logger.info(f"Failed to access private repo: {str(private_error)}")
            # If authenticated access fails, try public access
            try:
                github = self.get_public_github_instance()
                repo = github.get_repo(repo_name)
                file_contents = repo.get_contents(clean_file_path)
            except Exception as public_error:
                logger.error(f"Failed to access public repo: {str(public_error)}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Repository or file not found or inaccessible: {repo_name}/{clean_file_path}",
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

            if (start_line == end_line == 0) or (start_line == end_line == None):
                return decoded_content
            # added -2 to start and end line to include the function definition/ decorator line
            selected_lines = lines[start_line - 2 : end_line]
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

    def get_github_oauth_token(self, uid: str) -> str:
        user = self.db.query(User).filter(User.uid == uid).first()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return user.provider_info["access_token"]

    async def get_repos_for_user(self, user_id: str):
        try:
            logger.info(f"Getting repositories for user: {user_id}")
            user = self.db.query(User).filter(User.uid == user_id).first()
            if user is None:
                raise HTTPException(status_code=404, detail="User not found")
            logger.info(f"User found: {user}")

            firebase_uid = user.uid  # Assuming `uid` is the Firebase UID
            github_username = user.provider_username

            if not github_username:
                raise HTTPException(
                    status_code=400, detail="GitHub username not found for this user"
                )

            # Retrieve GitHub OAuth token from Firestore
            github_oauth_token = self.get_github_oauth_token(firebase_uid)
            if not github_oauth_token:
                raise HTTPException(
                    status_code=400, detail="GitHub OAuth token not found for this user"
                )

            # Initialize GitHub client with user's OAuth token
            user_github = Github(github_oauth_token)
            user_orgs = user_github.get_user().get_orgs()
            org_logins = [org.login.lower() for org in user_orgs]

            # Authenticate as GitHub App
            private_key = (
                "-----BEGIN RSA PRIVATE KEY-----\n"
                + config_provider.get_github_key()
                + "\n-----END RSA PRIVATE KEY-----\n"
            )
            app_id = os.environ["GITHUB_APP_ID"]

            auth = AppAuth(app_id=app_id, private_key=private_key)
            jwt = auth.create_jwt()
            installations_url = "https://api.github.com/app/installations"
            headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {jwt}",
                "X-GitHub-Api-Version": "2022-11-28",
            }

            response = requests.get(installations_url, headers=headers)

            if response.status_code != 200:
                logger.error(f"Failed to get installations. Response: {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to get installations: {response.text}",
                )

            all_installations = response.json()

            # Filter installations: user's personal installation + org installations where user is a member
            user_installations = []
            for installation in all_installations:
                account = installation["account"]
                account_login = account["login"].lower()
                account_type = account["type"]  # 'User' or 'Organization'

                if account_type == "User" and account_login == github_username.lower():
                    user_installations.append(installation)
                elif account_type == "Organization" and account_login in org_logins:
                    user_installations.append(installation)

            logger.info(
                f"Filtered installations: {[inst['id'] for inst in user_installations]}"
            )

            repos = []
            for installation in user_installations:
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
                        f"Failed to fetch repositories for installation ID {installation['id']}. Response: {repos_response.text}"
                    )

            # Remove duplicate repositories if any
            unique_repos = {repo["id"]: repo for repo in repos}.values()

            repo_list = [
                {
                    "id": repo["id"],
                    "name": repo["name"],
                    "full_name": repo["full_name"],
                    "private": repo["private"],
                    "url": repo["html_url"],
                    "owner": repo["owner"]["login"],
                }
                for repo in unique_repos
            ]

            return {"repositories": repo_list}

        except Exception as e:
            logger.error(f"Failed to fetch repositories: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Failed to fetch repositories: {str(e)}"
            )

    async def get_branch_list(self, repo_name: str):
        try:
            github, repo = self.get_repo(repo_name)
            default_branch = repo.default_branch
            branches = repo.get_branches()
            branch_list = [
                branch.name for branch in branches if branch.name != default_branch
            ]
            return {"branches": [default_branch] + branch_list}
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

    @classmethod
    def get_public_github_instance(cls):
        if not cls.gh_token_list:
            cls.initialize_tokens()
        token = random.choice(cls.gh_token_list)
        return Github(token)

    def get_repo(self, repo_name: str) -> Tuple[Github, Any]:
        logger.info(f"Attempting to access repo: {repo_name}")
        try:
            # Try authenticated access first
            logger.info(f"Trying authenticated access for repo: {repo_name}")
            github, _, _ = self.get_github_repo_details(repo_name)
            repo = github.get_repo(repo_name)
            logger.info(
                f"Successfully accessed repo {repo_name} with authenticated access"
            )
            return github, repo
        except Exception as private_error:
            logger.info(
                f"Failed to access private repo {repo_name}: {str(private_error)}"
            )
            # If authenticated access fails, try public access
            try:
                logger.info(f"Trying public access for repo: {repo_name}")
                github = self.get_public_github_instance()
                repo = github.get_repo(repo_name)
                logger.info(
                    f"Successfully accessed repo {repo_name} with public access"
                )
                return github, repo
            except Exception as public_error:
                logger.error(
                    f"Failed to access public repo {repo_name}: {str(public_error)}"
                )
                raise HTTPException(
                    status_code=404,
                    detail=f"Repository {repo_name} not found or inaccessible on GitHub",
                )

    def get_project_structure(self, project_id: str) -> str:
        logger.info(f"Fetching project structure for project ID: {project_id}")

        # Try to get the structure from Redis cache
        cache_key = f"project_structure:{project_id}"
        cached_structure = self.redis.get(cache_key)

        if cached_structure:
            logger.info(
                f"Project structure found in cache for project ID: {project_id}"
            )
            return cached_structure.decode("utf-8")

        project = self.project_manager.get_project_from_db_by_id_sync(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        repo_name = project["project_name"]
        if not repo_name:
            raise HTTPException(
                status_code=400, detail="Project has no associated GitHub repository"
            )

        try:
            github, repo = self.get_repo(repo_name)
            structure = self._fetch_repo_structure(repo)
            formatted_structure = self._format_tree_structure(structure)

            # Cache the formatted structure in Redis
            self.redis.setex(cache_key, 3600, formatted_structure)  # Cache for 1 hour

            return formatted_structure
        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(
                f"Error fetching project structure for {repo_name}: {str(e)}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch project structure: {str(e)}",
            )

    def _fetch_repo_structure(self, repo: Any, path: str = "") -> Dict[str, Any]:
        structure = {
            "type": "directory",
            "name": path.split("/")[-1] or repo.name,
            "children": [],
        }

        try:
            contents = repo.get_contents(path)
        except Exception as e:
            logger.error(f"Error fetching contents for path {path}: {str(e)}")
            return structure

        for item in contents:
            if item.type == "dir":
                structure["children"].append(
                    self._fetch_repo_structure(repo, item.path)
                )
            else:
                structure["children"].append(
                    {
                        "type": "file",
                        "name": item.name,
                        "path": item.path,
                    }
                )

        return structure

    def _format_tree_structure(
        self, structure: Dict[str, Any], prefix: str = ""
    ) -> str:
        output = []
        name = structure["name"]
        if prefix:
            output.append(f"{prefix[:-1]}└── {name}")
        else:
            output.append(name)

        children = sorted(structure["children"], key=lambda x: (x["type"], x["name"]))
        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            new_prefix = prefix + ("    " if is_last else "│   ")

            if child["type"] == "directory":
                output.append(self._format_tree_structure(child, new_prefix))
            else:
                output.append(f"{new_prefix[:-1]}└── {child['name']}")

        return "\n".join(output)

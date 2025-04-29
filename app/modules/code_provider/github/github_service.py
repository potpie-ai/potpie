import asyncio
import logging
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import chardet
import git
import requests
from fastapi import HTTPException
from github import Github
from github.Auth import AppAuth
from sqlalchemy import func
from sqlalchemy.orm import Session
from redis import Redis

from app.core.config_provider import config_provider
from app.modules.projects.projects_model import Project
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
        self.max_workers = 10
        self.max_depth = 4
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.is_development_mode = config_provider.get_is_development_mode()

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
            raise HTTPException(
                status_code=400, detail=f"Failed to get installation ID for {repo_name}"
            )

        app_auth = auth.get_installation_auth(response.json()["id"])
        github = Github(auth=app_auth)

        return github, response.json(), owner

    def get_github_app_client(self, repo_name: str) -> Github:
        try:
            # Try authenticated access first
            private_key = (
                "-----BEGIN RSA PRIVATE KEY-----\n"
                + config_provider.get_github_key()
                + "\n-----END RSA PRIVATE KEY-----\n"
            )
            app_id = os.environ["GITHUB_APP_ID"]
            auth = AppAuth(app_id=app_id, private_key=private_key)
            jwt = auth.create_jwt()

            # Get installation ID
            url = f"https://api.github.com/repos/{repo_name}/installation"
            headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {jwt}",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                raise Exception(f"Failed to get installation ID for {repo_name}")

            app_auth = auth.get_installation_auth(response.json()["id"])
            return Github(auth=app_auth)
        except Exception as private_error:
            logging.info(f"Failed to access private repo: {str(private_error)}")
            # If authenticated access fails, try public access
            try:
                return self.get_public_github_instance()
            except Exception as public_error:
                logging.error(f"Failed to access public repo: {str(public_error)}")
                raise Exception(
                    f"Repository {repo_name} not found or inaccessible on GitHub"
                )

    def get_file_content(
        self,
        repo_name: str,
        file_path: str,
        start_line: int,
        end_line: int,
        branch_name: str,
        project_id: str,
    ) -> str:
        logger.info(f"Attempting to access file: {file_path} in repo: {repo_name}")

        try:
            # Try authenticated access first
            github, repo = self.get_repo(repo_name)
            file_contents = repo.get_contents(file_path, ref=branch_name)
        except Exception as private_error:
            logger.info(f"Failed to access private repo: {str(private_error)}")
            # If authenticated access fails, try public access
            try:
                github = self.get_public_github_instance()
                repo = github.get_repo(repo_name)
                file_contents = repo.get_contents(file_path)
            except Exception as public_error:
                logger.error(f"Failed to access public repo: {str(public_error)}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Repository or file not found or inaccessible: {repo_name}/{file_path}",
                )

        if isinstance(file_contents, list):
            raise HTTPException(
                status_code=400, detail="Provided path is a directory, not a file"
            )

        try:
            content_bytes = file_contents.decoded_content
            encoding = self._detect_encoding(content_bytes)
            try:
                decoded_content = content_bytes.decode(encoding)
            except UnicodeDecodeError:
                # Fallback to utf-8 with replacement for errors
                try:
                    decoded_content = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    # Fallback to latin1 as last resort
                    decoded_content = content_bytes.decode("latin1", errors="replace")
            lines = decoded_content.splitlines()

            if (start_line == end_line == 0) or (start_line == end_line == None):
                return decoded_content
            # added -2 to start and end line to include the function definition/ decorator line
            start = start_line - 2 if start_line - 2 > 0 else 0
            selected_lines = lines[start:end_line]
            return "\n".join(selected_lines)
        except Exception as e:
            logger.error(
                f"Error processing file content for {repo_name}/{file_path}: {e}",
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

    def _parse_link_header(self, link_header: str) -> Dict[str, str]:
        """Parse GitHub Link header to extract pagination URLs."""
        links = {}
        if not link_header:
            return links

        for link in link_header.split(","):
            parts = link.strip().split(";")
            if len(parts) < 2:
                continue
            url = parts[0].strip()[1:-1]  # Remove < and >
            for p in parts[1:]:
                if "rel=" in p:
                    rel = p.strip().split("=")[1].strip('"')
                    links[rel] = url
                    break
        return links

    async def get_repos_for_user(self, user_id: str):
        if self.is_development_mode:
            return {"repositories": []}

        import time  # Import the time module

        start_time = time.time()  # Start timing the entire method
        try:
            user = self.db.query(User).filter(User.uid == user_id).first()
            if user is None:
                raise HTTPException(status_code=404, detail="User not found")

            firebase_uid = user.uid
            github_username = user.provider_username

            if not github_username:
                raise HTTPException(
                    status_code=400, detail="GitHub username not found for this user"
                )

            github_oauth_token = self.get_github_oauth_token(firebase_uid)
            if not github_oauth_token:
                raise HTTPException(
                    status_code=400, detail="GitHub OAuth token not found for this user"
                )

            user_github = Github(github_oauth_token)

            user_orgs = user_github.get_user().get_orgs()
            org_logins = [org.login.lower() for org in user_orgs]

            private_key = (
                "-----BEGIN RSA PRIVATE KEY-----\n"
                + config_provider.get_github_key()
                + "\n-----END RSA PRIVATE KEY-----\n"
            )
            app_id = os.environ["GITHUB_APP_ID"]

            auth = AppAuth(app_id=app_id, private_key=private_key)
            jwt = auth.create_jwt()

            all_installations = []
            base_url = "https://api.github.com/app/installations"
            headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {jwt}",
                "X-GitHub-Api-Version": "2022-11-28",
            }

            async with aiohttp.ClientSession() as session:
                # Get first page to determine total pages
                async with session.get(
                    f"{base_url}?per_page=100", headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            f"Failed to get installations. Response: {error_text}"
                        )
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Failed to get installations: {error_text}",
                        )

                    # Extract last page number from Link header
                    last_page = 1
                    if "Link" in response.headers:
                        links = self._parse_link_header(response.headers["Link"])
                        if "last" in links:
                            last_url = links["last"]
                            match = re.search(r"[?&]page=(\d+)", last_url)
                            if match:
                                last_page = int(match.group(1))

                    first_page_data = await response.json()
                    all_installations.extend(first_page_data)

                # Generate remaining page URLs (skip page 1)
                page_urls = [
                    f"{base_url}?page={page}&per_page=100"
                    for page in range(2, last_page + 1)
                ]

                # Process URLs in batches of 10
                async def fetch_page(url):
                    try:
                        async with session.get(url, headers=headers) as response:
                            if response.status == 200:
                                installations = await response.json()
                                return installations
                            else:
                                error_text = await response.text()
                                logger.error(
                                    f"Failed to fetch page {url}. Response: {error_text}"
                                )
                                return []
                    except Exception as e:
                        logger.error(f"Error fetching page {url}: {str(e)}")
                        return []

                # Process URLs in batches of 10
                for i in range(0, len(page_urls), 10):
                    batch = page_urls[i : i + 10]
                    batch_tasks = [fetch_page(url) for url in batch]
                    batch_results = await asyncio.gather(*batch_tasks)
                    for installations in batch_results:
                        all_installations.extend(installations)

                # Filter installations
                user_installations = []
                for installation in all_installations:
                    account = installation["account"]
                    account_login = account["login"].lower()
                    account_type = account["type"]

                    if (
                        account_type == "User"
                        and account_login == github_username.lower()
                    ):
                        user_installations.append(installation)
                    elif account_type == "Organization" and account_login in org_logins:
                        user_installations.append(installation)

                # Fetch repositories for each installation
                repos = []
                for installation in user_installations:
                    app_auth = auth.get_installation_auth(installation["id"])
                    repos_url = installation["repositories_url"]
                    github = Github(auth=app_auth)  # do not remove this line
                    auth_headers = {"Authorization": f"Bearer {app_auth.token}"}

                    async with session.get(
                        f"{repos_url}?per_page=100", headers=auth_headers
                    ) as response:
                        if response.status != 200:
                            logger.error(
                                f"Failed to fetch repositories for installation ID {installation['id']}. Response: {await response.text()}"
                            )
                            continue

                        first_page_data = await response.json()
                        repos.extend(first_page_data.get("repositories", []))

                        # Get last page from Link header
                        last_page = 1
                        if "Link" in response.headers:
                            links = self._parse_link_header(response.headers["Link"])
                            if "last" in links:
                                last_url = links["last"]
                                match = re.search(r"[?&]page=(\d+)", last_url)
                                if match:
                                    last_page = int(match.group(1))

                        if last_page > 1:
                            # Generate remaining page URLs (skip page 1)
                            page_urls = [
                                f"{repos_url}?page={page}&per_page=100"
                                for page in range(2, last_page + 1)
                            ]

                            # Process URLs in batches of 10
                            for i in range(0, len(page_urls), 10):
                                batch = page_urls[i : i + 10]
                                tasks = [
                                    session.get(url, headers=auth_headers)
                                    for url in batch
                                ]
                                responses = await asyncio.gather(*tasks)

                                for response in responses:
                                    async with response:
                                        if response.status == 200:
                                            page_data = await response.json()
                                            repos.extend(
                                                page_data.get("repositories", [])
                                            )
                                        else:
                                            logger.error(
                                                f"Failed to fetch repositories page. Response: {await response.text()}"
                                            )

                # Remove duplicate repositories
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
        finally:
            total_duration = time.time() - start_time  # Calculate total duration
            logger.info(
                f"get_repos_for_user executed in {total_duration:.2f} seconds"
            )  # Log total duration

    async def get_combined_user_repos(self, user_id: str):
        subquery = (
            self.db.query(Project.repo_name, func.min(Project.id).label("min_id"))
            .filter(Project.user_id == user_id)
            .group_by(Project.repo_name)
            .subquery()
        )
        projects = (
            self.db.query(Project)
            .join(
                subquery,
                (Project.repo_name == subquery.c.repo_name)
                & (Project.id == subquery.c.min_id),
            )
            .all()
        )
        project_list = (
            [
                {
                    "id": project.id,
                    "name": project.repo_name.split("/")[-1],
                    "full_name": (
                        project.repo_name
                        if not self.is_development_mode
                        else project.repo_path
                    ),
                    "private": False,
                    "url": f"https://github.com/{project.repo_name}",
                    "owner": project.repo_name.split("/")[0],
                }
                for project in projects
            ]
            if projects is not None
            else []
        )
        user_repo_response = await self.get_repos_for_user(user_id)
        user_repos = user_repo_response["repositories"]
        db_project_full_names = {project["full_name"] for project in project_list}

        filtered_user_repos = [
            {**user_repo, "private": True}
            for user_repo in user_repos
            if user_repo["full_name"]
            not in db_project_full_names  # Only include unique user repos
        ]
        combined_repos = list(reversed(project_list + filtered_user_repos))
        return {"repositories": combined_repos}

    async def get_branch_list(self, repo_name: str):
        try:
            # Check if repo_name is a path to a local repository
            if os.path.exists(repo_name) and os.path.isdir(repo_name):
                try:
                    # Handle local repository
                    local_repo = git.Repo(repo_name)

                    # Get the default branch
                    try:
                        default_branch = local_repo.git.symbolic_ref(
                            "refs/remotes/origin/HEAD"
                        ).split("/")[-1]
                    except git.GitCommandError:
                        # If no remote HEAD is found, use the current branch
                        default_branch = local_repo.active_branch.name

                    # Get all local branches
                    branches = [
                        branch.name
                        for branch in local_repo.heads
                        if branch.name != default_branch
                    ]

                    return {"branches": [default_branch] + branches}
                except git.InvalidGitRepositoryError:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Not a valid git repository: {repo_name}",
                    )
                except Exception as e:
                    logger.error(
                        f"Error fetching branches for local repo {repo_name}: {str(e)}",
                        exc_info=True,
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error fetching branches for local repo: {str(e)}",
                    )
            else:
                # Handle GitHub repository (existing functionality)
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
        try:
            # Try authenticated access first
            github, _, _ = self.get_github_repo_details(repo_name)
            repo = github.get_repo(repo_name)

            return github, repo
        except Exception as private_error:
            logger.info(
                f"Failed to access private repo {repo_name}: {str(private_error)}"
            )
            # If authenticated access fails, try public access
            try:
                github = self.get_public_github_instance()
                repo = github.get_repo(repo_name)
                return github, repo
            except Exception as public_error:
                logger.error(
                    f"Failed to access public repo {repo_name}: {str(public_error)}"
                )
                raise HTTPException(
                    status_code=404,
                    detail=f"Repository {repo_name} not found or inaccessible on GitHub",
                )

    async def get_project_structure_async(
        self, project_id: str, path: Optional[str] = None
    ) -> str:
        logger.info(
            f"Fetching project structure for project ID: {project_id}, path: {path}"
        )

        # Modify cache key to reflect that we're only caching the specific path
        cache_key = (
            f"project_structure:{project_id}:exact_path_{path}:depth_{self.max_depth}"
        )
        cached_structure = self.redis.get(cache_key)

        if cached_structure:
            logger.info(
                f"Project structure found in cache for project ID: {project_id}, path: {path}"
            )
            return cached_structure.decode("utf-8")

        project = await self.project_manager.get_project_from_db_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        repo_name = project["project_name"]
        if not repo_name:
            raise HTTPException(
                status_code=400, detail="Project has no associated GitHub repository"
            )

        try:
            github, repo = self.get_repo(repo_name)

            # If path is provided, verify it exists
            if path:
                try:
                    # Check if the path exists in the repository
                    repo.get_contents(path)
                except Exception:
                    raise HTTPException(
                        status_code=404, detail=f"Path {path} not found in repository"
                    )

            # Start structure fetch from the specified path with depth 0
            structure = await self._fetch_repo_structure_async(
                repo, path or "", current_depth=0, base_path=path
            )
            formatted_structure = self._format_tree_structure(structure)

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
                status_code=500, detail=f"Failed to fetch project structure: {str(e)}"
            )

    async def _fetch_repo_structure_async(
        self,
        repo: Any,
        path: str = "",
        current_depth: int = 0,
        base_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        exclude_extensions = [
            "png",
            "jpg",
            "jpeg",
            "gif",
            "bmp",
            "tiff",
            "webp",
            "ico",
            "svg",
            "mp4",
            "avi",
            "mov",
            "wmv",
            "flv",
            "ipynb",
            "zlib",
        ]

        # Calculate current depth relative to base_path
        if base_path:
            # If we have a base_path, calculate depth relative to it
            relative_path = path[len(base_path) :].strip("/")
            current_depth = len(relative_path.split("/")) if relative_path else 0
        else:
            # If no base_path, calculate depth from root
            current_depth = len(path.split("/")) if path else 0

        # If we've reached max depth, return truncated indicator
        if current_depth >= self.max_depth:
            return {
                "type": "directory",
                "name": path.split("/")[-1] or repo.name,
                "children": [{"type": "file", "name": "...", "path": "truncated"}],
            }

        structure = {
            "type": "directory",
            "name": path.split("/")[-1] or repo.name,
            "children": [],
        }

        try:
            contents = await asyncio.get_event_loop().run_in_executor(
                self.executor, repo.get_contents, path
            )

            if not isinstance(contents, list):
                contents = [contents]

            # Filter out files with excluded extensions
            contents = [
                item
                for item in contents
                if item.type == "dir"
                or not any(item.name.endswith(ext) for ext in exclude_extensions)
            ]

            tasks = []
            for item in contents:
                # Only process items within the base_path if it's specified
                if base_path and not item.path.startswith(base_path):
                    continue

                if item.type == "dir":
                    task = self._fetch_repo_structure_async(
                        repo,
                        item.path,
                        current_depth=current_depth,
                        base_path=base_path,
                    )
                    tasks.append(task)
                else:
                    structure["children"].append(
                        {
                            "type": "file",
                            "name": item.name,
                            "path": item.path,
                        }
                    )

            if tasks:
                children = await asyncio.gather(*tasks)
                structure["children"].extend(children)

        except Exception as e:
            logger.error(f"Error fetching contents for path {path}: {str(e)}")

        return structure

    def _format_tree_structure(
        self, structure: Dict[str, Any], root_path: str = ""
    ) -> str:
        """
        Creates a clear hierarchical structure using simple nested dictionaries.

        Args:
            self: The instance object
            structure: Dictionary containing name and children
            root_path: Optional root path string (unused but kept for signature compatibility)
        """

        def _format_node(node: Dict[str, Any], depth: int = 0) -> List[str]:
            output = []
            indent = "  " * depth
            if depth > 0:  # Skip root name
                output.append(f"{indent}{node['name']}")

            if "children" in node:
                children = sorted(node.get("children", []), key=lambda x: x["name"])
                for child in children:
                    output.extend(_format_node(child, depth + 1))

            return output

        return "\n".join(_format_node(structure))

    async def check_public_repo(self, repo_name: str) -> bool:
        try:
            github = self.get_public_github_instance()
            github.get_repo(repo_name)
            return True
        except Exception:
            return False

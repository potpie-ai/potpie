import asyncio
import json
import logging
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import ClientTimeout, ClientConnectorError
import chardet
import git
import requests
import ssl
import socket
import certifi
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
from app.modules.code_provider.github.github_provider import GitHubProvider
from app.modules.code_provider.provider_factory import CodeProviderFactory
from app.modules.code_provider.base.code_provider_interface import AuthMethod

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
        """
        Get GitHub client using provider abstraction.
        Maintains backward compatibility with existing code.
        """
        try:
            provider = CodeProviderFactory.create_provider_with_fallback(repo_name)
            return provider.client
        except Exception as e:
            logger.error(f"Failed to get GitHub client for {repo_name}: {str(e)}")
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
        commit_id: str,
    ) -> str:
        logger.info(f"Attempting to access file: {file_path} in repo: {repo_name}")

        try:
            # Try authenticated access first
            github, repo = self.get_repo(repo_name)
            file_contents = repo.get_contents(
                file_path, ref=commit_id if commit_id else branch_name
            )
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
            # start = start_line - 2 if start_line - 2 > 0 else 0
            selected_lines = lines[max(0, start_line - 1) : min(len(lines), end_line)]
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

    def get_github_oauth_token(self, uid: str) -> Optional[str]:
        """
        Get user's GitHub OAuth token from provider_info.

        Returns:
            OAuth token if available, None otherwise

        Raises:
            HTTPException: If user not found
        """
        user = self.db.query(User).filter(User.uid == uid).first()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        # Safely access provider_info and access_token
        if user.provider_info is None:
            logger.warning(f"User {uid} has no provider_info")
            return None

        if not isinstance(user.provider_info, dict):
            logger.warning(
                f"User {uid} provider_info is not a dict: {type(user.provider_info)}"
            )
            return None

        access_token = user.provider_info.get("access_token")
        if not access_token:
            logger.warning(f"User {uid} has no access_token in provider_info")
            return None

        return access_token

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

            # Try to get user's OAuth token first
            github_oauth_token = self.get_github_oauth_token(firebase_uid)

            # Fall back to system tokens if user OAuth token not available
            if not github_oauth_token:
                logger.info(
                    f"No user OAuth token for {firebase_uid}, falling back to system tokens"
                )
                # Try GH_TOKEN_LIST first
                token_list_str = os.getenv("GH_TOKEN_LIST", "")
                if token_list_str:
                    tokens = [t.strip() for t in token_list_str.split(",") if t.strip()]
                    if tokens:
                        github_oauth_token = random.choice(tokens)
                        logger.info("Using token from GH_TOKEN_LIST as fallback")

                # Fall back to CODE_PROVIDER_TOKEN if GH_TOKEN_LIST not available
                if not github_oauth_token:
                    github_oauth_token = os.getenv("CODE_PROVIDER_TOKEN")
                    if github_oauth_token:
                        logger.info("Using CODE_PROVIDER_TOKEN as fallback")

                # If still no token, raise error
                if not github_oauth_token:
                    raise HTTPException(
                        status_code=400,
                        detail="No GitHub authentication available (user OAuth token, GH_TOKEN_LIST, or CODE_PROVIDER_TOKEN)",
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

            ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(
                ssl=ssl_context,
                ttl_dns_cache=300,
                family=socket.AF_INET,
            )
            timeout = ClientTimeout(total=20)

            async with aiohttp.ClientSession(
                connector=connector, timeout=timeout
            ) as session:
                # Get first page to determine total pages
                first_url = f"{base_url}?per_page=100"
                attempt = 0
                max_attempts = 3
                backoff = 1
                while True:
                    try:
                        async with session.get(first_url, headers=headers) as response:
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
                                links = self._parse_link_header(
                                    response.headers["Link"]
                                )
                                if "last" in links:
                                    last_url = links["last"]
                                    match = re.search(r"[?&]page=(\d+)", last_url)
                                    if match:
                                        last_page = int(match.group(1))

                            first_page_data = await response.json()
                            all_installations.extend(first_page_data)
                            break
                    except (ClientConnectorError, asyncio.TimeoutError) as net_err:
                        attempt += 1
                        if attempt >= max_attempts:
                            logger.error(
                                f"Network error contacting GitHub installations API: {net_err}"
                            )
                            raise HTTPException(
                                status_code=503,
                                detail="Unable to reach GitHub API (installations). Please check network/proxy settings and try again.",
                            )
                        await asyncio.sleep(backoff)
                        backoff *= 2

                # Generate remaining page URLs (skip page 1)
                page_urls = [
                    f"{base_url}?page={page}&per_page=100"
                    for page in range(2, last_page + 1)
                ]

                # Process URLs in batches of 10
                async def fetch_page(url):
                    attempt = 0
                    max_attempts = 3
                    backoff = 1
                    while True:
                        try:
                            async with session.get(url, headers=headers) as response:
                                if response.status == 200:
                                    return await response.json()
                                error_text = await response.text()
                                logger.error(
                                    f"Failed to fetch page {url}. Response: {error_text}"
                                )
                                return []
                        except (ClientConnectorError, asyncio.TimeoutError) as net_err:
                            attempt += 1
                            if attempt >= max_attempts:
                                logger.error(f"Network error fetching {url}: {net_err}")
                                return []
                            await asyncio.sleep(backoff)
                            backoff *= 2

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
        """
        Get public GitHub instance using PAT from token pool.
        Uses new provider factory with PAT-first strategy.
        """
        # Initialize legacy token list if needed
        if not cls.gh_token_list:
            cls.initialize_tokens()

        # Use factory to create provider with PAT

        token = random.choice(cls.gh_token_list)
        provider = GitHubProvider()
        provider.authenticate({"token": token}, AuthMethod.PERSONAL_ACCESS_TOKEN)
        return provider.client

    def get_repo(self, repo_name: str) -> Tuple[Github, Any]:
        """
        Get repository using provider abstraction.
        Returns (Github client, Repository) for backward compatibility.

        Strategy:
        1. Try create_provider_with_fallback (which handles App-first or PAT-first based on config)
        2. If that fails with 404 and we haven't tried PAT yet, try PAT explicitly
        """
        try:
            # Try to create provider with authentication fallback
            provider = CodeProviderFactory.create_provider_with_fallback(repo_name)

            # For backward compatibility, return the PyGithub client and repo
            github_client = provider.client

            # Use provider's _get_repo if available (e.g., GitBucketProvider
            # normalizes repo names), otherwise call PyGithub directly
            if hasattr(provider, "_get_repo"):
                repo = provider._get_repo(repo_name)
            else:
                repo = github_client.get_repo(repo_name)

            return github_client, repo
        except HTTPException as he:
            # Re-raise HTTPException as-is
            raise he
        except Exception as e:
            error_str = str(e)
            is_not_found = "404" in error_str or "Not Found" in error_str

            # If it's a 404 and we might have tried App auth first, try PAT as final fallback
            # Only applicable for GitHub provider (not GitBucket or other providers)
            provider_type = os.getenv("CODE_PROVIDER", "github").lower()
            if is_not_found and provider_type == "github":
                app_id = os.getenv("GITHUB_APP_ID")
                private_key = config_provider.get_github_key()

                # Only retry with PAT if App was configured (meaning it was tried first)
                if app_id and private_key:
                    logger.info(
                        f"GitHub App auth failed with 404 for {repo_name}, "
                        f"attempting final fallback to PAT pool"
                    )
                    try:
                        # Force PAT authentication by using the public instance method
                        github_client = self.get_public_github_instance()
                        repo = github_client.get_repo(repo_name)
                        logger.info(
                            f"Successfully accessed {repo_name} using PAT after App auth failed"
                        )
                        return github_client, repo
                    except Exception as pat_error:
                        logger.warning(
                            f"PAT fallback also failed for {repo_name}: {str(pat_error)}"
                        )

            # If all methods failed, raise the original error
            logger.error(f"Failed to access repository {repo_name}: {error_str}")
            raise HTTPException(
                status_code=404,
                detail=f"Repository {repo_name} not found or inaccessible",
            )

    # Maximum depth for full structure cache (used for warm_file_structure_cache)
    MAX_STRUCTURE_DEPTH = 6

    def _calculate_path_depth(self, path: Optional[str]) -> int:
        """Calculate the depth of a path (number of directory levels)."""
        if not path:
            return 0
        return len(path.strip("/").split("/"))

    def _needs_deep_fetch(self, path: Optional[str], requested_depth: int) -> bool:
        """
        Check if the request needs more depth than the cached root structure can provide.

        The root structure is cached at MAX_STRUCTURE_DEPTH from root.
        If path is at depth P and user wants D more levels, we need P + D total depth.
        If P + D > MAX_STRUCTURE_DEPTH, cached structure won't have enough data.
        """
        path_depth = self._calculate_path_depth(path)
        total_depth_needed = path_depth + requested_depth
        return total_depth_needed > self.MAX_STRUCTURE_DEPTH

    async def get_project_structure_async(
        self, project_id: str, path: Optional[str] = None, max_depth: Optional[int] = None
    ) -> str:
        """
        Get project structure with two-tier caching strategy.

        The full structure (at MAX_STRUCTURE_DEPTH) is cached once per project/branch.
        Subsequent requests with different path/depth parameters are derived from
        the cached full structure, avoiding redundant GitHub API calls.

        For deep path requests that exceed cached depth, fetches fresh data for
        that specific path and caches it separately.
        """
        logger.info(
            f"Fetching project structure for project ID: {project_id}, path: {path}, max_depth: {max_depth}"
        )

        effective_depth = max_depth if max_depth is not None else self.max_depth

        # Get project info for branch-aware cache key
        project = await self.project_manager.get_project_from_db_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        repo_name = project["project_name"]
        if not repo_name:
            raise HTTPException(
                status_code=400, detail="Project has no associated GitHub repository"
            )

        branch_name = project.get("branch_name") or project.get("commit_id") or "main"

        # Check if this request needs deeper data than root cache can provide
        needs_deep_fetch = self._needs_deep_fetch(path, effective_depth)

        if needs_deep_fetch and path:
            # For deep path requests, use path-specific cache
            return await self._get_deep_path_structure(
                project_id, repo_name, branch_name, path, effective_depth
            )

        # Use root structure cache for normal requests
        full_cache_key = f"project_structure:{project_id}:branch_{branch_name}"

        # Try to get full cached structure first
        cached_full_structure = self.redis.get(full_cache_key)

        if cached_full_structure:
            logger.info(
                f"Full structure cache hit for project {project_id}, branch {branch_name}. Filtering for path={path}, depth={effective_depth}"
            )
            try:
                full_structure = json.loads(cached_full_structure.decode("utf-8"))
                return self._filter_structure(full_structure, path, effective_depth)
            except json.JSONDecodeError as e:
                logger.warning(f"Cache JSON decode error, will re-fetch: {e}")
                # Fall through to fetch fresh data

        # Cache miss - fetch full structure at max depth
        logger.info(
            f"Cache miss for project {project_id}. Fetching full structure at depth {self.MAX_STRUCTURE_DEPTH}"
        )

        try:
            github, repo = self.get_repo(repo_name)

            # Determine concurrency based on provider type
            # GitHub can handle higher concurrency than self-hosted GitBucket
            provider_type = os.getenv("CODE_PROVIDER", "github").lower()
            max_concurrent = 50 if provider_type == "github" else 5

            # Use parallel fetching for better performance on large repos
            logger.info(f"Using parallel fetch for project {project_id} at depth {self.MAX_STRUCTURE_DEPTH}, concurrency={max_concurrent}")
            full_structure = await self._fetch_repo_structure_parallel(
                repo,
                base_path="",
                max_depth=self.MAX_STRUCTURE_DEPTH,
                ref=branch_name,
                max_concurrent=max_concurrent,
            )

            # Cache the full structure as JSON (no TTL - invalidate on re-parse only)
            try:
                self.redis.set(full_cache_key, json.dumps(full_structure))
                logger.info(f"Cached full structure for project {project_id}, branch {branch_name}")
            except Exception as cache_error:
                logger.warning(f"Failed to cache structure: {cache_error}")

            # Return filtered result based on requested path and depth
            return self._filter_structure(full_structure, path, effective_depth)

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

    async def _get_deep_path_structure(
        self,
        project_id: str,
        repo_name: str,
        branch_name: str,
        path: str,
        max_depth: int,
    ) -> str:
        """
        Fetch and cache structure for a deep path that exceeds root cache depth.

        Uses a path-specific cache key to store deep structure for specific paths.
        """
        # Normalize path for cache key
        normalized_path = path.strip("/").replace("/", "_")
        deep_cache_key = f"project_structure:{project_id}:branch_{branch_name}:deep_path_{normalized_path}:depth_{max_depth}"

        # Check path-specific cache first
        cached_deep_structure = self.redis.get(deep_cache_key)
        if cached_deep_structure:
            logger.info(
                f"Deep path cache hit for project {project_id}, path={path}, depth={max_depth}"
            )
            try:
                return cached_deep_structure.decode("utf-8")
            except Exception as e:
                logger.warning(f"Deep cache decode error, will re-fetch: {e}")

        # Fetch fresh data for this specific path
        logger.info(
            f"Fetching deep structure for project {project_id}, path={path}, depth={max_depth}"
        )

        try:
            github, repo = self.get_repo(repo_name)

            # Verify path exists
            try:
                repo.get_contents(path, ref=branch_name)
            except Exception:
                return f"Path '{path}' not found in repository"

            # Determine concurrency based on provider type
            provider_type = os.getenv("CODE_PROVIDER", "github").lower()
            max_concurrent = 50 if provider_type == "github" else 20

            # Fetch structure starting from the specified path using parallel method
            structure = await self._fetch_repo_structure_parallel(
                repo,
                base_path=path,
                max_depth=max_depth,
                ref=branch_name,
                max_concurrent=max_concurrent,
            )

            formatted_structure = self._format_tree_structure(structure)

            # Cache the deep path structure (no TTL - invalidate on re-parse)
            try:
                self.redis.set(deep_cache_key, formatted_structure)
                logger.info(
                    f"Cached deep path structure for project {project_id}, path={path}, depth={max_depth}"
                )
            except Exception as cache_error:
                logger.warning(f"Failed to cache deep structure: {cache_error}")

            return formatted_structure

        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(
                f"Error fetching deep structure for {repo_name}/{path}: {str(e)}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to fetch deep structure: {str(e)}"
            )

    async def _fetch_repo_structure_async(
        self,
        repo: Any,
        path: str = "",
        current_depth: int = 0,
        base_path: Optional[str] = None,
        max_depth: int = 4,
        ref: Optional[str] = None,
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
        if current_depth >= max_depth:
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
                self.executor, lambda: repo.get_contents(path, ref=ref)
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
                        max_depth=max_depth,
                        ref=ref,
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

    async def _fetch_repo_structure_parallel(
        self,
        repo: Any,
        base_path: str = "",
        max_depth: int = 6,
        ref: Optional[str] = None,
        max_concurrent: int = 20,
    ) -> Dict[str, Any]:
        """
        Fetch repository structure using parallel work queue pattern.

        This is significantly faster than the recursive approach for large repos
        because workers don't wait for directory levels to complete.

        Args:
            repo: PyGithub repository object
            base_path: Starting path (empty string for root)
            max_depth: Maximum depth to traverse
            ref: Branch name or commit SHA
            max_concurrent: Maximum concurrent API requests

        Returns:
            Dictionary representing the file structure tree
        """
        exclude_extensions = {
            "png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp", "ico", "svg",
            "mp4", "avi", "mov", "wmv", "flv", "ipynb", "zlib",
        }

        # Flat list to collect all items
        all_items: List[Dict[str, Any]] = []
        items_lock = asyncio.Lock()

        # Work queue: (path, depth)
        queue: asyncio.Queue = asyncio.Queue()

        # Semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        # Track active workers
        active_workers = 0
        active_lock = asyncio.Lock()
        done_event = asyncio.Event()

        # Calculate base depth for relative depth tracking
        base_depth = len(base_path.strip("/").split("/")) if base_path else 0

        async def fetch_directory(path: str, depth: int) -> List[tuple]:
            """Fetch contents of a single directory and return subdirectories to process."""
            subdirs = []

            async with semaphore:
                try:
                    contents = await asyncio.get_event_loop().run_in_executor(
                        self.executor, lambda: repo.get_contents(path, ref=ref)
                    )

                    if not isinstance(contents, list):
                        contents = [contents]

                    items_to_add = []

                    for item in contents:
                        # Skip excluded extensions
                        if item.type != "dir":
                            ext = item.name.split(".")[-1].lower() if "." in item.name else ""
                            if ext in exclude_extensions:
                                continue

                        item_data = {
                            "type": "directory" if item.type == "dir" else "file",
                            "name": item.name,
                            "path": item.path,
                            "parent_path": path,
                        }
                        items_to_add.append(item_data)

                        # Queue subdirectories if within depth limit
                        if item.type == "dir" and depth + 1 < max_depth:
                            subdirs.append((item.path, depth + 1))

                    # Add items to global list
                    async with items_lock:
                        all_items.extend(items_to_add)

                except Exception as e:
                    logger.warning(f"Error fetching directory {path}: {e}")

            return subdirs

        async def worker():
            """Worker that processes directories from the queue."""
            nonlocal active_workers

            while True:
                try:
                    # Get work from queue with timeout
                    try:
                        path, depth = await asyncio.wait_for(queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        # Check if we should exit
                        async with active_lock:
                            if queue.empty() and active_workers == 0:
                                return
                        continue

                    async with active_lock:
                        active_workers += 1

                    try:
                        # Fetch directory and get subdirectories
                        subdirs = await fetch_directory(path, depth)

                        # Add subdirectories to queue
                        for subdir in subdirs:
                            await queue.put(subdir)
                    finally:
                        async with active_lock:
                            active_workers -= 1
                        queue.task_done()

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Worker error: {e}")

        # Start with root directory
        initial_depth = 0 if not base_path else 0
        await queue.put((base_path, initial_depth))

        # Create workers
        num_workers = min(max_concurrent, 50)  # Cap workers
        workers = [asyncio.create_task(worker()) for _ in range(num_workers)]

        # Wait for queue to be fully processed
        await queue.join()

        # Cancel workers
        for w in workers:
            w.cancel()

        # Wait for workers to finish
        await asyncio.gather(*workers, return_exceptions=True)

        # Build tree structure from flat list
        return self._build_tree_from_flat_list(all_items, base_path, repo.name, max_depth)

    def _build_tree_from_flat_list(
        self,
        items: List[Dict[str, Any]],
        base_path: str,
        repo_name: str,
        max_depth: int,
    ) -> Dict[str, Any]:
        """
        Build hierarchical tree structure from flat list of items.

        Args:
            items: Flat list of items with path and parent_path
            base_path: Base path for the tree root
            repo_name: Repository name for root node
            max_depth: Max depth for truncation indicators

        Returns:
            Hierarchical tree structure
        """
        # Create root
        root_name = base_path.split("/")[-1] if base_path else repo_name
        root = {
            "type": "directory",
            "name": root_name,
            "children": [],
        }

        # Index for quick lookup: path -> node
        path_to_node: Dict[str, Dict] = {base_path: root}

        # Sort items by path depth (parents before children)
        items_sorted = sorted(items, key=lambda x: x["path"].count("/"))

        for item in items_sorted:
            parent_path = item["parent_path"]

            # Get or create parent node
            if parent_path not in path_to_node:
                # This shouldn't happen often, but handle gracefully
                continue

            parent_node = path_to_node[parent_path]

            # Create node for this item
            node = {
                "type": item["type"],
                "name": item["name"],
            }

            if item["type"] == "file":
                node["path"] = item["path"]
            else:
                node["children"] = []
                path_to_node[item["path"]] = node

            parent_node["children"].append(node)

        # Add truncation indicators for directories at max depth
        base_depth = base_path.count("/") if base_path else -1

        def add_truncation_indicators(node: Dict, current_depth: int):
            if node.get("type") == "directory" and "children" in node:
                # Check if this directory has any subdirectories that weren't fetched
                if current_depth >= max_depth - 1:
                    # Check if any children are directories - they might be truncated
                    for child in node.get("children", []):
                        if child.get("type") == "directory" and not child.get("children"):
                            child["children"] = [{"type": "file", "name": "...", "path": "truncated"}]
                else:
                    for child in node.get("children", []):
                        add_truncation_indicators(child, current_depth + 1)

        add_truncation_indicators(root, 0)

        return root

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

    def _find_path_in_structure(
        self, structure: Dict[str, Any], target_path: str
    ) -> Optional[Dict[str, Any]]:
        """
        Navigate to a specific path in the structure tree.

        Args:
            structure: The full structure dictionary
            target_path: Path to find (e.g., "src/components")

        Returns:
            The subtree at the target path, or None if not found
        """
        if not target_path:
            return structure

        path_parts = target_path.strip("/").split("/")
        current = structure

        for part in path_parts:
            if "children" not in current:
                return None
            found = False
            for child in current["children"]:
                if child.get("name") == part and child.get("type") == "directory":
                    current = child
                    found = True
                    break
            if not found:
                return None

        return current

    def _truncate_depth(
        self, structure: Dict[str, Any], max_depth: int, current_depth: int = 0
    ) -> Dict[str, Any]:
        """
        Truncate the structure tree to a maximum depth.

        Args:
            structure: The structure dictionary to truncate
            max_depth: Maximum depth to include
            current_depth: Current depth in recursion

        Returns:
            Truncated structure dictionary
        """
        result = {
            "type": structure.get("type", "directory"),
            "name": structure.get("name", ""),
        }

        if "path" in structure:
            result["path"] = structure["path"]

        if "children" not in structure:
            return result

        if current_depth >= max_depth:
            # At max depth, show truncation indicator for directories with children
            if structure.get("children"):
                result["children"] = [{"type": "file", "name": "...", "path": "truncated"}]
            else:
                result["children"] = []
            return result

        result["children"] = []
        for child in structure.get("children", []):
            if child.get("type") == "directory":
                truncated_child = self._truncate_depth(child, max_depth, current_depth + 1)
                result["children"].append(truncated_child)
            else:
                result["children"].append(child.copy())

        return result

    def _filter_structure(
        self, full_structure: Dict[str, Any], path: Optional[str], max_depth: int
    ) -> str:
        """
        Filter the full structure to return only the requested path at specified depth.

        Args:
            full_structure: The complete cached structure
            path: Optional path to navigate to
            max_depth: Maximum depth to return

        Returns:
            Formatted string of the filtered structure
        """
        # Navigate to the requested path if provided
        if path:
            structure = self._find_path_in_structure(full_structure, path)
            if not structure:
                return f"Path '{path}' not found in repository structure"
        else:
            structure = full_structure

        # Truncate to requested depth
        truncated = self._truncate_depth(structure, max_depth)

        return self._format_tree_structure(truncated)

    async def warm_file_structure_cache(self, project_id: str) -> None:
        """
        Pre-fetch and cache the full file structure at max depth.

        This method should be called during parsing to populate the cache.
        The cached structure is then reused for all subsequent requests
        (agent chat, tools, etc.) with different path/depth parameters.

        Args:
            project_id: The project ID to cache structure for
        """
        logger.info(f"Warming file structure cache for project {project_id}")
        try:
            # This will fetch at MAX_STRUCTURE_DEPTH and cache the result
            await self.get_project_structure_async(
                project_id,
                path=None,
                max_depth=self.MAX_STRUCTURE_DEPTH,
            )
            logger.info(f"Successfully warmed file structure cache for project {project_id}")
        except Exception as e:
            logger.error(f"Failed to warm file structure cache for {project_id}: {e}")
            # Don't re-raise - cache warming failure shouldn't block parsing

    async def invalidate_file_structure_cache(
        self, project_id: str, branch_name: Optional[str] = None
    ) -> None:
        """
        Invalidate the cached file structure for a project/branch.

        This method clears both the root structure cache and any deep path caches.
        Should be called before re-parsing a branch to ensure fresh data is fetched.

        Args:
            project_id: The project ID
            branch_name: Optional branch name. If not provided, will look up from project.
        """
        try:
            if not branch_name:
                project = await self.project_manager.get_project_from_db_by_id(project_id)
                if project:
                    branch_name = project.get("branch_name") or project.get("commit_id") or "main"
                else:
                    logger.warning(f"Project {project_id} not found for cache invalidation")
                    return

            # Delete root structure cache
            root_cache_key = f"project_structure:{project_id}:branch_{branch_name}"
            root_deleted = self.redis.delete(root_cache_key)

            # Delete all deep path caches for this project/branch using SCAN
            deep_path_pattern = f"project_structure:{project_id}:branch_{branch_name}:deep_path_*"
            deep_deleted_count = 0

            # Use SCAN to find and delete matching keys (safer than KEYS for large datasets)
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(cursor=cursor, match=deep_path_pattern, count=100)
                if keys:
                    self.redis.delete(*keys)
                    deep_deleted_count += len(keys)
                if cursor == 0:
                    break

            total_deleted = (1 if root_deleted else 0) + deep_deleted_count
            if total_deleted > 0:
                logger.info(
                    f"Invalidated file structure cache for project {project_id}, branch {branch_name}: "
                    f"root={'yes' if root_deleted else 'no'}, deep_paths={deep_deleted_count}"
                )
            else:
                logger.debug(f"No cache entries found to invalidate for project {project_id}, branch {branch_name}")
        except Exception as e:
            logger.error(f"Failed to invalidate file structure cache for {project_id}: {e}")

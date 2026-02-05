import asyncio
import os
import secrets
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
from app.modules.utils.logger import setup_logger
from sqlalchemy.orm import Session
from redis import Redis

from app.core.config_provider import config_provider
from app.modules.projects.projects_model import Project
from app.modules.projects.projects_service import ProjectService
from app.modules.users.user_model import User
from app.modules.code_provider.github.github_provider import GitHubProvider
from app.modules.code_provider.provider_factory import CodeProviderFactory
from app.modules.code_provider.base.code_provider_interface import AuthMethod

logger = setup_logger(__name__)


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

            if (start_line == end_line == 0) or (start_line == end_line is None):
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
        Get user's GitHub OAuth token from UserAuthProvider (new system) or provider_info (legacy).

        Returns:
            OAuth token if available, None otherwise

        Raises:
            HTTPException: If user not found
        """
        user = self.db.query(User).filter(User.uid == uid).first()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        # Try new UserAuthProvider system first
        try:
            from app.modules.auth.auth_provider_model import UserAuthProvider
            from app.modules.integrations.token_encryption import decrypt_token

            github_provider = (
                self.db.query(UserAuthProvider)
                .filter(
                    UserAuthProvider.user_id == uid,
                    UserAuthProvider.provider_type == "firebase_github",
                )
                .first()
            )
            if github_provider and github_provider.access_token:
                logger.info("Found GitHub token in UserAuthProvider for user %s", uid)
                # Decrypt the token before returning
                try:
                    decrypted_token = decrypt_token(github_provider.access_token)
                    return decrypted_token
                except Exception as e:
                    logger.warning(
                        "Failed to decrypt GitHub token for user %s: %s. "
                        "Assuming plaintext token (backward compatibility).",
                        uid,
                        str(e),
                    )
                    # Token might be plaintext (from before encryption was added)
                    return github_provider.access_token
        except Exception as e:
            logger.debug("Error checking UserAuthProvider: %s", str(e))

        # Fallback to legacy provider_info system
        if user.provider_info is None:
            logger.warning("User %s has no provider_info", uid)
            return None

        if not isinstance(user.provider_info, dict):
            logger.warning(
                "User %s provider_info is not a dict: %s",
                uid,
                type(user.provider_info),
            )
            return None

        access_token = user.provider_info.get("access_token")
        if not access_token:
            logger.warning("User %s has no access_token in provider_info", uid)
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

            # Check if user has GitHub provider via unified auth system
            from app.modules.auth.auth_provider_model import UserAuthProvider

            github_provider = (
                self.db.query(UserAuthProvider)
                .filter(
                    UserAuthProvider.user_id == user_id,
                    UserAuthProvider.provider_type == "firebase_github",
                )
                .first()
            )

            # If no GitHub provider linked, check if user needs to link GitHub
            if not github_provider:
                # Check legacy provider_username as fallback (for old accounts)
                if not user.provider_username:
                    raise HTTPException(
                        status_code=400,
                        detail="GitHub account not linked. Please link your GitHub account to access repositories.",
                    )
                # If legacy username exists, continue (backward compatibility)
                github_username = user.provider_username
            else:
                # Get GitHub username from provider data (new unified auth system)
                github_username = None
                if github_provider.provider_data:
                    provider_data = github_provider.provider_data
                    if isinstance(provider_data, dict):
                        github_username = provider_data.get(
                            "username"
                        ) or provider_data.get("login")

                # Fallback to legacy provider_username field
                if not github_username:
                    github_username = user.provider_username

            # Try to get user's OAuth token first
            github_oauth_token = self.get_github_oauth_token(firebase_uid)

            # If we have a token but no username, get it from GitHub API
            if not github_username and github_oauth_token:
                try:
                    user_github = Github(github_oauth_token)
                    github_user = user_github.get_user()
                    github_username = github_user.login
                    logger.info(
                        f"Retrieved GitHub username {github_username} from API for user {user_id}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to get GitHub username from API: {str(e)}")

            # If still no username, we can't proceed
            if not github_username:
                raise HTTPException(
                    status_code=400,
                    detail="GitHub username not found. Please ensure your GitHub account is properly linked.",
                )

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
                        github_oauth_token = secrets.choice(tokens)
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
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
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
                    Github(auth=app_auth)  # do not remove this line
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
            logger.exception("Failed to fetch repositories", user_id=user_id)
            raise HTTPException(
                status_code=500, detail="Failed to fetch repositories"
            ) from e
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

        token = secrets.choice(cls.gh_token_list)
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
            repo = github_client.get_repo(repo_name)

            return github_client, repo
        except HTTPException as he:
            # Re-raise HTTPException as-is
            raise he
        except Exception as e:
            error_str = str(e)
            is_not_found = "404" in error_str or "Not Found" in error_str

            # If it's a 404 and we might have tried App auth first, try PAT as final fallback
            if is_not_found:
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
                repo,
                path or "",
                current_depth=0,
                base_path=path or "",
                ref=(
                    project.get("branch_name")
                    if project.get("branch_name")
                    else project.get("commit_id")
                ),
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

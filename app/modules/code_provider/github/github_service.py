import asyncio
import os
import secrets
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

import aiohttp
from aiohttp import ClientTimeout, ClientConnectorError
import chardet
import requests
import ssl
import socket
import certifi
from fastapi import HTTPException
from github import Github
from github.Auth import AppAuth
from github.GithubException import GithubException

# Lazy import for GitPython - top-level import causes SIGSEGV in forked workers
if TYPE_CHECKING:
    import git as git_module


def _get_git_module():
    """Lazy import git module to avoid fork-safety issues."""
    import git
    return git
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.modules.utils.logger import setup_logger
from sqlalchemy.orm import Session
from redis import Redis
from redis.exceptions import RedisError

from app.core.config_provider import config_provider
from app.modules.code_provider.base.code_provider_interface import AuthMethod
from app.modules.code_provider.github.github_provider import GitHubProvider
from app.modules.code_provider.provider_factory import CodeProviderFactory
from app.modules.projects.projects_model import Project
from app.modules.projects.projects_service import ProjectService
from app.modules.users.user_model import User

try:
    import redis.asyncio as redis_async
except ImportError:
    redis_async = None  # type: ignore[assignment]

logger = setup_logger(__name__)

# Lazy async Redis client for project structure cache (shared across instances)
_async_redis_cache: Optional[Any] = None
_async_redis_cache_lock = asyncio.Lock()


async def _get_async_redis_cache():  # noqa: ANN201
    """Return shared async Redis client for cache; create on first use (guarded by lock)."""
    global _async_redis_cache
    if _async_redis_cache is not None:
        return _async_redis_cache
    if redis_async is None or not config_provider.get_redis_url():
        return None
    async with _async_redis_cache_lock:
        if _async_redis_cache is not None:
            return _async_redis_cache
        try:
            _async_redis_cache = redis_async.from_url(
                config_provider.get_redis_url(), decode_responses=False
            )
            return _async_redis_cache
        except Exception as e:
            logger.warning("Async Redis cache unavailable: %s", e)
            return None


async def close_github_async_redis_cache() -> None:
    """Close the global async Redis cache. Call from app shutdown to avoid connection leaks."""
    global _async_redis_cache
    if _async_redis_cache is not None:
        try:
            await _async_redis_cache.aclose()
        except Exception as e:
            logger.warning("Failed to close GitHub async Redis cache: %s", e)
        _async_redis_cache = None


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
        raw_key = config_provider.get_github_key()
        if not raw_key.startswith("-----BEGIN"):
            raw_key = (
                "-----BEGIN RSA PRIVATE KEY-----\n"
                + raw_key
                + "\n-----END RSA PRIVATE KEY-----\n"
            )
        private_key = raw_key
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
        try:
            response = requests.get(url, headers=headers, timeout=30)
        except requests.exceptions.Timeout as e:
            raise HTTPException(
                status_code=504,
                detail=f"GitHub API request timed out: {e}",
            ) from e
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=502,
                detail=f"GitHub API request failed: {e}",
            ) from e
        if response.status_code == 401:
            detail = (
                f"Failed to get installation for {repo_name}: GitHub returned 401 (JWT could not be decoded). "
                "Check GITHUB_APP_ID and GITHUB_PRIVATE_KEY: use the correct PEM for this App; "
                "if the key is in .env as one line, use literal \\n for newlines."
            )
            raise HTTPException(status_code=400, detail=detail)
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
                    # Check if token looks like a valid GitHub token (starts with gh* and ~40 chars)
                    # If so, it might be plaintext from before encryption was added
                    raw_token = github_provider.access_token
                    is_likely_plaintext = (
                        raw_token
                        and len(raw_token) < 100  # Real tokens are short
                        and (
                            raw_token.startswith("gh")
                            or raw_token.startswith("gho_")
                            or raw_token.startswith("ghs_")
                        )
                    )

                    if is_likely_plaintext:
                        logger.warning(
                            "Failed to decrypt GitHub token for user %s: %s. "
                            "Token looks like plaintext (backward compatibility), using as-is.",
                            uid,
                            str(e),
                        )
                        return raw_token
                    else:
                        # Token is likely encrypted but can't be decrypted
                        # Don't use it - let code fall back to system tokens
                        logger.error(
                            "Failed to decrypt GitHub token for user %s: %s. "
                            "Token appears to be encrypted (length=%d). "
                            "Will fall back to system tokens.",
                            uid,
                            str(e),
                            len(raw_token) if raw_token else 0,
                        )
                        # Don't return the encrypted token - it will cause 414 errors
                        # Fall through to legacy system or system tokens
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

    async def async_get_github_oauth_token(
        self, uid: str, session: AsyncSession
    ) -> Optional[str]:
        """
        Async variant: get user's GitHub OAuth token using AsyncSession.
        Same logic as get_github_oauth_token but with await session.execute(...).
        """
        from app.modules.auth.auth_provider_model import UserAuthProvider
        from app.modules.integrations.token_encryption import decrypt_token

        stmt_user = select(User).where(User.uid == uid).limit(1)
        result_user = await session.execute(stmt_user)
        user = result_user.scalar_one_or_none()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        try:
            stmt_provider = (
                select(UserAuthProvider)
                .where(
                    UserAuthProvider.user_id == uid,
                    UserAuthProvider.provider_type == "firebase_github",
                )
                .limit(1)
            )
            result_provider = await session.execute(stmt_provider)
            github_provider = result_provider.scalar_one_or_none()
            if github_provider and github_provider.access_token:
                logger.info("Found GitHub token in UserAuthProvider for user %s", uid)
                try:
                    return decrypt_token(github_provider.access_token)
                except Exception as e:
                    raw_token = github_provider.access_token
                    is_likely_plaintext = (
                        raw_token
                        and len(raw_token) < 100
                        and (
                            raw_token.startswith("gh")
                            or raw_token.startswith("gho_")
                            or raw_token.startswith("ghs_")
                        )
                    )
                    if is_likely_plaintext:
                        logger.warning(
                            "Failed to decrypt GitHub token for user %s: %s. "
                            "Token looks like plaintext (backward compatibility), using as-is.",
                            uid,
                            str(e),
                        )
                        return raw_token
                    logger.error(
                        "Failed to decrypt GitHub token for user %s: %s. "
                        "Token appears to be encrypted (length=%d). "
                        "Will fall back to system tokens.",
                        uid,
                        str(e),
                        len(raw_token) if raw_token else 0,
                    )
                    return None
        except Exception as e:
            logger.debug("Error checking UserAuthProvider: %s", str(e))

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

    async def get_repos_for_user(
        self, user_id: str, async_session: Optional[AsyncSession] = None
    ):
        if self.is_development_mode:
            return {"repositories": []}

        import time  # Import the time module

        start_time = time.time()  # Start timing the entire method
        try:
            if async_session is not None:
                stmt_user = select(User).where(User.uid == user_id).limit(1)
                result_user = await async_session.execute(stmt_user)
                user = result_user.scalar_one_or_none()
            else:
                user = self.db.query(User).filter(User.uid == user_id).first()
            if user is None:
                raise HTTPException(status_code=404, detail="User not found")

            firebase_uid = user.uid

            # Check if user has GitHub provider via unified auth system
            from app.modules.auth.auth_provider_model import UserAuthProvider

            if async_session is not None:
                stmt_provider = (
                    select(UserAuthProvider)
                    .where(
                        UserAuthProvider.user_id == user_id,
                        UserAuthProvider.provider_type == "firebase_github",
                    )
                    .limit(1)
                )
                result_provider = await async_session.execute(stmt_provider)
                github_provider = result_provider.scalar_one_or_none()
            else:
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

            # Try to get user's OAuth token first (async when async_session provided)
            if async_session is not None:
                github_oauth_token = await self.async_get_github_oauth_token(
                    firebase_uid, async_session
                )
            else:
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
                except GithubException as e:
                    error_str = str(e)
                    is_414_error = (
                        (hasattr(e, "status") and e.status == 414)
                        or "414" in error_str
                        or "URI Too Long" in error_str
                    )
                    if is_414_error:
                        logger.warning(
                            f"414 URI Too Long when fetching user info for user {user_id}. "
                            f"Error: {error_str[:500]}. Cannot retrieve GitHub username from API."
                        )
                    else:
                        logger.warning(
                            f"Failed to get GitHub username from API: {str(e)}"
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

            # Get user organizations - handle 414 errors gracefully
            # NOTE: Using direct API call instead of PyGithub to avoid 414 errors
            # PyGithub's get_orgs() may construct URLs with query parameters that exceed limits
            # Root cause: PyGithub might add pagination/filter params that make URLs too long,
            # or production proxy/load balancer might modify URLs before forwarding
            org_logins = []
            try:
                # Use direct API call instead of PyGithub to avoid URL construction issues
                # This gives us more control over the URL and prevents 414 errors
                orgs_url = "https://api.github.com/user/orgs?per_page=100"
                orgs_headers = {
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {github_oauth_token}",
                    "X-GitHub-Api-Version": "2022-11-28",
                }

                # Log URL length for debugging
                logger.debug(
                    f"Fetching organizations from: {orgs_url} (length: {len(orgs_url)})"
                )

                # Connect 10s, read 30s — orgs list can be slow under load
                response = requests.get(
                    orgs_url, headers=orgs_headers, timeout=(10, 30)
                )
                if response.status_code == 414:
                    logger.warning(
                        f"414 URI Too Long when fetching organizations for user {user_id}. "
                        f"URL length: {len(orgs_url)}. This may indicate a proxy/load balancer issue. "
                        f"Continuing without organization filtering."
                    )
                    org_logins = []  # Continue without org filtering
                elif response.status_code == 200:
                    orgs_data = response.json()
                    org_logins = [org["login"].lower() for org in orgs_data]
                    logger.info(
                        f"Retrieved {len(org_logins)} organizations for user {user_id}"
                    )
                else:
                    logger.warning(
                        f"Failed to get organizations for user {user_id}. "
                        f"Status: {response.status_code}. Response: {response.text[:200]}"
                    )
                    org_logins = []  # Continue without org filtering
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Request error when fetching organizations for user {user_id}: {str(e)}. "
                    f"Continuing without organization filtering."
                )
                org_logins = []  # Continue without org filtering
            except Exception as e:
                logger.warning(
                    f"Failed to get organizations for user {user_id}: {str(e)}. "
                    f"Continuing without organization filtering."
                )
                org_logins = []  # Continue without org filtering

            raw_key = config_provider.get_github_key()
            if not raw_key.startswith("-----BEGIN"):
                raw_key = (
                    "-----BEGIN RSA PRIVATE KEY-----\n"
                    + raw_key
                    + "\n-----END RSA PRIVATE KEY-----\n"
                )
            app_id = os.environ["GITHUB_APP_ID"]

            auth = AppAuth(app_id=app_id, private_key=raw_key)
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
            # Connect 10s, total 60s per request — pagination can have many large responses
            timeout = ClientTimeout(sock_connect=10, total=60)

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
                            if response.status == 414:
                                logger.warning(
                                    f"414 URI Too Long when fetching installations. "
                                    f"URL length: {len(first_url)}. This may indicate an issue with the GitHub App JWT or API. "
                                    f"Returning empty repository list."
                                )
                                # Return empty list gracefully instead of raising exception
                                return {"repositories": []}
                            elif response.status != 200:
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
                                if response.status == 414:
                                    logger.warning(
                                        f"414 URI Too Long for installations page. URL length: {len(url)}. "
                                        f"URL: {url[:200]}... Skipping this page."
                                    )
                                    return []  # Skip this page gracefully
                                elif response.status == 200:
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
                    try:
                        app_auth = auth.get_installation_auth(installation["id"])
                        repos_url = installation["repositories_url"]
                        github = Github(auth=app_auth)  # do not remove this line
                        auth_headers = {"Authorization": f"Bearer {app_auth.token}"}
                    except GithubException as e:
                        if hasattr(e, "status") and e.status == 414:
                            logger.warning(
                                f"414 URI Too Long when getting installation auth for installation {installation['id']}. "
                                f"Skipping this installation."
                            )
                            continue
                        else:
                            # Re-raise other GithubExceptions
                            raise
                    except Exception as e:
                        logger.warning(
                            f"Error getting installation auth for installation {installation['id']}: {str(e)}. "
                            f"Skipping this installation."
                        )
                        continue

                    # Log the original repos_url to debug 414 errors
                    logger.info(
                        f"[get_repos_for_user] Processing installation {installation['id']}, "
                        f"repos_url length: {len(repos_url)}, repos_url: {repos_url[:200]}..."
                    )

                    # Check if original repos_url is already too long (before adding query params)
                    # This can happen if GitHub's API returns a URL with many query parameters
                    if len(repos_url) > 2000:
                        logger.warning(
                            f"Skipping installation {installation['id']} - original repos_url from GitHub API is too long: "
                            f"{len(repos_url)} chars (limit: 2000). URL: {repos_url[:200]}..."
                        )
                        continue

                    # Construct URL with proper query parameter handling
                    # repos_url might already have query params, so use URL parsing
                    # This prevents 414 URI Too Long errors when repos_url has existing query params
                    parsed_url = None
                    try:
                        parsed_url = urlparse(repos_url)
                        query_params = parse_qs(parsed_url.query)
                        # Remove any existing per_page to avoid duplicates
                        query_params.pop("per_page", None)
                        query_params["per_page"] = ["100"]  # Set per_page to 100
                        # Reconstruct URL with updated query params
                        new_query = urlencode(query_params, doseq=True)
                        first_page_url = urlunparse(
                            (
                                parsed_url.scheme,
                                parsed_url.netloc,
                                parsed_url.path,
                                parsed_url.params,
                                new_query,
                                parsed_url.fragment,
                            )
                        )

                        # Log URL length for debugging 414 errors
                        logger.info(
                            f"[get_repos_for_user] Constructed first_page_url for installation {installation['id']}: "
                            f"length={len(first_page_url)}, url={first_page_url[:200]}..."
                        )
                        # GitHub's URI limit is typically 8KB, but some proxies/servers have lower limits
                        # Skip installations with URLs that are too long to prevent 414 errors
                        if len(first_page_url) > 2000:
                            logger.warning(
                                f"Skipping installation {installation['id']} due to URL too long: "
                                f"{len(first_page_url)} chars (limit: 2000). URL: {first_page_url[:200]}..."
                            )
                            continue
                    except Exception as url_error:
                        logger.error(
                            f"Error parsing repos_url for installation {installation['id']}: {url_error}. "
                            f"repos_url: {repos_url}"
                        )
                        # Fallback to simple concatenation if parsing fails
                        # Check if per_page already exists to avoid duplicates
                        if "?" in repos_url:
                            if "per_page=" in repos_url:
                                # Remove all existing per_page params (handle multiple occurrences)
                                repos_url_cleaned = re.sub(
                                    r"[?&]per_page=\d+", "", repos_url
                                )
                                # Fix invalid query string if per_page was the first param
                                # If query string now starts with & (e.g., ?&other=value), remove the &
                                repos_url_cleaned = re.sub(
                                    r"\?&", "?", repos_url_cleaned
                                )
                                # Ensure we have proper separator
                                separator = "&" if "?" in repos_url_cleaned else "?"
                                first_page_url = (
                                    f"{repos_url_cleaned}{separator}per_page=100"
                                )
                            else:
                                first_page_url = f"{repos_url}&per_page=100"
                        else:
                            first_page_url = f"{repos_url}?per_page=100"

                        logger.warning(
                            f"[get_repos_for_user] Using fallback URL construction for installation {installation['id']}: "
                            f"length={len(first_page_url)}, url={first_page_url[:200]}..."
                        )
                        # Check if fallback URL is also too long
                        if len(first_page_url) > 2000:
                            logger.warning(
                                f"Skipping installation {installation['id']} due to fallback URL too long: "
                                f"{len(first_page_url)} chars (limit: 2000). URL: {first_page_url[:200]}..."
                            )
                            continue

                        # For fallback, we need to parse the URL for pagination later
                        try:
                            parsed_url = urlparse(first_page_url)
                        except Exception:
                            parsed_url = None

                    async with session.get(
                        first_page_url, headers=auth_headers
                    ) as response:
                        if response.status == 414:
                            logger.warning(
                                f"414 URI Too Long for installation {installation['id']}. "
                                f"URL length: {len(first_page_url)}. Skipping this installation."
                            )
                            continue
                        elif response.status != 200:
                            error_text = await response.text()
                            logger.error(
                                f"Failed to fetch repositories for installation ID {installation['id']}. "
                                f"Status: {response.status}. Response: {error_text[:500]}"
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
                            # Use the same URL parsing approach to ensure proper query params
                            page_urls = []
                            if parsed_url is None:
                                # If parsed_url is not available, use Link header URLs from GitHub
                                logger.warning(
                                    f"Cannot generate pagination URLs for installation {installation['id']} - parsed_url not available. "
                                    f"Using Link header URLs instead."
                                )
                                # Extract page URLs from Link header
                                if "Link" in response.headers:
                                    links = self._parse_link_header(
                                        response.headers["Link"]
                                    )
                                    for rel, url in links.items():
                                        # Handle "next" link or page links (e.g., "page2", "page3")
                                        should_add = False
                                        if rel == "next":
                                            should_add = True
                                        elif rel.startswith("page"):
                                            try:
                                                page_num = int(rel.replace("page", ""))
                                                if page_num > 1:
                                                    should_add = True
                                            except (ValueError, AttributeError):
                                                # Skip if rel format is unexpected (e.g., "pageabc")
                                                logger.debug(
                                                    f"Skipping unexpected rel format in Link header: {rel}"
                                                )
                                                continue

                                        if should_add:
                                            # Check URL length before adding to prevent 414 errors
                                            if len(url) > 2000:
                                                logger.warning(
                                                    f"Skipping {rel} page URL for installation {installation['id']} - URL too long: {len(url)} chars (limit: 2000). URL: {url[:200]}..."
                                                )
                                                continue
                                            page_urls.append(url)
                            else:
                                for page in range(2, last_page + 1):
                                    page_query_params = parse_qs(parsed_url.query)
                                    # Remove existing page param to avoid duplicates
                                    page_query_params.pop("page", None)
                                    page_query_params["per_page"] = ["100"]
                                    page_query_params["page"] = [str(page)]
                                    page_query = urlencode(
                                        page_query_params, doseq=True
                                    )
                                    page_url = urlunparse(
                                        (
                                            parsed_url.scheme,
                                            parsed_url.netloc,
                                            parsed_url.path,
                                            parsed_url.params,
                                            page_query,
                                            parsed_url.fragment,
                                        )
                                    )
                                    # Check URL length before adding
                                    if len(page_url) > 2000:
                                        logger.warning(
                                            f"Skipping page {page} for installation {installation['id']} - URL too long: {len(page_url)} chars"
                                        )
                                        continue
                                    page_urls.append(page_url)

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
                                        if response.status == 414:
                                            logger.warning(
                                                "414 URI Too Long for pagination request. Skipping this page."
                                            )
                                            continue
                                        elif response.status == 200:
                                            page_data = await response.json()
                                            repos.extend(
                                                page_data.get("repositories", [])
                                            )
                                        else:
                                            error_text = await response.text()
                                            logger.error(
                                                f"Failed to fetch repositories page. Status: {response.status}. Response: {error_text[:500]}"
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

        except GithubException as e:
            # Handle 414 errors from PyGithub specifically
            # Check both status attribute and error message string (PyGithub may put status in message)
            error_str = str(e)
            is_414_error = (
                (hasattr(e, "status") and e.status == 414)
                or "414" in error_str
                or "URI Too Long" in error_str
            )

            if is_414_error:
                logger.warning(
                    f"414 URI Too Long error from PyGithub for user {user_id}. "
                    f"Error: {error_str[:500]}. "
                    f"This may indicate an issue with GitHub API URL construction. "
                    f"Returning empty repository list."
                )
                return {"repositories": []}
            else:
                # Re-raise other GithubExceptions
                logger.exception(
                    "GithubException in get_repos_for_user", user_id=user_id
                )
                raise HTTPException(
                    status_code=e.status if hasattr(e, "status") else 500,
                    detail=f"GitHub API error: {error_str}",
                ) from e
        except Exception as e:
            # Check if this is a 414 error that might have been wrapped
            error_str = str(e)
            is_414_error = (
                "414" in error_str
                or "URI Too Long" in error_str
                or (hasattr(e, "status") and e.status == 414)
            )

            if is_414_error:
                logger.warning(
                    f"414 URI Too Long error (caught as generic Exception) for user {user_id}. "
                    f"Error: {error_str[:500]}. Returning empty repository list."
                )
                return {"repositories": []}

            logger.exception("Failed to fetch repositories", user_id=user_id)
            raise HTTPException(
                status_code=500, detail="Failed to fetch repositories"
            ) from e
        finally:
            total_duration = time.time() - start_time  # Calculate total duration
            logger.info(
                f"get_repos_for_user executed in {total_duration:.2f} seconds"
            )  # Log total duration

    async def get_combined_user_repos(
        self, user_id: str, async_session: Optional[AsyncSession] = None
    ):
        if async_session is not None:
            subquery = (
                select(Project.repo_name, func.min(Project.id).label("min_id"))
                .where(Project.user_id == user_id)
                .group_by(Project.repo_name)
                .subquery()
            )
            stmt = (
                select(Project)
                .join(
                    subquery,
                    (Project.repo_name == subquery.c.repo_name)
                    & (Project.id == subquery.c.min_id),
                )
            )
            result = await async_session.execute(stmt)
            projects = result.scalars().all()
        else:
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
        user_repo_response = await self.get_repos_for_user(
            user_id, async_session=async_session
        )
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
                    git = _get_git_module()
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
        cached_structure = None
        async_redis = await _get_async_redis_cache()
        try:
            if async_redis:
                cached_structure = await async_redis.get(cache_key)
            else:
                cached_structure = await asyncio.to_thread(
                    self.redis.get, cache_key
                )
        except (RedisError, OSError) as e:
            logger.warning(
                "Redis cache read failed for project_structure (cache_key=%s): %s",
                cache_key,
                e,
            )
            cached_structure = None

        if cached_structure:
            logger.info(
                f"Project structure found in cache for project ID: {project_id}, path: {path}"
            )
            return (
                cached_structure.decode("utf-8")
                if isinstance(cached_structure, bytes)
                else cached_structure
            )

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

            async_redis = await _get_async_redis_cache()
            try:
                if async_redis:
                    await async_redis.setex(
                        cache_key, 3600, formatted_structure
                    )  # Cache for 1 hour
                else:
                    await asyncio.to_thread(
                        self.redis.setex, cache_key, 3600, formatted_structure
                    )
            except (RedisError, OSError) as e:
                logger.warning(
                    "Redis cache write failed for project_structure (cache_key=%s): %s",
                    cache_key,
                    e,
                )

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

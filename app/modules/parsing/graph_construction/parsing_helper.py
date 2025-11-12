import json
import logging
import os
import shutil
import tarfile
import uuid
from pathlib import Path
from typing import Any, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import requests
import requests.auth
from fastapi import HTTPException
from git import GitCommandError, Repo
from sqlalchemy.orm import Session

from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.parsing.graph_construction.parsing_schema import RepoDetails
from app.modules.parsing.utils.repo_name_normalizer import normalize_repo_name
from app.modules.projects.projects_schema import ProjectStatusEnum
from app.modules.projects.projects_service import ProjectService

logger = logging.getLogger(__name__)


class ParsingServiceError(Exception):
    """Base exception class for ParsingService errors."""


class ParsingFailedError(ParsingServiceError):
    """Raised when a parsing fails."""


class ParseHelper:
    def __init__(self, db_session: Session):
        self.project_manager = ProjectService(db_session)
        self.db = db_session
        self.github_service = CodeProviderService(db_session)

        # Initialize repo manager if enabled
        self.repo_manager = None
        try:
            repo_manager_enabled = (
                os.getenv("REPO_MANAGER_ENABLED", "false").lower() == "true"
            )
            if repo_manager_enabled:
                from app.modules.repo_manager import RepoManager

                self.repo_manager = RepoManager()
                logger.info("RepoManager initialized in ParseHelper")
        except Exception as e:
            logger.warning(f"Failed to initialize RepoManager: {e}")

    @staticmethod
    def get_directory_size(path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size

    async def clone_or_copy_repository(
        self, repo_details: RepoDetails, user_id: str
    ) -> Tuple[Any, str, Any]:
        owner = None
        auth = None
        repo = None

        if repo_details.repo_path:
            if not os.path.exists(repo_details.repo_path):
                raise HTTPException(
                    status_code=400,
                    detail="Local repository does not exist on the given path",
                )
            repo = Repo(repo_details.repo_path)
            logger.info(
                f"ParsingHelper: clone_or_copy_repository created local Repo object for path: {repo_details.repo_path}"
            )
        else:
            try:
                github, repo = self.github_service.get_repo(repo_details.repo_name)
                owner = repo.owner.login

                # Extract auth from the Github client
                # The auth is stored in the _Github__requester.auth attribute
                if hasattr(github, "_Github__requester") and hasattr(
                    github._Github__requester, "auth"
                ):
                    auth = github._Github__requester.auth
                elif hasattr(github, "get_app_auth"):
                    # Fallback for older method
                    auth = github.get_app_auth()
                else:
                    logger.warning(
                        f"Could not extract auth from GitHub client for {repo_details.repo_name}"
                    )
            except HTTPException as he:
                raise he
            except Exception as e:
                logger.error(f"Failed to fetch repository: {str(e)}")
                raise HTTPException(
                    status_code=404,
                    detail="Repository not found or inaccessible on GitHub",
                )

        return repo, owner, auth

    def is_text_file(self, file_path):
        def open_text_file(file_path):
            """
            Try multiple encodings to detect if file is text.

            Order of encodings to try:
            1. utf-8 (most common)
            2. utf-8-sig (UTF-8 with BOM)
            3. utf-16 (common in Windows C# files)
            4. latin-1/iso-8859-1 (fallback, accepts all byte sequences)
            """
            encodings = ["utf-8", "utf-8-sig", "utf-16", "latin-1"]

            for encoding in encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        # Read first 8KB to detect encoding
                        f.read(8192)
                    return True
                except (UnicodeDecodeError, UnicodeError):
                    continue
                except Exception:
                    # Handle other errors (permissions, file not found, etc.)
                    return False

            # If all encodings fail, likely a binary file
            return False

        ext = file_path.split(".")[-1]
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
        ]
        include_extensions = [
            "py",
            "js",
            "ts",
            "c",
            "cs",
            "cpp",
            "h",
            "hpp",
            "el",
            "ex",
            "exs",
            "elm",
            "go",
            "java",
            "ml",
            "mli",
            "php",
            "ql",
            "rb",
            "rs",
            "md",
            "txt",
            "json",
            "yaml",
            "yml",
            "toml",
            "ini",
            "cfg",
            "conf",
            "xml",
            "html",
            "css",
            "sh",
            "ps1",
            "psm1",
            "md",
            "mdx",
            "xsq",
            "proto",
        ]
        if ext in exclude_extensions:
            return False
        elif ext in include_extensions or open_text_file(file_path):
            return True
        else:
            return False

    async def download_and_extract_tarball(
        self, repo, branch, target_dir, auth, repo_details, user_id
    ):
        # Get repo name for logging - handle both Repo objects and repo objects with full_name
        repo_name = (
            repo.working_tree_dir
            if isinstance(repo, Repo)
            else getattr(repo, "full_name", "unknown")
        )

        logger.info(
            f"ParsingHelper: Starting tarball download for repo '{repo_name}', branch '{branch}'"
        )

        try:
            logger.info(
                f"ParsingHelper: Getting archive link for repo '{repo_name}', branch '{branch}'"
            )
            tarball_url = repo.get_archive_link("tarball", branch)
            logger.info(f"ParsingHelper: Retrieved tarball URL: {tarball_url}")

            # Validate that tarball_url is a string, not an exception object
            if not isinstance(tarball_url, str):
                logger.error(
                    f"ParsingHelper: Invalid tarball URL type: {type(tarball_url)}, value: {tarball_url}"
                )
                raise ValueError(
                    f"Expected string URL, got {type(tarball_url)}: {tarball_url}"
                )

            # For GitBucket private repos, use PyGithub client's requester for authenticated requests
            # According to GitBucket API docs: https://github.com/gitbucket/gitbucket/wiki/API-WebHook
            # Authentication: "Authorization: token YOUR_TOKEN" in header
            provider_type = os.getenv("CODE_PROVIDER", "github").lower()

            if (
                provider_type == "gitbucket"
                and hasattr(repo, "_provider")
                and repo._provider
            ):
                # For GitBucket, use the provider's authentication
                # According to GitBucket API docs: https://github.com/gitbucket/gitbucket/wiki/API-WebHook
                # Authentication format: "Authorization: token YOUR_TOKEN" in header
                try:
                    github_client = repo._provider.client
                    if hasattr(github_client, "_Github__requester"):
                        requester = github_client._Github__requester

                        # Use the requester's session which has authentication already configured
                        if hasattr(requester, "_Requester__session"):
                            session = requester._Requester__session
                            response = session.get(tarball_url, stream=True, timeout=30)

                            # If we get 401, the session auth might not be working, fall back to manual token
                            if response.status_code == 401:
                                raise requests.exceptions.HTTPError(
                                    "401 Unauthorized from session"
                                )
                        else:
                            raise AttributeError("Requester session not available")
                    else:
                        raise AttributeError("Requester not found")
                except Exception:
                    # Fallback to manual token extraction
                    token = None
                    headers = {}

                    # Priority 1: Try to get token from auth parameter
                    if auth and hasattr(auth, "token"):
                        token = auth.token

                    # Priority 2: Try to extract from PyGithub client's requester
                    if not token and hasattr(repo, "_provider") and repo._provider:
                        try:
                            github_client = repo._provider.client
                            if hasattr(github_client, "_Github__requester"):
                                requester = github_client._Github__requester

                                if hasattr(requester, "auth") and hasattr(
                                    requester.auth, "token"
                                ):
                                    token = requester.auth.token
                                elif hasattr(
                                    requester, "_Requester__authorizationHeader"
                                ):
                                    auth_header = (
                                        requester._Requester__authorizationHeader
                                    )
                                    if auth_header:
                                        if auth_header.startswith("token "):
                                            token = auth_header[6:]
                                        elif auth_header.startswith("Bearer "):
                                            token = auth_header[7:]
                        except Exception:
                            pass  # Token extraction failed, will try next method

                    # Priority 3: Fallback to environment variable
                    if not token:
                        token = os.getenv("CODE_PROVIDER_TOKEN")

                    if not token:
                        error_msg = "No authentication token available for GitBucket archive download"
                        logger.error(f"ParsingHelper: {error_msg}")
                        raise ValueError(error_msg)

                    # GitBucket web endpoints (archive downloads) may require Basic Auth
                    # Try token header format first (API standard per GitBucket docs)
                    headers = {"Authorization": f"token {token}"}
                    logger.debug(
                        "ParsingHelper: Attempting archive download with token header"
                    )

                    response = requests.get(
                        tarball_url, stream=True, headers=headers, timeout=30
                    )

                    # If token header fails with 401, try Basic Auth with repo owner username
                    # GitBucket web endpoints sometimes require Basic Auth (supported since v4.3)
                    if response.status_code == 401:
                        logger.debug(
                            "ParsingHelper: Token header auth failed, trying Basic Auth"
                        )
                        response.close()

                        # Try Basic Auth with repo owner username and token as password
                        if hasattr(repo, "owner") and hasattr(repo.owner, "login"):
                            username = repo.owner.login
                            basic_auth = requests.auth.HTTPBasicAuth(username, token)
                            response = requests.get(
                                tarball_url, stream=True, auth=basic_auth, timeout=30
                            )
                            logger.debug(
                                f"ParsingHelper: Basic Auth response status: {response.status_code}"
                            )
            else:
                # For GitHub and other providers, use standard token auth
                headers = {}
                if auth:
                    headers = {"Authorization": f"token {auth.token}"}
                response = requests.get(
                    tarball_url, stream=True, headers=headers, timeout=30
                )

            response.raise_for_status()

        except requests.exceptions.HTTPError as e:
            # If we get 401, archive download might not be supported for private repos
            # Fall back to git clone for GitBucket
            status_code = None
            if hasattr(e, "response") and e.response is not None:
                status_code = e.response.status_code
            elif "401" in str(e):
                status_code = 401

            if status_code == 401:
                provider_type = os.getenv("CODE_PROVIDER", "github").lower()
                if provider_type == "gitbucket":
                    logger.info(
                        "ParsingHelper: Archive download failed with 401 for GitBucket private repo, "
                        "falling back to git clone"
                    )
                    return await self._clone_repository_with_auth(
                        repo, branch, target_dir, user_id
                    )

            logger.error(f"ParsingHelper: Failed to download repository archive: {e}")
            raise ParsingFailedError("Failed to download repository archive") from e
        except requests.exceptions.RequestException as e:
            logger.error(f"ParsingHelper: Error fetching tarball: {e}")
            raise ParsingFailedError("Failed to download repository archive") from e
        except Exception as e:
            logger.exception("ParsingHelper: Unexpected error in tarball download")
            raise ParsingFailedError(
                "Unexpected error during repository download"
            ) from e
        tarball_path = os.path.join(
            target_dir,
            f"{repo.full_name.replace('/', '-').replace('.', '-')}-{branch.replace('/', '-').replace('.', '-')}.tar.gz",
        )

        final_dir = os.path.join(
            target_dir,
            f"{repo.full_name.replace('/', '-').replace('.', '-')}-{branch.replace('/', '-').replace('.', '-')}-{user_id}",
        )

        logger.info(f"ParsingHelper: Tarball path: {tarball_path}")
        logger.info(f"ParsingHelper: Final directory: {final_dir}")

        try:
            logger.info(f"ParsingHelper: Writing tarball to {tarball_path}")
            with open(tarball_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            tarball_size = os.path.getsize(tarball_path)
            logger.info(
                f"ParsingHelper: Successfully downloaded tarball, size: {tarball_size} bytes"
            )

            # Validate tarball size - very small files are likely error responses
            if tarball_size < 100:
                error_msg = (
                    f"Tarball is suspiciously small ({tarball_size} bytes). "
                    "This may indicate an error response from the server or an empty repository."
                )
                logger.error(f"ParsingHelper: {error_msg}")
                raise ParsingFailedError(error_msg)

            logger.info(f"ParsingHelper: Extracting tarball to {final_dir}")
            try:
                with tarfile.open(tarball_path, "r:gz") as tar:
                    # Validate that the tarball is not empty
                    if not tar.getmembers():
                        error_msg = "Tarball contains no files"
                        logger.error(f"ParsingHelper: {error_msg}")
                        raise ParsingFailedError(error_msg)

                    temp_dir = os.path.join(final_dir, "temp_extract")
                    os.makedirs(temp_dir, exist_ok=True)
                    tar.extractall(path=temp_dir)
                    logger.info(
                        f"ParsingHelper: Extracted tarball contents to {temp_dir}"
                    )

                    # Check if extraction directory has contents
                    extracted_contents = os.listdir(temp_dir)
                    if not extracted_contents:
                        error_msg = (
                            "Tarball extraction resulted in empty directory. "
                            "The archive may be corrupted or the repository may be empty."
                        )
                        logger.error(f"ParsingHelper: {error_msg}")
                        raise ParsingFailedError(error_msg)

                    extracted_dir = os.path.join(temp_dir, extracted_contents[0])
                    logger.info(
                        f"ParsingHelper: Main extracted directory: {extracted_dir}"
                    )

                    text_files_count = 0
                    for root, dirs, files in os.walk(extracted_dir):
                        for file in files:
                            if file.startswith("."):
                                continue
                            file_path = os.path.join(root, file)
                            if self.is_text_file(file_path):
                                try:
                                    relative_path = os.path.relpath(
                                        file_path, extracted_dir
                                    )
                                    dest_path = os.path.join(final_dir, relative_path)
                                    os.makedirs(
                                        os.path.dirname(dest_path), exist_ok=True
                                    )
                                    shutil.copy2(file_path, dest_path)
                                    text_files_count += 1
                                except (shutil.Error, OSError) as e:
                                    logger.error(
                                        f"ParsingHelper: Error copying file {file_path}: {e}"
                                    )

                    logger.info(
                        f"ParsingHelper: Copied {text_files_count} text files to final directory"
                    )
                    # Remove the temporary directory
                    try:
                        shutil.rmtree(temp_dir)
                    except OSError as e:
                        logger.error(f"Error removing temporary directory: {e}")
                        pass
            except tarfile.TarError as e:
                error_msg = f"Failed to extract tarball: {e}. The archive may be corrupted or invalid."
                logger.error(f"ParsingHelper: {error_msg}")
                raise ParsingFailedError(error_msg) from e

        except (IOError, tarfile.TarError, shutil.Error) as e:
            logger.error(f"Error handling tarball: {e}")
            raise ParsingFailedError("Failed to process repository archive") from e
        finally:
            if os.path.exists(tarball_path):
                os.remove(tarball_path)

        return final_dir

    async def _clone_repository_with_auth(self, repo, branch, target_dir, user_id):
        """
        Clone repository using git with authentication.
        Fallback method when archive download fails for private repos.

        Requires GITBUCKET_USERNAME and GITBUCKET_PASSWORD environment variables.

        This method clones to a temporary directory, filters text files using is_text_file(),
        and copies only text files to the final directory to prevent binary file parsing errors.
        """
        repo_name = (
            repo.working_tree_dir
            if isinstance(repo, Repo)
            else getattr(repo, "full_name", "unknown")
        )

        logger.info(
            f"ParsingHelper: Cloning repository '{repo_name}' branch '{branch}' using git"
        )

        final_dir = os.path.join(
            target_dir,
            f"{repo.full_name.replace('/', '-').replace('.', '-')}-{branch.replace('/', '-').replace('.', '-')}-{user_id}",
        )

        # Create temporary clone directory
        temp_clone_dir = os.path.join(target_dir, f"{uuid.uuid4()}_temp_clone")

        # Get credentials from environment variables
        username = os.getenv("GITBUCKET_USERNAME")
        password = os.getenv("GITBUCKET_PASSWORD")

        if not username or not password:
            error_msg = (
                "GITBUCKET_USERNAME and GITBUCKET_PASSWORD environment variables "
                "are required for cloning private GitBucket repositories"
            )
            logger.error(f"ParsingHelper: {error_msg}")
            raise ParsingFailedError(error_msg)

        # Construct GitBucket clone URL with embedded credentials
        # Format: http://username:password@hostname/path/owner/repo.git
        base_url = os.getenv("CODE_PROVIDER_BASE_URL", "http://localhost:8080")
        if base_url.endswith("/api/v3"):
            base_url = base_url[:-7]  # Remove '/api/v3'

        parsed = urlparse(base_url)
        # Preserve the path component from base URL (e.g., /gitbucket)
        base_path = parsed.path.rstrip("/")  # Remove trailing slash if present
        repo_path = (
            f"{base_path}/{repo.full_name}.git"
            if base_path
            else f"/{repo.full_name}.git"
        )

        clone_url_with_auth = urlunparse(
            (
                parsed.scheme,
                f"{username}:{password}@{parsed.netloc}",
                repo_path,
                "",
                "",
                "",
            )
        )

        # Log URL without credentials for security
        safe_url = urlunparse((parsed.scheme, parsed.netloc, repo_path, "", "", ""))
        logger.info(f"ParsingHelper: Cloning from {safe_url}")

        try:
            # Clone the repository to temporary directory with shallow clone for faster download
            repo_obj = Repo.clone_from(
                clone_url_with_auth, temp_clone_dir, branch=branch, depth=1
            )
            logger.info(
                f"ParsingHelper: Successfully cloned repository to temporary directory: {temp_clone_dir}"
            )

            # Filter and copy only text files to final directory
            logger.info(
                f"ParsingHelper: Filtering text files from clone to {final_dir}"
            )
            os.makedirs(final_dir, exist_ok=True)

            text_files_count = 0
            for root, dirs, files in os.walk(temp_clone_dir):
                # Skip .git directory
                if ".git" in root.split(os.sep):
                    continue

                # Skip hidden directories
                if any(part.startswith(".") for part in root.split(os.sep)):
                    continue

                for file in files:
                    # Skip hidden files
                    if file.startswith("."):
                        continue

                    file_path = os.path.join(root, file)

                    # Filter using is_text_file check
                    if self.is_text_file(file_path):
                        try:
                            # Calculate relative path from clone root
                            relative_path = os.path.relpath(file_path, temp_clone_dir)
                            dest_path = os.path.join(final_dir, relative_path)

                            # Create destination directory structure
                            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                            # Copy text file to final directory
                            shutil.copy2(file_path, dest_path)
                            text_files_count += 1
                        except (shutil.Error, OSError) as e:
                            logger.error(
                                f"ParsingHelper: Error copying file {file_path}: {e}"
                            )

            logger.info(
                f"ParsingHelper: Copied {text_files_count} text files from git clone to final directory"
            )

            # Clean up temporary clone directory
            try:
                shutil.rmtree(temp_clone_dir)
                logger.info(
                    f"ParsingHelper: Cleaned up temporary clone directory: {temp_clone_dir}"
                )
            except Exception as e:
                logger.warning(
                    f"ParsingHelper: Failed to clean up temp clone directory: {e}"
                )

            return final_dir

        except GitCommandError as e:
            logger.error(f"ParsingHelper: Git clone failed: {e}")
            # Clean up temp directory on error
            if os.path.exists(temp_clone_dir):
                try:
                    shutil.rmtree(temp_clone_dir)
                except Exception:
                    pass
            raise ParsingFailedError(f"Failed to clone repository: {e}") from e
        except Exception as e:
            logger.error(f"ParsingHelper: Unexpected error during git clone: {e}")
            # Clean up temp directory on error
            if os.path.exists(temp_clone_dir):
                try:
                    shutil.rmtree(temp_clone_dir)
                except Exception:
                    pass
            raise ParsingFailedError(
                f"Unexpected error during repository clone: {e}"
            ) from e

    @staticmethod
    def detect_repo_language(repo_dir):
        lang_count = {
            "c_sharp": 0,
            "c": 0,
            "cpp": 0,
            "elisp": 0,
            "elixir": 0,
            "elm": 0,
            "go": 0,
            "java": 0,
            "javascript": 0,
            "ocaml": 0,
            "php": 0,
            "python": 0,
            "ql": 0,
            "ruby": 0,
            "rust": 0,
            "typescript": 0,
            "markdown": 0,
            "xml": 0,
            "other": 0,
        }
        total_chars = 0

        try:
            for root, _, files in os.walk(repo_dir):
                if any(part.startswith(".") for part in root.split(os.sep)):
                    continue

                for file in files:
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file)[1].lower()

                    # Try multiple encodings for robustness
                    content = None
                    encodings = ["utf-8", "utf-8-sig", "utf-16", "latin-1"]

                    for encoding in encodings:
                        try:
                            with open(file_path, "r", encoding=encoding) as f:
                                content = f.read()
                                break
                        except (UnicodeDecodeError, UnicodeError):
                            continue
                        except Exception:
                            break

                    if content is not None:
                        try:
                            total_chars += len(content)
                            if ext == ".cs":
                                lang_count["c_sharp"] += 1
                            elif ext == ".c":
                                lang_count["c"] += 1
                            elif ext in [".cpp", ".cxx", ".cc"]:
                                lang_count["cpp"] += 1
                            elif ext == ".el":
                                lang_count["elisp"] += 1
                            elif ext == ".ex" or ext == ".exs":
                                lang_count["elixir"] += 1
                            elif ext == ".elm":
                                lang_count["elm"] += 1
                            elif ext == ".go":
                                lang_count["go"] += 1
                            elif ext == ".java":
                                lang_count["java"] += 1
                            elif ext in [".js", ".jsx"]:
                                lang_count["javascript"] += 1
                            elif ext == ".ml" or ext == ".mli":
                                lang_count["ocaml"] += 1
                            elif ext == ".php":
                                lang_count["php"] += 1
                            elif ext == ".py":
                                lang_count["python"] += 1
                            elif ext == ".ql":
                                lang_count["ql"] += 1
                            elif ext == ".rb":
                                lang_count["ruby"] += 1
                            elif ext == ".rs":
                                lang_count["rust"] += 1
                            elif ext in [".ts", ".tsx"]:
                                lang_count["typescript"] += 1
                            elif ext in [".md", ".mdx"]:
                                lang_count["markdown"] += 1
                            elif ext in [".xml", ".xsq"]:
                                lang_count["xml"] += 1
                            else:
                                lang_count["other"] += 1
                        except Exception as e:
                            logger.warning(f"Error processing file {file_path}: {e}")
                            continue
                    else:
                        logger.warning(
                            f"Could not read file with any encoding: {file_path}"
                        )
                        continue
        except (TypeError, FileNotFoundError, PermissionError) as e:
            logger.error(f"Error accessing directory '{repo_dir}': {e}")

        # Determine the predominant language based on counts
        predominant_language = max(lang_count, key=lang_count.get)
        return predominant_language if lang_count[predominant_language] > 0 else "other"

    async def setup_project_directory(
        self,
        repo,
        branch,
        auth,
        repo_details,
        user_id,
        project_id=None,  # Change type to str
        commit_id=None,
    ):
        # Check if this is a local repository by examining the repo object
        # In development mode: repo is Repo object, repo_details is ParsingRequest
        # In non-development mode: both repo and repo_details can be Repo objects
        logger.info(
            f"ParsingHelper: setup_project_directory called with repo type: {type(repo).__name__}, "
            f"repo_details type: {type(repo_details).__name__}"
        )

        if isinstance(repo, Repo):
            # Local repository - use full path from Repo object
            repo_path = repo.working_tree_dir
            full_name = repo_path.split("/")[
                -1
            ]  # Extract just the directory name for display
            logger.info(
                f"ParsingHelper: Detected local repository at {repo_path} with name {full_name}"
            )
        elif isinstance(repo_details, Repo):
            # Alternative: repo_details is the Repo object (non-dev mode)
            repo_path = repo_details.working_tree_dir
            full_name = repo_path.split("/")[-1]
            logger.info(
                f"ParsingHelper: Detected local repository at {repo_path} with name {full_name}"
            )
        else:
            # Remote repository - get name from repo_details (ParsingRequest)
            repo_path = None
            if hasattr(repo_details, "repo_name"):
                full_name = repo_details.repo_name
            else:
                full_name = repo.full_name if hasattr(repo, "full_name") else None
            logger.info(f"ParsingHelper: Detected remote repository {full_name}")

        if full_name is None:
            full_name = repo_path.split("/")[-1] if repo_path else "unknown"

        # Normalize repository name for consistent database lookups
        normalized_full_name = normalize_repo_name(full_name)
        logger.info(
            f"ParsingHelper: Original full_name: {full_name}, Normalized: {normalized_full_name}, repo_path: {repo_path}"
        )

        project = await self.project_manager.get_project_from_db(
            normalized_full_name, branch, user_id, repo_path, commit_id
        )
        if not project:
            project_id = await self.project_manager.register_project(
                normalized_full_name,
                branch,
                user_id,
                project_id,
                commit_id=commit_id,
                repo_path=repo_path,  # Pass repo_path when registering
            )
        if repo_path is not None:
            # Local repository detected - return the path directly without downloading tarball
            logger.info(f"ParsingHelper: Using local repository at {repo_path}")
            return repo_path, project_id
        if isinstance(repo_details, Repo):
            extracted_dir = repo_details.working_tree_dir
            try:
                current_dir = os.getcwd()
                os.chdir(extracted_dir)  # Change to the cloned repo directory
                if commit_id:
                    repo_details.git.checkout(commit_id)
                    latest_commit_sha = commit_id
                else:
                    repo_details.git.checkout(branch)
                    branch_details = repo_details.head.commit
                    latest_commit_sha = branch_details.hexsha
            except GitCommandError as e:
                logger.error(
                    f"Error checking out {'commit' if commit_id else 'branch'}: {e}"
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to checkout {'commit ' + commit_id if commit_id else 'branch ' + branch}",
                )
            finally:
                os.chdir(current_dir)  # Restore the original working directory
        else:
            try:
                if commit_id:
                    # For GitHub API, we need to download tarball for specific commit
                    extracted_dir = await self.download_and_extract_tarball(
                        repo,
                        commit_id,
                        os.getenv("PROJECT_PATH"),
                        auth,
                        repo_details,
                        user_id,
                    )
                    latest_commit_sha = commit_id
                else:
                    extracted_dir = await self.download_and_extract_tarball(
                        repo,
                        branch,
                        os.getenv("PROJECT_PATH"),
                        auth,
                        repo_details,
                        user_id,
                    )
                    # Use repo.get_branch() instead of repo_details.get_branch()
                    # repo is the MockRepo object (or PyGithub Repository) with get_branch method
                    # repo_details can be ParsingRequest in dev mode, which doesn't have get_branch
                    branch_details = repo.get_branch(branch)
                    latest_commit_sha = branch_details.commit.sha
            except ParsingFailedError as e:
                logger.exception("Failed to download repository")
                raise HTTPException(
                    status_code=500, detail=f"Repository download failed: {e}"
                ) from e
            except Exception as e:
                logger.exception("Unexpected error during repository download")
                raise HTTPException(
                    status_code=500, detail=f"Repository download failed: {e}"
                ) from e

        # Use repo instead of repo_details for metadata extraction
        # repo is always the MockRepo (remote) or Repo (local) object with required methods
        # repo_details can be ParsingRequest in dev mode, which lacks these methods
        repo_metadata = ParseHelper.extract_repository_metadata(repo)
        repo_metadata["error_message"] = None
        project_metadata = json.dumps(repo_metadata).encode("utf-8")
        ProjectService.update_project(
            self.db,
            project_id,
            properties=project_metadata,
            commit_id=latest_commit_sha,
            status=ProjectStatusEnum.CLONED.value,
        )

        # Copy repo to .repos if repo manager is enabled
        if self.repo_manager and extracted_dir and os.path.exists(extracted_dir):
            try:
                await self._copy_repo_to_repo_manager(
                    normalized_full_name,
                    extracted_dir,
                    branch,
                    latest_commit_sha,
                    user_id,
                    repo_metadata,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to copy repo to repo manager: {e}. Continuing with parsing."
                )

        return extracted_dir, project_id

    async def _copy_repo_to_repo_manager(
        self,
        repo_name: str,
        extracted_dir: str,
        branch: Optional[str],
        commit_id: Optional[str],
        user_id: str,
        metadata: dict,
    ):
        """
        Copy repository to .repos folder using git worktree and register with repo manager.

        Args:
            repo_name: Full repository name (e.g., 'owner/repo')
            extracted_dir: Path to extracted repository
            branch: Branch name
            commit_id: Commit SHA
            user_id: User ID
            metadata: Repository metadata
        """
        if not self.repo_manager:
            return

        # Check if repo is already available
        if self.repo_manager.is_repo_available(
            repo_name, branch=branch, commit_id=commit_id, user_id=user_id
        ):
            logger.info(
                f"Repo {repo_name}@{commit_id or branch} already available in repo manager"
            )
            # Update last accessed time
            self.repo_manager.update_last_accessed(
                repo_name, branch=branch, commit_id=commit_id, user_id=user_id
            )
            return

        # Determine base repo path in .repos (hierarchical: owner/repo)
        base_repo_path = self.repo_manager._get_repo_local_path(repo_name)

        # Determine ref (commit_id takes precedence over branch)
        ref = commit_id if commit_id else branch
        if not ref:
            logger.warning(
                f"No branch or commit_id provided for {repo_name}, skipping worktree creation"
            )
            return

        try:
            # Initialize or get the base git repository
            base_repo = self._initialize_base_repo(base_repo_path, extracted_dir)

            # Create worktree for the specific branch/commit
            worktree_path = self._create_worktree(
                base_repo, ref, commit_id is not None, extracted_dir
            )

            logger.info(f"Created worktree for {repo_name}@{ref} at {worktree_path}")

            # Register with repo manager (store worktree path)
            self.repo_manager.register_repo(
                repo_name=repo_name,
                local_path=str(worktree_path),
                branch=branch,
                commit_id=commit_id,
                user_id=user_id,
                metadata=metadata,
            )
            logger.info(
                f"Registered repo {repo_name}@{ref} with repo manager at {worktree_path}"
            )
        except Exception as e:
            logger.error(f"Error creating worktree for repo manager: {e}")
            raise

    def _initialize_base_repo(self, base_repo_path: Path, extracted_dir: str) -> Repo:
        """
        Initialize or get the base git repository.

        If the base repo doesn't exist, initialize it and copy the extracted repo.
        If it exists, return the existing repo.
        """
        from git import Repo, InvalidGitRepositoryError

        # Check if base repo already exists and is a valid git repo
        if base_repo_path.exists():
            try:
                base_repo = Repo(base_repo_path)
                logger.info(f"Using existing base repo at {base_repo_path}")
                return base_repo
            except InvalidGitRepositoryError:
                logger.warning(
                    f"Path {base_repo_path} exists but is not a git repo, removing"
                )
                shutil.rmtree(base_repo_path)

        # Create base directory
        base_repo_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize bare repository (worktrees need a bare or regular repo)
        # We'll use a regular repo with a detached HEAD initially
        logger.info(f"Initializing base git repository at {base_repo_path}")

        # Copy extracted repo to base location
        shutil.copytree(extracted_dir, base_repo_path, dirs_exist_ok=True)

        # Initialize git repo if not already a git repo
        try:
            base_repo = Repo(base_repo_path)
        except InvalidGitRepositoryError:
            # Initialize new git repo
            base_repo = Repo.init(base_repo_path)
            # Add all files and create initial commit
            base_repo.git.add(A=True)
            try:
                base_repo.index.commit("Initial commit from parsing")
            except Exception as e:
                logger.warning(f"Could not create initial commit: {e}")

        return base_repo

    def _create_worktree(
        self, base_repo: Repo, ref: str, is_commit: bool, extracted_dir: str
    ) -> Path:
        """
        Create a git worktree for the given ref.

        Args:
            base_repo: Base git repository
            ref: Branch name or commit SHA
            is_commit: Whether ref is a commit SHA
            extracted_dir: Path to extracted repository (to copy files from)

        Returns:
            Path to the worktree
        """
        from git import GitCommandError

        # Generate worktree path
        base_path = Path(base_repo.working_tree_dir or base_repo.git_dir)
        worktrees_dir = base_path / "worktrees"
        worktree_name = ref.replace("/", "_").replace("\\", "_")
        worktree_path = worktrees_dir / worktree_name

        # Remove existing worktree if it exists
        if worktree_path.exists():
            try:
                logger.info(f"Removing existing worktree at {worktree_path}")
                base_repo.git.worktree("remove", str(worktree_path), force=True)
            except GitCommandError:
                # Worktree might not be registered, just remove directory
                shutil.rmtree(worktree_path, ignore_errors=True)

        # Create worktree directory
        worktrees_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Try to create worktree from existing ref
            if is_commit:
                # For commits, use detached HEAD
                base_repo.git.worktree("add", str(worktree_path), ref, "--detach")
            else:
                # For branches, try to checkout branch
                try:
                    base_repo.git.worktree("add", str(worktree_path), ref)
                except GitCommandError:
                    # Branch might not exist, create it from extracted_dir
                    # First, ensure the ref exists in the base repo
                    # Copy files from extracted_dir to worktree and commit
                    worktree_path.mkdir(parents=True, exist_ok=True)
                    # Copy files
                    for item in os.listdir(extracted_dir):
                        if item == ".git":
                            continue
                        src = os.path.join(extracted_dir, item)
                        dst = worktree_path / item
                        if os.path.isdir(src):
                            shutil.copytree(src, dst, dirs_exist_ok=True)
                        else:
                            shutil.copy2(src, dst)

                    # Initialize worktree as new repo and add as worktree
                    worktree_repo = Repo.init(worktree_path)
                    worktree_repo.git.add(A=True)
                    try:
                        worktree_repo.index.commit(f"Initial commit for {ref}")
                    except Exception:
                        pass

                    # Add remote reference in base repo if needed
                    # For now, we'll just use the worktree directly
                    logger.info(
                        f"Created worktree directory at {worktree_path} with copied files"
                    )
        except GitCommandError as e:
            logger.warning(f"Could not create worktree using git command: {e}")
            # Fallback: create directory and copy files
            if not worktree_path.exists():
                worktree_path.mkdir(parents=True, exist_ok=True)

            # Copy files from extracted_dir
            for item in os.listdir(extracted_dir):
                if item == ".git":
                    continue
                src = os.path.join(extracted_dir, item)
                dst = worktree_path / item
                if os.path.isdir(src):
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)

            logger.info(f"Created worktree at {worktree_path} by copying files")

        return worktree_path

    def extract_repository_metadata(repo):
        if isinstance(repo, Repo):
            metadata = ParseHelper.extract_local_repo_metadata(repo)
        else:
            metadata = ParseHelper.extract_remote_repo_metadata(repo)
        return metadata

    def extract_local_repo_metadata(repo):
        languages = ParseHelper.get_local_repo_languages(repo.working_tree_dir)
        total_bytes = sum(languages.values())

        metadata = {
            "basic_info": {
                "full_name": os.path.basename(repo.working_tree_dir),
                "description": None,
                "created_at": None,
                "updated_at": None,
                "default_branch": repo.head.ref.name,
            },
            "metrics": {
                "size": ParseHelper.get_directory_size(repo.working_tree_dir),
                "stars": None,
                "forks": None,
                "watchers": None,
                "open_issues": None,
            },
            "languages": {
                "breakdown": languages,
                "total_bytes": total_bytes,
            },
            "commit_info": {"total_commits": len(list(repo.iter_commits()))},
            "contributors": {
                "count": len(list(repo.iter_commits("--all"))),
            },
            "topics": [],
        }

        return metadata

    def get_local_repo_languages(path):
        total_bytes = 0
        python_bytes = 0

        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                file_extension = os.path.splitext(filename)[1]
                file_path = os.path.join(dirpath, filename)
                file_size = os.path.getsize(file_path)
                total_bytes += file_size
                if file_extension == ".py":
                    python_bytes += file_size

        languages = {}
        if total_bytes > 0:
            languages["Python"] = python_bytes
            languages["Other"] = total_bytes - python_bytes

        return languages

    def extract_remote_repo_metadata(repo):
        languages = repo.get_languages()
        total_bytes = sum(languages.values())

        metadata = {
            "basic_info": {
                "full_name": repo.full_name,
                "description": repo.description,
                "created_at": repo.created_at.isoformat(),
                "updated_at": repo.updated_at.isoformat(),
                "default_branch": repo.default_branch,
            },
            "metrics": {
                "size": repo.size,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "watchers": repo.watchers_count,
                "open_issues": repo.open_issues_count,
            },
            "languages": {
                "breakdown": languages,
                "total_bytes": total_bytes,
            },
            "commit_info": {"total_commits": repo.get_commits().totalCount},
            "contributors": {
                "count": repo.get_contributors().totalCount,
            },
            "topics": repo.get_topics(),
        }

        return metadata

    async def check_commit_status(
        self, project_id: str, requested_commit_id: str = None
    ) -> bool:
        """
        Check if the current commit ID of the project matches the latest commit ID from the repository.

        Args:
            project_id (str): The ID of the project to check.
            requested_commit_id (str, optional): The commit ID from the current parse request.
                If provided, indicates this is a pinned commit parse (not branch-based).
        Returns:
            bool: True if the commit IDs match or if this is a pinned commit parse, False otherwise.
        """
        logger.info(
            f"check_commit_status: Checking commit status for project {project_id}, "
            f"requested_commit_id={requested_commit_id}"
        )

        project = await self.project_manager.get_project_from_db_by_id(project_id)
        if not project:
            logger.error(f"Project with ID {project_id} not found")
            return False

        current_commit_id = project.get("commit_id")
        repo_name = project.get("project_name")
        branch_name = project.get("branch_name")

        logger.info(
            f"check_commit_status: Project {project_id} - repo={repo_name}, "
            f"branch={branch_name}, current_commit_id={current_commit_id}"
        )

        # Check if this is a pinned commit parse
        # If the user explicitly provided a commit_id in the parse request,
        # this is a pinned commit parse (not branch-based)
        if requested_commit_id is not None:
            logger.info(
                f"check_commit_status: Pinned commit parse detected "
                f"(requested_commit_id={requested_commit_id})"
            )
            # For pinned commits, check if the requested commit matches the stored commit
            if requested_commit_id == current_commit_id:
                logger.info(
                    f"check_commit_status: Pinned commit {requested_commit_id} matches "
                    f"stored commit, no reparse needed"
                )
                return True
            else:
                logger.info(
                    f"check_commit_status: Pinned commit changed from {current_commit_id} "
                    f"to {requested_commit_id}, reparse needed"
                )
                return False

        # If we reach here, this is a branch-based parse (not pinned commit)
        # We need to compare the stored commit with the latest branch commit

        if not repo_name:
            logger.error(
                f"Repository name or branch name not found for project ID {project_id}"
            )
            return False

        if not branch_name:
            logger.error(
                f"Branch is empty so sticking to commit and not updating it for: {project_id}"
            )
            return True

        if len(repo_name.split("/")) < 2:
            # Local repo, always parse local repos
            logger.info("check_commit_status: Local repo detected, forcing reparse")
            return False

        try:
            logger.info(
                f"check_commit_status: Branch-based parse - getting repo info for {repo_name}"
            )
            _github, repo = self.github_service.get_repo(repo_name)

            # If current_commit_id is None, we should reparse
            if current_commit_id is None:
                logger.info(
                    f"check_commit_status: Project {project_id} has no commit_id, will reparse"
                )
                return False

            # Get the latest commit from the branch
            logger.info(
                f"check_commit_status: Getting latest commit from branch {branch_name}"
            )
            branch = repo.get_branch(branch_name)
            latest_commit_id = branch.commit.sha

            # Compare current commit with latest commit
            is_up_to_date = current_commit_id == latest_commit_id
            logger.info(
                f"check_commit_status: Project {project_id} commit status for branch {branch_name}: "
                f"{'Up to date' if is_up_to_date else 'Outdated'} - "
                f"Current: {current_commit_id}, Latest: {latest_commit_id}"
            )

            return is_up_to_date
        except Exception as e:
            logger.error(
                f"check_commit_status: Error fetching latest commit for {repo_name}/{branch_name}: {e}",
                exc_info=True,
            )
            return False

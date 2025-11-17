import json
import logging
import os
import shutil
import tarfile
import uuid
from typing import Any, Tuple
from urllib.parse import urlparse, urlunparse

import chardet
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

    def is_allowed_for_gitbucket(self, file_path):
        """
        Check if file extension is allowed for GitBucket private repo parsing.
        Only specific file types are supported to reduce processing overhead.

        Allowed extensions: cs, c, cpp, h, xml, json, sql, js, ts, py, css
        """
        allowed_extensions = [
            "cs",    # C#
            "c",     # C
            "cpp",   # C++
            "cxx",   # C++ alternative
            "cc",    # C++ alternative
            "h",     # C/C++ headers
            "hpp",   # C++ headers
            "json",  # JSON
            "sql",   # SQL
            "js",    # JavaScript
            "jsx",   # React JSX
            "ts",    # TypeScript
            "tsx",   # React TSX
            "py",    # Python
            "css",  # CSS
            "md",    # Markdown
            "txt",   # Text
            "yaml"   # YAML
        ]

        ext = file_path.split(".")[-1].lower()
        return ext in allowed_extensions

    def is_text_file(self, file_path):
        """
        Determine if a file is a text file by checking extension and content.
        Uses chardet for automatic encoding detection to handle all codecs.
        """

        def check_text_content(file_path):
            """
            Check if file content is text using chardet for encoding detection.
            Also validates if content looks like text (printable characters).
            """
            try:
                # Read a sample of the file (8KB for better detection)
                with open(file_path, "rb") as f:
                    sample = f.read(8192)

                if not sample:
                    # Empty files are considered text
                    return True

                # Use chardet to detect encoding
                detection = chardet.detect(sample)
                encoding = detection.get("encoding")
                confidence = detection.get("confidence", 0)

                # If chardet can't detect with reasonable confidence, likely binary
                if not encoding or confidence < 0.5:
                    logger.debug(
                        f"ParseHelper: File {file_path} has low encoding confidence "
                        f"({confidence:.2%}), likely binary"
                    )
                    return False

                # Try to decode with detected encoding
                try:
                    content = sample.decode(encoding)
                except (UnicodeDecodeError, LookupError):
                    logger.debug(
                        f"ParseHelper: Failed to decode {file_path} with detected encoding {encoding}"
                    )
                    return False

                # Calculate printable character ratio
                printable_count = sum(c.isprintable() or c.isspace() for c in content)
                printable_ratio = printable_count / len(content) if content else 0

                # If less than 70% printable, likely binary
                if printable_ratio < 0.70:
                    logger.debug(
                        f"ParseHelper: File {file_path} is likely binary "
                        f"(printable ratio: {printable_ratio:.2%}, encoding: {encoding})"
                    )
                    return False

                logger.debug(
                    f"ParseHelper: Validated {file_path} as text "
                    f"(encoding: {encoding}, confidence: {confidence:.2%}, "
                    f"printable ratio: {printable_ratio:.2%})"
                )
                return True

            except FileNotFoundError:
                logger.warning(f"ParseHelper: File not found during text check: {file_path}")
                return False
            except PermissionError:
                logger.warning(f"ParseHelper: Permission denied during text check: {file_path}")
                return False
            except Exception as e:
                logger.error(
                    f"ParseHelper: Unexpected error checking if file is text {file_path}: {e}"
                )
                return False

        ext = file_path.split(".")[-1].lower()

        # Binary/executable extensions that should NEVER be parsed as text
        exclude_extensions = [
            # Images
            "png",
            "jpg",
            "jpeg",
            "gif",
            "bmp",
            "tiff",
            "webp",
            "ico",
            "svg",
            # Videos
            "mp4",
            "avi",
            "mov",
            "wmv",
            "flv",
            "webm",
            "mkv",
            # Archives
            "zip",
            "tar",
            "gz",
            "bz2",
            "7z",
            "rar",
            "xz",
            # Executables and libraries
            "exe",
            "dll",
            "so",
            "dylib",
            "lib",
            "a",
            "o",
            "obj",
            # Cryptographic/binary formats
            "snk",
            "pfx",
            "cer",
            "der",
            "p12",
            "key",
            "crt",
            "pem",
            # Audio
            "wav",
            "mp3",
            "ogg",
            "flac",
            "aac",
            "wma",
            # Databases
            "db",
            "sqlite",
            "sqlite3",
            "mdb",
            # Jupyter notebooks (should be parsed differently)
            "ipynb",
            # Other binary
            "bin",
            "dat",
            "pyc",
            "pyo",
            "class",
            "jar",
            "war",
            # Fonts
            "ttf",
            "otf",
            "woff",
            "woff2",
            "eot",
            # PDF and Office
            "pdf",
            "doc",
            "docx",
            "xls",
            "xlsx",
            "ppt",
            "pptx",
        ]

        include_extensions = [
            "py",
            "js",
            "ts",
            "tsx",
            "jsx",
            "c",
            "cs",
            "cpp",
            "cxx",
            "cc",
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
            "mdx",
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
            "htm",
            "css",
            "scss",
            "sass",
            "less",
            "sh",
            "bash",
            "zsh",
            "fish",
            "ps1",
            "psm1",
            "bat",
            "cmd",
            "xsq",
            "proto",
            "sql",
            "r",
            "R",
            "scala",
            "kt",
            "swift",
            "m",
            "vue",
            "svelte",
        ]

        if ext in exclude_extensions:
            return False
        elif ext in include_extensions or check_text_content(file_path):
            return True
        else:
            return False

    async def download_and_extract_tarball(
        self, repo, branch, target_dir, auth, repo_details, user_id
    ):
        # Ensure target_dir is an absolute path to avoid path issues in celery workers
        target_dir = os.path.abspath(target_dir)
        logger.info(f"ParsingHelper: Using absolute target_dir: {target_dir}")

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
                raise ValueError(f"Expected string URL, got {type(tarball_url)}: {tarball_url}")

            # For GitBucket private repos, use PyGithub client's requester for authenticated requests
            # According to GitBucket API docs: https://github.com/gitbucket/gitbucket/wiki/API-WebHook
            # Authentication: "Authorization: token YOUR_TOKEN" in header
            provider_type = os.getenv("CODE_PROVIDER", "github").lower()

            if provider_type == "gitbucket" and hasattr(repo, "_provider") and repo._provider:
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
                                raise requests.exceptions.HTTPError("401 Unauthorized from session")
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

                                if hasattr(requester, "auth") and hasattr(requester.auth, "token"):
                                    token = requester.auth.token
                                elif hasattr(requester, "_Requester__authorizationHeader"):
                                    auth_header = requester._Requester__authorizationHeader
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
                        error_msg = (
                            "No authentication token available for GitBucket archive download"
                        )
                        logger.error(f"ParsingHelper: {error_msg}")
                        raise ValueError(error_msg)
                    
                    # GitBucket web endpoints (archive downloads) may require Basic Auth
                    # Try token header format first (API standard per GitBucket docs)
                    headers = {"Authorization": f"token {token}"}
                    logger.debug("ParsingHelper: Attempting archive download with token header")

                    response = requests.get(tarball_url, stream=True, headers=headers, timeout=30)

                    # If token header fails with 401, try Basic Auth with repo owner username
                    # GitBucket web endpoints sometimes require Basic Auth (supported since v4.3)
                    if response.status_code == 401:
                        logger.debug("ParsingHelper: Token header auth failed, trying Basic Auth")
                        response.close()
                        
                        # Try Basic Auth with repo owner username and token as password
                        if hasattr(repo, "owner") and hasattr(repo.owner, "login"):
                            username = repo.owner.login
                            basic_auth = requests.auth.HTTPBasicAuth(username, token)
                            response = requests.get(
                                tarball_url, stream=True, auth=basic_auth, timeout=30
                            )
                            logger.debug(f"ParsingHelper: Basic Auth response status: {response.status_code}")
            else:
                # For GitHub and other providers, use standard token auth
                headers = {}
                if auth:
                    headers = {"Authorization": f"token {auth.token}"}
                response = requests.get(tarball_url, stream=True, headers=headers, timeout=30)

            response.raise_for_status()

        except requests.exceptions.HTTPError as e:
            # If we get 401, archive download might not be supported for private repos
            # Fall back to git clone for GitBucket
            status_code = None
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
            elif '401' in str(e):
                status_code = 401

            if status_code == 401:
                provider_type = os.getenv("CODE_PROVIDER", "github").lower()
                if provider_type == "gitbucket":
                    logger.info(
                        "ParsingHelper: Archive download failed with 401 for GitBucket private repo, "
                        "falling back to git clone"
                    )
                    return await self._clone_repository_with_auth(repo, branch, target_dir, user_id)

            logger.error(f"ParsingHelper: Failed to download repository archive: {e}")
            raise ParsingFailedError("Failed to download repository archive") from e
        except requests.exceptions.RequestException as e:
            logger.error(f"ParsingHelper: Error fetching tarball: {e}")
            raise ParsingFailedError("Failed to download repository archive") from e
        except Exception as e:
            logger.exception("ParsingHelper: Unexpected error in tarball download")
            raise ParsingFailedError("Unexpected error during repository download") from e
        # Include user_id in tarball path to prevent collisions between concurrent downloads
        # Even with locking, this provides additional safety
        tarball_path = os.path.join(
            target_dir,
            f"{repo.full_name.replace('/', '-').replace('.', '-')}-{branch.replace('/', '-').replace('.', '-')}-{user_id}.tar.gz",
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
            
            # Check if file exists and get size (handle race conditions)
            if not os.path.exists(tarball_path):
                error_msg = f"Tarball file was not created at {tarball_path}. Download may have failed or file was deleted by another process."
                logger.error(f"ParsingHelper: {error_msg}")
                raise ParsingFailedError(error_msg)
            
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
                logger.debug(f"ParsingHelper: Opening tarball file: {tarball_path}")
                with tarfile.open(tarball_path, "r:gz") as tar:
                    # Validate that the tarball is not empty
                    members = tar.getmembers()
                    logger.debug(f"ParsingHelper: Tarball contains {len(members)} members")
                    if not members:
                        error_msg = "Tarball contains no files"
                        logger.error(f"ParsingHelper: {error_msg}")
                        raise ParsingFailedError(error_msg)
                    
                    temp_dir = os.path.join(final_dir, "temp_extract")
                    logger.debug(f"ParsingHelper: Creating temp directory: {temp_dir}")
                    os.makedirs(temp_dir, exist_ok=True)
                    logger.debug(f"ParsingHelper: Extracting {len(members)} members to {temp_dir}")
                    tar.extractall(path=temp_dir)
                    logger.info(f"ParsingHelper: Extracted tarball contents to {temp_dir}")

                    # Check if extraction directory has contents
                    extracted_contents = os.listdir(temp_dir)
                    logger.debug(f"ParsingHelper: Temp directory contains {len(extracted_contents)} items: {extracted_contents[:5] if len(extracted_contents) > 5 else extracted_contents}")
                    if not extracted_contents:
                        error_msg = (
                            "Tarball extraction resulted in empty directory. "
                            "The archive may be corrupted or the repository may be empty."
                        )
                        logger.error(f"ParsingHelper: {error_msg}")
                        raise ParsingFailedError(error_msg)

                    extracted_dir = os.path.join(temp_dir, extracted_contents[0])
                    logger.info(f"ParsingHelper: Main extracted directory: {extracted_dir}")

                    text_files_count = 0
                    skipped_files_count = 0
                    error_files_count = 0

                    # Check if this is a GitBucket private repo to apply extension filter
                    provider_type = os.getenv("CODE_PROVIDER", "github").lower()
                    is_gitbucket_private = provider_type == "gitbucket"
                    if is_gitbucket_private:
                        logger.info("ParsingHelper: GitBucket private repo detected - applying extension filter (cs, c, cpp, h, xml, json, sql, js, ts, py, css)")

                    for root, dirs, files in os.walk(extracted_dir):
                        logger.debug(f"ParsingHelper: Processing directory: {root} with {len(files)} files")
                        for file in files:
                            if file.startswith("."):
                                logger.debug(f"ParsingHelper: Skipping hidden file: {file}")
                                skipped_files_count += 1
                                continue
                            file_path = os.path.join(root, file)

                            # For GitBucket private repos, only allow specific extensions
                            if is_gitbucket_private and not self.is_allowed_for_gitbucket(file_path):
                                logger.debug(f"ParsingHelper: Skipping non-allowed extension for GitBucket: {file_path}")
                                skipped_files_count += 1
                                continue

                            logger.debug(f"ParsingHelper: Checking if file is text: {file_path}")

                            try:
                                is_text = self.is_text_file(file_path)
                                logger.debug(f"ParsingHelper: File {file_path} is_text={is_text}")
                            except Exception as check_error:
                                logger.error(f"ParsingHelper: Error checking if file is text {file_path}: {check_error}")
                                error_files_count += 1
                                is_text = False
                            
                            if is_text:
                                try:
                                    relative_path = os.path.relpath(file_path, extracted_dir)
                                    dest_path = os.path.join(final_dir, relative_path)
                                    logger.debug(f"ParsingHelper: Copying text file from {file_path} to {dest_path}")
                                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                                    shutil.copy2(file_path, dest_path)
                                    text_files_count += 1
                                    if text_files_count % 100 == 0:
                                        logger.info(f"ParsingHelper: Copied {text_files_count} text files so far...")
                                except (shutil.Error, OSError) as e:
                                    logger.error(
                                        f"ParsingHelper: Error copying file {file_path}: {e}"
                                    )
                                    error_files_count += 1
                            else:
                                skipped_files_count += 1

                    logger.info(
                        f"ParsingHelper: File processing complete - Copied {text_files_count} text files, "
                        f"skipped {skipped_files_count} files, encountered {error_files_count} errors"
                    )
                    # Remove the temporary directory
                    logger.debug(f"ParsingHelper: Removing temporary directory: {temp_dir}")
                    try:
                        shutil.rmtree(temp_dir)
                        logger.debug(f"ParsingHelper: Successfully removed temporary directory")
                    except OSError as e:
                        logger.error(f"ParsingHelper: Error removing temporary directory: {e}")
                        pass
            except tarfile.TarError as e:
                error_msg = (
                    f"Failed to extract tarball: {e}. The archive may be corrupted or invalid."
                )
                logger.error(f"ParsingHelper: {error_msg}")
                logger.error(f"ParsingHelper: Tarball path: {tarball_path}, size: {os.path.getsize(tarball_path) if os.path.exists(tarball_path) else 'N/A'}")
                raise ParsingFailedError(error_msg) from e

        except (IOError, OSError, tarfile.TarError, shutil.Error) as e:
            # OSError includes FileNotFoundError which can occur if tarball is deleted by another process
            logger.error(f"Error handling tarball: {e}")
            if isinstance(e, FileNotFoundError):
                error_msg = f"Tarball file not found: {tarball_path}. This may indicate a race condition with concurrent downloads or a failed download."
                logger.error(f"ParsingHelper: {error_msg}")
                raise ParsingFailedError(error_msg) from e
            raise ParsingFailedError("Failed to process repository archive") from e
        finally:
            if os.path.exists(tarball_path):
                os.remove(tarball_path)

        return final_dir

    async def _clone_repository_with_auth(
        self, repo, branch, target_dir, user_id
    ):
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

        logger.info(f"ParsingHelper: Cloning repository '{repo_name}' branch '{branch}' using git")

        # Ensure target_dir is an absolute path to avoid path issues in celery workers
        target_dir = os.path.abspath(target_dir)
        logger.info(f"ParsingHelper: Using absolute target_dir: {target_dir}")

        final_dir = os.path.join(
            target_dir,
            f"{repo.full_name.replace('/', '-').replace('.', '-')}-{branch.replace('/', '-').replace('.', '-')}-{user_id}",
        )
        logger.info(f"ParsingHelper: Final clone directory will be: {final_dir}")

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
        repo_path = f"{base_path}/{repo.full_name}.git" if base_path else f"/{repo.full_name}.git"

        clone_url_with_auth = urlunparse((
            parsed.scheme,
            f"{username}:{password}@{parsed.netloc}",
            repo_path,
            "",
            "",
            ""
        ))

        # Log URL without credentials for security
        safe_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            repo_path,
            "",
            "",
            ""
        ))
        logger.info(f"ParsingHelper: Cloning from {safe_url}")

        try:
            # Clone the repository to temporary directory with shallow clone for faster download
            repo_obj = Repo.clone_from(clone_url_with_auth, temp_clone_dir, branch=branch, depth=1)
            logger.info(f"ParsingHelper: Successfully cloned repository to temporary directory: {temp_clone_dir}")

            # Filter and copy only text files to final directory
            logger.info(f"ParsingHelper: Filtering text files from clone to {final_dir}")
            os.makedirs(final_dir, exist_ok=True)
            logger.debug(f"ParsingHelper: Created final directory: {final_dir}")

            text_files_count = 0
            skipped_files_count = 0
            error_files_count = 0

            # Check if this is a GitBucket private repo to apply extension filter
            provider_type = os.getenv("CODE_PROVIDER", "github").lower()
            is_gitbucket_private = provider_type == "gitbucket"
            if is_gitbucket_private:
                logger.info("ParsingHelper: GitBucket private repo detected - applying extension filter (cs, c, cpp, h, xml, json, sql, js, ts, py, css)")

            logger.debug(f"ParsingHelper: Starting to walk through cloned directory: {temp_clone_dir}")
            for root, dirs, files in os.walk(temp_clone_dir):
                # Skip .git directory
                if '.git' in root.split(os.sep):
                    logger.debug(f"ParsingHelper: Skipping .git directory: {root}")
                    continue

                # Skip hidden directories
                if any(part.startswith('.') for part in root.split(os.sep)):
                    logger.debug(f"ParsingHelper: Skipping hidden directory: {root}")
                    continue

                logger.debug(f"ParsingHelper: Processing directory: {root} with {len(files)} files")

                for file in files:
                    # Skip hidden files
                    if file.startswith('.'):
                        logger.debug(f"ParsingHelper: Skipping hidden file: {file}")
                        skipped_files_count += 1
                        continue

                    file_path = os.path.join(root, file)

                    # For GitBucket private repos, only allow specific extensions
                    if is_gitbucket_private and not self.is_allowed_for_gitbucket(file_path):
                        logger.debug(f"ParsingHelper: Skipping non-allowed extension for GitBucket: {file_path}")
                        skipped_files_count += 1
                        continue

                    logger.debug(f"ParsingHelper: Checking if file is text: {file_path}")

                    # Filter using is_text_file check
                    try:
                        is_text = self.is_text_file(file_path)
                        logger.debug(f"ParsingHelper: File {file_path} is_text={is_text}")
                    except Exception as check_error:
                        logger.error(f"ParsingHelper: Error checking if file is text {file_path}: {check_error}")
                        error_files_count += 1
                        is_text = False
                    
                    if is_text:
                        try:
                            # Calculate relative path from clone root
                            relative_path = os.path.relpath(file_path, temp_clone_dir)
                            dest_path = os.path.join(final_dir, relative_path)
                            logger.debug(f"ParsingHelper: Copying text file from {file_path} to {dest_path}")

                            # Create destination directory structure
                            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                            # Copy text file to final directory
                            shutil.copy2(file_path, dest_path)
                            text_files_count += 1
                            
                            if text_files_count % 100 == 0:
                                logger.info(f"ParsingHelper: Copied {text_files_count} text files from git clone so far...")
                        except (shutil.Error, OSError) as e:
                            logger.error(f"ParsingHelper: Error copying file {file_path}: {e}")
                            error_files_count += 1
                    else:
                        skipped_files_count += 1

            logger.info(
                f"ParsingHelper: Git clone file processing complete - Copied {text_files_count} text files, "
                f"skipped {skipped_files_count} files, encountered {error_files_count} errors"
            )

            # Clean up temporary clone directory
            try:
                shutil.rmtree(temp_clone_dir)
                logger.info(f"ParsingHelper: Cleaned up temporary clone directory: {temp_clone_dir}")
            except Exception as e:
                logger.warning(f"ParsingHelper: Failed to clean up temp clone directory: {e}")

            logger.info(f"ParsingHelper: Git clone completed successfully, returning final_dir: {final_dir}")
            logger.info(f"ParsingHelper: Verifying final_dir exists: {os.path.exists(final_dir)}")
            if os.path.exists(final_dir):
                logger.info(f"ParsingHelper: final_dir contains {len(os.listdir(final_dir))} items")

            return final_dir

        except GitCommandError as e:            # Exit code 128 typically indicates authentication/permission issues
            exit_code = getattr(e, 'status', None) or getattr(e, 'returncode', None)
            error_msg = str(e)
            
            logger.error(f"ParsingHelper: Git clone failed with exit code {exit_code}: {error_msg}")
            
            # Provide specific error messages based on exit code
            if exit_code == 128:
                # Common causes: authentication failure, permission denied, invalid URL
                diagnostic_msg = (
                    f"Git clone failed with exit code 128 (authentication/permission error). "
                    f"Possible causes:\n"
                    f"  1. Invalid credentials (username/password)\n"
                    f"  2. Repository not found or inaccessible\n"
                    f"  3. URL format issue (check CODE_PROVIDER_BASE_URL)\n"
                    f"  4. Network connectivity issue\n"
                    f"Repository: {repo.full_name}, Branch: {branch}\n"
                    f"Base URL: {base_url}, Repo path: {repo_path}\n"
                    f"Username: {username if username else 'NOT SET'}"
                )
                logger.error(f"ParsingHelper: {diagnostic_msg}")
                
                # Check if credentials are set
                if not username or not password:
                    raise ParsingFailedError(
                        "GitBucket credentials not configured. "
                        "Set GITBUCKET_USERNAME and GITBUCKET_PASSWORD environment variables."
                    ) from e
                
                # Check URL format (construct safe URL for logging)
                safe_url_check = urlunparse((
                    parsed.scheme,
                    parsed.netloc,
                    repo_path,
                    "", "", ""
                ))
                if not clone_url_with_auth.startswith(('http://', 'https://')):
                    raise ParsingFailedError(
                        f"Invalid clone URL format: {safe_url_check}. "
                        f"Check CODE_PROVIDER_BASE_URL configuration."
                    ) from e
                
                raise ParsingFailedError(
                    f"Git clone authentication failed. Verify credentials and repository access. "
                    f"Error: {error_msg}"
                ) from e
            else:
                # Other git errors
                raise ParsingFailedError(f"Failed to clone repository (exit code {exit_code}): {error_msg}") from e
        except Exception as e:
            logger.error(f"ParsingHelper: Unexpected error during git clone: {e}")
            # Clean up temp directory on error
            if os.path.exists(temp_clone_dir):
                try:
                    shutil.rmtree(temp_clone_dir)
                except Exception:
                    pass
            raise ParsingFailedError(f"Unexpected error during repository clone: {e}") from e

    @staticmethod
    def read_file_with_encoding(file_path, max_size=None):
        """
        Read a file with automatic encoding detection using chardet.
        This handles all codecs properly (UTF-8, Windows-1252, Shift-JIS, GB2312, etc.)

        Args:
            file_path: Path to the file to read
            max_size: Maximum number of bytes to read (None = read entire file)

        Returns:
            tuple: (content, encoding_used) where content is the decoded string

        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file can't be accessed
            UnicodeDecodeError: If file can't be decoded with detected encoding
            Exception: For other errors
        """
        try:
            # Read file as bytes for encoding detection
            with open(file_path, "rb") as f:
                raw_bytes = f.read(max_size) if max_size else f.read()

            if not raw_bytes:
                # Empty file
                return "", "utf-8"

            # Try UTF-8 first (most common for source code)
            try:
                content = raw_bytes.decode("utf-8")
                return content, "utf-8"
            except UnicodeDecodeError:
                # Not UTF-8, continue with chardet detection
                pass

            # Detect encoding using chardet
            detection = chardet.detect(raw_bytes)
            encoding = detection.get("encoding")
            confidence = detection.get("confidence", 0)

            # If chardet can't detect with reasonable confidence, try common encodings
            if not encoding or confidence < 0.5:
                logger.debug(
                    f"ParseHelper: Low encoding confidence ({confidence:.2%}) for {file_path}, "
                    f"trying fallback encodings"
                )
                # Try common encodings as fallback
                fallback_encodings = ["utf-8", "cp1252", "latin-1"]
                for fallback_encoding in fallback_encodings:
                    try:
                        content = raw_bytes.decode(fallback_encoding)
                        logger.debug(
                            f"ParseHelper: Successfully decoded {file_path} with {fallback_encoding}"
                        )
                        return content, fallback_encoding
                    except (UnicodeDecodeError, LookupError):
                        continue
                # If all fallbacks fail, raise an error
                raise UnicodeDecodeError(
                    "unknown",
                    raw_bytes,
                    0,
                    len(raw_bytes),
                    f"Could not decode file with any encoding (chardet confidence: {confidence:.2%})",
                )

            # Try to decode with detected encoding
            try:
                content = raw_bytes.decode(encoding)
                logger.debug(
                    f"ParseHelper: Read {file_path} with {encoding} encoding "
                    f"(confidence: {confidence:.2%})"
                )
                return content, encoding
            except (UnicodeDecodeError, LookupError) as e:
                # Chardet detection failed, try fallback encodings
                logger.debug(
                    f"ParseHelper: Failed to decode {file_path} with detected encoding {encoding} "
                    f"(confidence: {confidence:.2%}): {e}. Trying fallback encodings."
                )
                fallback_encodings = ["utf-8", "cp1252", "latin-1"]
                for fallback_encoding in fallback_encodings:
                    try:
                        content = raw_bytes.decode(fallback_encoding)
                        logger.debug(
                            f"ParseHelper: Successfully decoded {file_path} with fallback {fallback_encoding}"
                        )
                        return content, fallback_encoding
                    except (UnicodeDecodeError, LookupError):
                        continue
                # If all fallbacks fail, raise the original error
                logger.warning(
                    f"ParseHelper: Failed to decode {file_path} with detected encoding {encoding} "
                    f"and all fallback encodings"
                )
                raise

        except FileNotFoundError:
            logger.error(f"ParseHelper: File not found: {file_path}")
            raise
        except PermissionError:
            logger.error(f"ParseHelper: Permission denied: {file_path}")
            raise
        except Exception as e:
            logger.error(f"ParseHelper: Error reading file {file_path}: {e}")
            raise

    @staticmethod
    def detect_repo_language(repo_dir):
        logger.info(f"ParseHelper: detect_repo_language called for directory: {repo_dir}")
        
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
        total_files_scanned = 0
        encoding_errors = []
        skipped_directories = []

        try:
            logger.info(f"ParseHelper: Starting directory walk for {repo_dir}")
            for root, _, files in os.walk(repo_dir):
                # Skip hidden directories
                if any(part.startswith(".") for part in root.split(os.sep)):
                    skipped_directories.append(root)
                    continue

                for file in files:
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file)[1].lower()

                    try:
                        # Use the new helper that handles all encodings with chardet
                        content, encoding_used = ParseHelper.read_file_with_encoding(file_path)
                        total_chars += len(content)
                        total_files_scanned += 1

                        # Log when non-UTF-8 encoding was needed (info level for visibility)
                        if encoding_used.lower() not in ["utf-8", "ascii"]:
                            logger.info(
                                f"ParseHelper: Read {file_path} using {encoding_used} encoding"
                            )

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
                    except (
                        UnicodeDecodeError,
                        FileNotFoundError,
                        PermissionError,
                    ) as e:
                        encoding_errors.append({
                            "file": file_path,
                            "error": str(e),
                            "type": type(e).__name__
                        })
                        logger.warning(f"ParseHelper: Error reading file {file_path}: {e}")
                        continue
                    except Exception as e:
                        encoding_errors.append({
                            "file": file_path,
                            "error": str(e),
                            "type": type(e).__name__
                        })
                        logger.warning(
                            f"ParseHelper: Unexpected error reading file {file_path}: {e}"
                        )
                        continue
        except (TypeError, FileNotFoundError, PermissionError) as e:
            logger.error(f"ParseHelper: Error accessing directory '{repo_dir}': {e}")
            logger.exception("ParseHelper: Directory access exception details:")
        except Exception as e:
            logger.error(f"ParseHelper: Unexpected error during language detection: {e}")
            logger.exception("ParseHelper: Language detection exception details:")

        # Log summary statistics
        logger.info(f"ParseHelper: Language detection complete - scanned {total_files_scanned} files, {total_chars} total characters")
        logger.info(f"ParseHelper: Language counts: {lang_count}")
        logger.info(f"ParseHelper: Skipped {len(skipped_directories)} hidden directories")
        
        if encoding_errors:
            logger.warning(f"ParseHelper: Encountered {len(encoding_errors)} encoding/file errors during scan")
            logger.debug(f"ParseHelper: First 10 encoding errors: {encoding_errors[:10]}")
            
            # Log summary by error type
            error_types = {}
            for error in encoding_errors:
                error_type = error.get("type", "Unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1
            logger.info(f"ParseHelper: Encoding error summary by type: {error_types}")

        # Determine the predominant language based on counts
        # Exclude 'other' from consideration - only use it if no actual programming language is found
        programming_languages = {k: v for k, v in lang_count.items() if k != "other"}

        if programming_languages:
            # If we have any actual programming language files, use the most common one
            predominant_language = max(programming_languages, key=programming_languages.get)
            result_language = predominant_language if programming_languages[predominant_language] > 0 else "other"
            logger.info(
                f"ParseHelper: Detected predominant language: {result_language} "
                f"(count: {lang_count.get(result_language, 0)}), "
                f"programming language files found: {sum(programming_languages.values())}, "
                f"other files: {lang_count.get('other', 0)}"
            )
        else:
            # No programming language files found, default to 'other'
            result_language = "other"
            logger.info(
                f"ParseHelper: No supported programming language files found, "
                f"defaulting to 'other' (total files: {lang_count.get('other', 0)})"
            )

        return result_language

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
            full_name = repo_path.split("/")[-1]  # Extract just the directory name for display
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
            
            # Verify the path exists and is accessible
            if not os.path.exists(repo_path):
                error_msg = f"Local repository path does not exist: {repo_path}"
                logger.error(f"ParsingHelper: {error_msg}")
                raise FileNotFoundError(error_msg)
            
            if not os.path.isdir(repo_path):
                error_msg = f"Local repository path is not a directory: {repo_path}"
                logger.error(f"ParsingHelper: {error_msg}")
                raise NotADirectoryError(error_msg)
            
            try:
                contents = os.listdir(repo_path)
                logger.info(f"ParsingHelper: Local repository contains {len(contents)} items")
                logger.debug(f"ParsingHelper: Repository contents (first 20): {contents[:20]}")
            except Exception as e:
                logger.error(f"ParsingHelper: Cannot list local repository contents: {e}")
                raise
            
            logger.info(f"ParsingHelper: Validated local repository is accessible")
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
                logger.error(f"Error checking out {'commit' if commit_id else 'branch'}: {e}")
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
                    logger.info(f"ParsingHelper: Tarball extracted to: {extracted_dir}")
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
                    logger.info(f"ParsingHelper: Tarball extracted to: {extracted_dir}")
                    logger.info(f"ParsingHelper: Tarball extracted to: {extracted_dir}")
                    # Use repo.get_branch() instead of repo_details.get_branch()
                    # repo is the MockRepo object (or PyGithub Repository) with get_branch method
                    # repo_details can be ParsingRequest in dev mode, which doesn't have get_branch
                    branch_details = repo.get_branch(branch)
                    latest_commit_sha = branch_details.commit.sha
                
                # Validate extracted directory
                logger.info(f"ParsingHelper: Validating extracted directory: {extracted_dir}")
                if not extracted_dir:
                    error_msg = "Extraction returned None or empty path"
                    logger.error(f"ParsingHelper: {error_msg}")
                    raise ParsingFailedError(error_msg)
                
                if not os.path.exists(extracted_dir):
                    error_msg = f"Extracted directory does not exist: {extracted_dir}"
                    logger.error(f"ParsingHelper: {error_msg}")
                    raise ParsingFailedError(error_msg)
                
                if not os.path.isdir(extracted_dir):
                    error_msg = f"Extracted path is not a directory: {extracted_dir}"
                    logger.error(f"ParsingHelper: {error_msg}")
                    raise ParsingFailedError(error_msg)
                
                try:
                    contents = os.listdir(extracted_dir)
                    logger.info(f"ParsingHelper: Extracted directory contains {len(contents)} items")
                    logger.debug(f"ParsingHelper: Directory contents (first 20): {contents[:20]}")
                    
                    if len(contents) == 0:
                        error_msg = f"Extracted directory is empty: {extracted_dir}"
                        logger.error(f"ParsingHelper: {error_msg}")
                        raise ParsingFailedError(error_msg)
                except Exception as e:
                    logger.error(f"ParsingHelper: Cannot list extracted directory contents: {e}")
                    raise
                
                logger.info(f"ParsingHelper: Validated extracted directory is accessible and contains files")
                
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

        return extracted_dir, project_id

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

    async def check_commit_status(self, project_id: str, requested_commit_id: str = None) -> bool:
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
            logger.error(f"Repository name or branch name not found for project ID {project_id}")
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
            logger.info(f"check_commit_status: Getting latest commit from branch {branch_name}")
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

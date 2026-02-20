import asyncio
import json
import os
import shutil
import uuid
import subprocess
from typing import Any, Optional, Tuple
from urllib.parse import urlparse, urlunparse
from pathlib import Path
from collections import defaultdict


from fastapi import HTTPException
from git import GitCommandError, InvalidGitRepositoryError, Repo
from sqlalchemy.orm import Session

from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.parsing.graph_construction.parsing_schema import RepoDetails
from app.modules.parsing.utils.repo_name_normalizer import normalize_repo_name
from app.modules.projects.projects_schema import ProjectStatusEnum
from app.modules.projects.projects_service import ProjectService
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


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
        for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
            # # Skip symlinked directories
            # dirnames[:] = [
            #     d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))
            # ]

            for f in filenames:
                fp = os.path.join(dirpath, f)
                # Skip all symlinks
                if os.path.islink(fp):
                    continue
                total_size += os.path.getsize(fp)
        return total_size

    async def clone_or_copy_repository(
        self,
        repo_details: RepoDetails,
        user_id: str,
        *,
        auth_token: Optional[str] = None,
    ) -> Tuple[Any, Optional[str], Any, Optional[str]]:
        """
        Clone or copy repository, using RepoManager as primary source when enabled.

        Returns:
            Tuple of (repo, owner, auth, repo_manager_path)
            - repo: Git Repo object or PyGithub Repository object
            - owner: Repository owner login
            - auth: Authentication object for GitHub API
            - repo_manager_path: Path to repo in RepoManager if available, None otherwise
              When this is set, setup_project_directory should use this path directly
              and skip tarball download.
        """
        owner = None
        auth = None
        repo = None
        repo_manager_path = None  # New: path if repo is from/cloned to RepoManager

        logger.info(
            f"ParsingHelper: clone_or_copy_repository called for repo_name={repo_details.repo_name}, repo_path={repo_details.repo_path}"
        )

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
            # When RepoManager is enabled, it becomes the primary source of truth
            if self.repo_manager and repo_details.repo_name:
                logger.info(
                    f"ParsingHelper: RepoManager enabled, checking for existing repo: {repo_details.repo_name}"
                )

                # Check for exact match (same branch/commit)
                cached_repo_path = self.repo_manager.get_repo_path(
                    repo_name=repo_details.repo_name,
                    branch=repo_details.branch_name,
                    commit_id=repo_details.commit_id,
                    user_id=user_id,
                )

                if cached_repo_path and os.path.exists(cached_repo_path):
                    # Verify this is the correct worktree for the commit_id
                    actual_commit_id = None
                    try:
                        worktree_repo = Repo(cached_repo_path)
                        actual_commit_id = worktree_repo.head.commit.hexsha
                        logger.info(
                            f"ParsingHelper: Verified worktree commit: requested={repo_details.commit_id}, "
                            f"actual={actual_commit_id}, path={cached_repo_path}"
                        )
                        if (
                            repo_details.commit_id
                            and actual_commit_id != repo_details.commit_id
                        ):
                            logger.warning(
                                f"ParsingHelper: Worktree commit mismatch! Requested {repo_details.commit_id}, "
                                f"but worktree has {actual_commit_id}. Will create new worktree."
                            )
                            cached_repo_path = (
                                None  # Force creation of correct worktree
                            )
                    except Exception as e:
                        logger.warning(
                            f"ParsingHelper: Could not verify commit in worktree {cached_repo_path}: {e}"
                        )
                        # If we can't verify, but commit_id was specified, be cautious
                        if repo_details.commit_id:
                            logger.warning(
                                "ParsingHelper: Cannot verify commit_id, but it was specified. "
                                "Will attempt to use existing worktree."
                            )

                if cached_repo_path and os.path.exists(cached_repo_path):
                    logger.info(
                        f"ParsingHelper: Found existing repo in RepoManager at {cached_repo_path} "
                        f"(branch={repo_details.branch_name}, commit={repo_details.commit_id}), skipping clone"
                    )
                    # Update last accessed timestamp
                    try:
                        self.repo_manager.update_last_accessed(
                            repo_name=repo_details.repo_name,
                            branch=repo_details.branch_name,
                            commit_id=repo_details.commit_id,
                            user_id=user_id,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to update last_accessed for repo {repo_details.repo_name}: {e}"
                        )

                    repo_manager_path = cached_repo_path

                    # Extract owner from repo_name (format: owner/repo)
                    if "/" in repo_details.repo_name:
                        owner = repo_details.repo_name.split("/")[0]
                    else:
                        logger.debug(
                            f"Repo name '{repo_details.repo_name}' doesn't contain owner, will be extracted later if needed"
                        )

                    # Try to create a GitPython Repo from cached path (now we have real git repos)
                    try:
                        repo = Repo(cached_repo_path)
                        logger.info(
                            f"ParsingHelper: Created Repo object from cached path {cached_repo_path}"
                        )
                    except InvalidGitRepositoryError:
                        # Not a git repo - might be old tarball-based cache
                        # Fall back to GitHub API for metadata
                        logger.info(
                            "Cached path is not a git repo, getting GitHub API object"
                        )
                        try:
                            github, github_repo = self.github_service.get_repo(
                                repo_details.repo_name
                            )
                            owner = github_repo.owner.login

                            if hasattr(github, "_Github__requester") and hasattr(
                                github._Github__requester, "auth"
                            ):
                                auth = github._Github__requester.auth

                            repo = github_repo
                        except Exception as e:
                            logger.warning(
                                f"Could not get GitHub repo object: {e}. "
                                "Will use local language detection."
                            )
                            repo = None

                    logger.info(
                        f"ParsingHelper: Using cached repo from RepoManager at {cached_repo_path} "
                        f"(branch={repo_details.branch_name}, commit={repo_details.commit_id})"
                    )
                    return repo, owner, auth, repo_manager_path

                # Repo not in RepoManager - clone directly to RepoManager instead of .projects
                logger.info(
                    "ParsingHelper: Repo not found in RepoManager, cloning directly to .repos"
                )

                try:
                    # Get repo info from GitHub first (needed for auth and metadata)
                    logger.info(
                        f"ParsingHelper: Getting repo info from github_service for {repo_details.repo_name}"
                    )
                    github, github_repo = self.github_service.get_repo(
                        repo_details.repo_name
                    )
                    logger.info(
                        f"ParsingHelper: github_service.get_repo completed for {repo_details.repo_name}"
                    )
                    owner = github_repo.owner.login

                    # Extract auth from the Github client
                    if hasattr(github, "_Github__requester") and hasattr(
                        github._Github__requester, "auth"
                    ):
                        auth = github._Github__requester.auth
                    elif hasattr(github, "get_app_auth"):
                        auth = github.get_app_auth()
                    else:
                        logger.warning(
                            f"Could not extract auth from GitHub client for {repo_details.repo_name}"
                        )

                    # Clone directly to RepoManager directory
                    repo_manager_path = await self._clone_to_repo_manager(
                        github_repo=github_repo,
                        repo_name=repo_details.repo_name,
                        branch=repo_details.branch_name,
                        commit_id=repo_details.commit_id,
                        user_id=user_id,
                        auth=auth,
                        auth_token=auth_token,
                    )

                    if repo_manager_path:
                        # We now use git clone, so we have a real git repo
                        # Try to create Repo object from cloned path
                        try:
                            repo = Repo(repo_manager_path)
                            logger.info(
                                f"ParsingHelper: Cloned repo to RepoManager at {repo_manager_path}"
                            )
                        except InvalidGitRepositoryError:
                            # Fallback to github_repo for API access
                            logger.warning(
                                "Cloned path is not a valid git repo, using GitHub API object"
                            )
                            repo = github_repo
                        return repo, owner, auth, repo_manager_path
                    else:
                        # Fallback to normal flow if RepoManager clone failed
                        logger.warning(
                            "ParsingHelper: Failed to clone to RepoManager, falling back to normal flow"
                        )
                        repo = github_repo

                except HTTPException as he:
                    raise he
                except Exception:
                    logger.exception("Failed to fetch/clone repository")
                    raise HTTPException(
                        status_code=404,
                        detail="Repository not found or inaccessible on GitHub",
                    )
            else:
                # RepoManager disabled - use original flow
                try:
                    logger.info(
                        f"ParsingHelper: About to call github_service.get_repo for {repo_details.repo_name}"
                    )
                    github, repo = self.github_service.get_repo(repo_details.repo_name)
                    logger.info(
                        f"ParsingHelper: github_service.get_repo completed for {repo_details.repo_name}"
                    )
                    owner = repo.owner.login

                    # Extract auth from the Github client
                    if hasattr(github, "_Github__requester") and hasattr(
                        github._Github__requester, "auth"
                    ):
                        auth = github._Github__requester.auth
                    elif hasattr(github, "get_app_auth"):
                        auth = github.get_app_auth()
                    else:
                        logger.warning(
                            f"Could not extract auth from GitHub client for {repo_details.repo_name}"
                        )
                except HTTPException as he:
                    raise he
                except Exception:
                    logger.exception("Failed to fetch repository")
                    raise HTTPException(
                        status_code=404,
                        detail="Repository not found or inaccessible on GitHub",
                    )

        return repo, owner, auth, repo_manager_path

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
            _ = Repo.clone_from(
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
            logger.exception("ParsingHelper: Git clone failed")
            # Clean up temp directory on error
            if os.path.exists(temp_clone_dir):
                shutil.rmtree(temp_clone_dir)
            raise ParsingFailedError(f"Failed to clone repository: {e}") from e
        except Exception as e:
            logger.exception("ParsingHelper: Unexpected error during git clone")
            # Clean up temp directory on error
            if os.path.exists(temp_clone_dir):
                shutil.rmtree(temp_clone_dir)
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
        total_files_checked = 0
        files_by_ext = {}

        logger.info(
            f"detect_repo_language: Starting detection for {repo_dir} "
            f"(exists: {os.path.exists(repo_dir)}, isdir: {os.path.isdir(repo_dir) if os.path.exists(repo_dir) else False})"
        )

        if not os.path.exists(repo_dir):
            logger.error(f"detect_repo_language: Directory does not exist: {repo_dir}")
            return "other"

        if not os.path.isdir(repo_dir):
            logger.error(
                f"detect_repo_language: Path exists but is not a directory: {repo_dir}"
            )
            return "other"

        # Log a sample of what's in the directory
        try:
            dir_contents = os.listdir(repo_dir)
            logger.info(
                f"detect_repo_language: Directory contains {len(dir_contents)} items. "
                f"Sample: {dir_contents[:10]}"
            )
        except Exception as e:
            logger.warning(
                f"detect_repo_language: Could not list directory contents: {e}"
            )

        try:
            for root, _, files in os.walk(repo_dir):
                # Get relative path from repo_dir to avoid skipping paths that contain .repos_local etc.
                try:
                    rel_path = Path(root).relative_to(repo_dir)
                    # Handle root directory (rel_path == '.') and convert to tuple
                    rel_parts = rel_path.parts if rel_path != Path(".") else ()
                except ValueError:
                    # If relative_to fails, skip this path (shouldn't happen in normal os.walk)
                    continue

                # Skip .git directory (worktrees have .git as a file, not a directory)
                skip_this_dir = False
                if ".git" in rel_parts:
                    # Find where .git appears in the relative path
                    for i, part in enumerate(rel_parts):
                        if part == ".git":
                            # Check if this .git is a directory
                            git_path = Path(repo_dir) / Path(*rel_parts[: i + 1])
                            if git_path.is_dir():
                                # Skip this .git directory
                                skip_this_dir = True
                                break
                            # If it's a file, it's a worktree - continue processing
                            break

                if skip_this_dir:
                    continue

                # Skip hidden directories except .github, .vscode, etc. that might contain code
                # Only check relative path parts, not the base path
                if any(
                    part.startswith(".") and part not in [".github", ".vscode"]
                    for part in rel_parts
                ):
                    continue

                for file in files:
                    # Skip hidden files
                    if file.startswith("."):
                        continue

                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file)[1].lower()
                    total_files_checked += 1

                    # Track file extensions for debugging
                    files_by_ext[ext] = files_by_ext.get(ext, 0) + 1

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
                            # else:
                            #     lang_count["other"] += 1
                        except Exception as e:
                            logger.warning(f"Error processing file {file_path}: {e}")
                            continue
                    else:
                        logger.debug(
                            f"Could not read file with any encoding: {file_path}"
                        )
                        continue
        except (TypeError, FileNotFoundError, PermissionError) as e:
            logger.exception(f"Error accessing directory {repo_dir}: {e}")
            return "other"

        # Log detection results
        logger.info(
            f"detect_repo_language: Checked {total_files_checked} files, "
            f"found {sum(lang_count.values())} supported language files. "
            f"Language counts: {dict((k, v) for k, v in lang_count.items() if v > 0)}. "
            f"Top extensions: {dict(sorted(files_by_ext.items(), key=lambda x: x[1], reverse=True)[:10])}"
        )

        # Determine the predominant language based on counts
        predominant_language = max(lang_count, key=lang_count.get)
        result = (
            predominant_language if lang_count[predominant_language] > 0 else "other"
        )

        if result == "other":
            logger.warning(
                f"detect_repo_language: No supported language files found in {repo_dir}. "
                f"Total files checked: {total_files_checked}, "
                f"Top 10 extensions: {dict(sorted(files_by_ext.items(), key=lambda x: x[1], reverse=True)[:10])}"
            )

        return result

    async def download_and_extract_tarball(
        self,
        repo,
        ref: str,
        target_dir: str,
        auth: Any,
        repo_details: Any,
        user_id: str,
    ) -> str:
        """
        Download repository tarball from GitHub (or provider) and extract to target_dir.

        Args:
            repo: PyGithub Repository or MockRepo with get_archive_link
            ref: Branch name or commit SHA
            target_dir: Base directory for extraction (e.g. projects/)
            auth: Auth object (token used for authenticated download)
            repo_details: ParsingRequest or repo object (for full_name etc.)
            user_id: User ID for unique directory naming

        Returns:
            Path to the extracted directory (top-level content dir).
        """
        import tarfile
        import tempfile

        import aiohttp

        full_name = getattr(repo, "full_name", None) or getattr(
            repo_details, "repo_name", "unknown"
        )
        safe_name = full_name.replace("/", "-").replace(".", "-")
        safe_ref = ref.replace("/", "_").replace("\\", "_") if ref else "head"
        extract_subdir = os.path.join(
            target_dir, f"{safe_name}-{safe_ref}-{user_id}-{uuid.uuid4().hex[:8]}"
        )
        os.makedirs(extract_subdir, exist_ok=True)

        # Get archive URL (GitHub API or provider)
        if hasattr(repo, "get_archive_link"):
            archive_url = repo.get_archive_link("tarball", ref)
        elif hasattr(repo, "full_name"):
            # PyGithub Repository: build GitHub tarball URL
            archive_url = f"https://api.github.com/repos/{repo.full_name}/tarball/{ref}"
        else:
            raise ParsingFailedError(
                "Cannot get archive URL: repo has no get_archive_link or full_name"
            )

        # Resolve auth token for download
        token = None
        if auth is not None:
            if hasattr(auth, "token"):
                token = auth.token
            elif hasattr(auth, "password"):
                token = auth.password

        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        headers.setdefault("Accept", "application/vnd.github+json")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    archive_url, headers=headers, allow_redirects=True
                ) as resp:
                    if resp.status != 200:
                        raise ParsingFailedError(
                            f"Failed to download tarball: HTTP {resp.status}"
                        )
                    data = await resp.read()

            # Write to temp file and extract
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            try:
                with tarfile.open(tmp_path, "r:gz") as tf:
                    tf.extractall(extract_subdir)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            # GitHub tarballs have a single top-level dir (e.g. owner-repo-abc123/)
            entries = [e for e in os.listdir(extract_subdir) if not e.startswith(".")]
            if len(entries) == 1 and os.path.isdir(
                os.path.join(extract_subdir, entries[0])
            ):
                return os.path.join(extract_subdir, entries[0])
            return extract_subdir

        except ParsingFailedError:
            raise
        except Exception as e:
            if os.path.exists(extract_subdir):
                shutil.rmtree(extract_subdir, ignore_errors=True)
            logger.exception("ParsingHelper: download_and_extract_tarball failed")
            raise ParsingFailedError(
                f"Failed to download and extract repository: {e}"
            ) from e

    def _ensure_clean_worktree(
        self,
        repo_name: str,
        ref: str,
        auth_token: Optional[str],
        user_id: Optional[str],
        is_commit: bool = False,
    ) -> str:
        """
        Ensure a clean worktree exists by aggressively cleaning up any stale/corrupt worktrees.
        This method should NEVER fail - it tries multiple cleanup strategies and creates fresh.

        Args:
            repo_name: Repository name (owner/repo)
            ref: Branch name or commit SHA
            auth_token: Optional authentication token for cloning
            user_id: User ID for tracking
            is_commit: Whether ref is a commit SHA

        Returns:
            Path to the clean worktree directory

        Raises:
            HTTPException: Only if all recovery strategies fail
        """
        

        bare_repo_path = self.repo_manager.repos_base_path / repo_name / ".bare"
        worktree_path_result = self.repo_manager.get_worktree_path(repo_name, ref)
        if worktree_path_result:
            worktree_path = worktree_path_result
        else:
            worktree_path = self.repo_manager.repos_base_path / repo_name / "worktrees" / ref.replace("/", "_").replace("\\", "_")

        logger.info(f"_ensure_clean_worktree: Ensuring clean worktree for {repo_name}@{ref}")

        # Strategy 1: Prune stale git worktree registrations
        if os.path.exists(bare_repo_path):
            try:
                result = subprocess.run(
                    ["git", "-C", str(bare_repo_path), "worktree", "prune"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    logger.info(f"Pruned stale worktrees for {repo_name}")
            except Exception as e:
                logger.debug(f"Git worktree prune failed (non-critical): {e}")

        # Strategy 2: Try to unregister from git first (while directory still exists)
        if os.path.exists(bare_repo_path):
            try:
                # Try force remove first
                subprocess.run(
                    [
                        "git", "-C", str(bare_repo_path),
                        "worktree", "remove", "--force",
                        str(worktree_path)
                    ],
                    capture_output=True,
                    timeout=30,
                )
                logger.info(f"Force-removed worktree registration for {repo_name}@{ref}")
            except Exception:
                pass  # Ignore errors, prune already cleaned stale entries

        # Strategy 3: Remove worktree directory if it still exists (corrupt or empty)
        if os.path.exists(worktree_path):
            logger.info(f"Removing existing worktree directory: {worktree_path}")
            shutil.rmtree(worktree_path, ignore_errors=True)

        # Strategy 4: Create fresh worktree with exists_ok=True
        try:
            logger.info(f"Creating fresh worktree for {repo_name}@{ref}")
            new_path = self.repo_manager.create_worktree(
                repo_name=repo_name,
                ref=ref,
                auth_token=auth_token,
                is_commit=is_commit,
                user_id=user_id,
                exists_ok=True,  # Should succeed now that we've cleaned up
            )
            logger.info(f"Successfully created clean worktree at {new_path}")
            return str(new_path)
        except Exception as e:
            logger.error(f"Failed to create worktree even after cleanup: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create worktree for {repo_name}@{ref}: {str(e)}",
            ) from e

    async def setup_project_directory(
        self,
        repo,
        branch,
        auth,
        repo_details,
        user_id,
        project_id=None,  # Change type to str
        commit_id=None,
        repo_manager_path: Optional[
            str
        ] = None,  # New: path from RepoManager if available
        auth_token: Optional[str] = None,  # User's GitHub OAuth token for cloning
    ):
        """
        Set up the project directory for parsing.

        When repo_manager_path is provided (repo was cloned directly to RepoManager),
        this method skips tarball download and uses that path directly.

        Args:
            repo: Git Repo object or PyGithub Repository object
            branch: Branch name
            auth: Authentication object for GitHub API
            repo_details: ParsingRequest or Repo object with repository details
            user_id: User ID
            project_id: Project ID (optional)
            commit_id: Specific commit to checkout (optional)
            repo_manager_path: Path from RepoManager if repo was already cloned there
        """
        # Check if this is a local repository by examining the repo object
        # In development mode: repo is Repo object, repo_details is ParsingRequest
        # In non-development mode: both repo and repo_details can be Repo objects
        logger.info(
            f"ParsingHelper: setup_project_directory called with repo type: {type(repo).__name__}, "
            f"repo_details type: {type(repo_details).__name__}, repo_manager_path: {repo_manager_path}"
        )

        if repo_manager_path:
            # RepoManager-cached remote repo - DON'T set repo_path (it's a cached remote, not true local)
            repo_path = None
            if hasattr(repo_details, "repo_name"):
                full_name = repo_details.repo_name
            else:
                full_name = repo.full_name if hasattr(repo, "full_name") else None
            logger.info(
                f"ParsingHelper: Detected RepoManager-cached remote repository {full_name}"
            )
        elif isinstance(repo, Repo):
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

        # Check if we already have a path from RepoManager (cloned directly there)
        if repo_manager_path and os.path.exists(repo_manager_path):
            logger.info(
                f"ParsingHelper: Using RepoManager path directly at {repo_manager_path}, skipping tarball download"
            )

            # Validate that the path contains actual files (not just .git)
            file_count = 0
            try:
                for root, dirs, files in os.walk(repo_manager_path):
                    # Skip .git directory (check if .git is a directory component, not substring)
                    root_parts = root.split(os.sep)
                    if ".git" in root_parts:
                        git_idx = root_parts.index(".git")
                        git_path = os.sep.join(root_parts[: git_idx + 1])
                        if os.path.isdir(git_path):
                            continue
                    # Skip other hidden directories
                    if any(
                        part.startswith(".") and part not in [".github", ".vscode"]
                        for part in root_parts
                    ):
                        continue
                    file_count += sum(1 for f in files if not f.startswith("."))
                    if file_count > 10:
                        break
            except Exception as e:
                logger.warning(f"Error checking files in {repo_manager_path}: {e}")

            if file_count == 0:
                logger.error(
                    f"RepoManager path {repo_manager_path} exists but contains no source files. "
                    "This might be a worktree issue. Will recreate using bulletproof method."
                )
                # Use bulletproof helper to clean up and recreate worktree
                try:
                    repo_manager_path = self._ensure_clean_worktree(
                        repo_name=normalized_full_name,
                        ref=commit_id if commit_id else (branch if branch else "main"),
                        auth_token=auth_token,
                        user_id=user_id,
                        is_commit=bool(commit_id),
                    )
                    # Validate the new worktree has files
                    new_file_count = sum(
                        1 for _, _, files in os.walk(repo_manager_path)
                        for f in files if not f.startswith(".")
                    )
                    if new_file_count == 0:
                        raise HTTPException(
                            status_code=500,
                            detail="Recreated worktree still empty - possible bare repo corruption",
                        )
                    logger.info(
                        f"Successfully recreated worktree at {repo_manager_path} with {new_file_count} files"
                    )
                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"Failed to recreate worktree: {e}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to recreate worktree: {str(e)}",
                    )

            # At this point we have a valid worktree (either originally valid or recreated)
            logger.info(
                f"RepoManager path validated: using {repo_manager_path}"
            )
            extracted_dir = repo_manager_path

            # Get commit SHA from RepoManager metadata or from git
            latest_commit_sha = commit_id
            if not latest_commit_sha:
                try:
                    # Try to get from RepoManager metadata
                    if self.repo_manager:
                        repo_info = self.repo_manager.get_repo_info(
                            repo_name=normalized_full_name,
                            branch=branch,
                            commit_id=commit_id,
                            user_id=user_id,
                        )
                        if repo_info and repo_info.get("commit_id"):
                            latest_commit_sha = repo_info["commit_id"]

                    # Fallback: try to get from git
                    if not latest_commit_sha:
                        try:
                            git_repo = Repo(repo_manager_path)
                            latest_commit_sha = git_repo.head.commit.hexsha
                        except Exception:
                            # Last resort: get from GitHub API if repo is not a local git repo
                            if hasattr(repo, "get_branch"):
                                branch_details = repo.get_branch(branch)
                                latest_commit_sha = branch_details.commit.sha
                except Exception as e:
                    logger.warning(f"Could not determine commit SHA: {e}")
                    latest_commit_sha = commit_id or "unknown"

            # Extract metadata from repo for project update
            try:
                if repo is None:
                    # No repo object available (cached without API access)
                    repo_metadata = {}
                elif isinstance(repo, Repo):
                    repo_metadata = ParseHelper.extract_local_repo_metadata(repo)
                else:
                    repo_metadata = ParseHelper.extract_remote_repo_metadata(repo)
            except Exception as e:
                logger.warning(f"Could not extract repo metadata: {e}")
                repo_metadata = {}

            repo_metadata["error_message"] = None
            project_metadata = json.dumps(repo_metadata).encode("utf-8")
            ProjectService.update_project(
                self.db,
                project_id,
                properties=project_metadata,
                commit_id=latest_commit_sha,
                status=ProjectStatusEnum.CLONED.value,
            )

            logger.info(
                f"ParsingHelper: Project directory setup complete using RepoManager path: {extracted_dir}"
            )
            return extracted_dir, project_id

        # RepoManager path was set but path missing - try to recreate worktree if bare repo exists
        if repo_manager_path and not os.path.exists(repo_manager_path):
            # Extract repo name from repo_details or use the stored full_name
            repo_name_for_recreate = None
            if hasattr(repo_details, "repo_name"):
                repo_name_for_recreate = repo_details.repo_name
            elif 'full_name' in locals():
                repo_name_for_recreate = full_name

            if repo_name_for_recreate:
                # Check if bare repo exists - if so, we can recreate just the worktree
                bare_repo_path = self.repo_manager._get_bare_repo_path(repo_name_for_recreate)
                if os.path.exists(bare_repo_path):
                    logger.info(
                        f"RepoManager worktree missing at {repo_manager_path} but bare repo exists. "
                        f"Using bulletproof recreation for {repo_name_for_recreate}..."
                    )
                    try:
                        ref = commit_id if commit_id else (branch if branch else "main")
                        is_commit = bool(commit_id)
                        # Use bulletproof helper to ensure clean worktree creation
                        new_worktree_path = self._ensure_clean_worktree(
                            repo_name=repo_name_for_recreate,
                            ref=ref,
                            auth_token=auth_token,
                            user_id=user_id,
                            is_commit=is_commit,
                        )
                        logger.info(
                            f"Successfully recreated worktree at {new_worktree_path}"
                        )
                        repo_manager_path = str(new_worktree_path)

                        # Persist project metadata after worktree recreation (same as "path exists" branch)
                        latest_commit_sha = commit_id
                        if not latest_commit_sha:
                            try:
                                if self.repo_manager:
                                    repo_info = self.repo_manager.get_repo_info(
                                        repo_name=normalized_full_name,
                                        branch=branch,
                                        commit_id=commit_id,
                                        user_id=user_id,
                                    )
                                    if repo_info and repo_info.get("commit_id"):
                                        latest_commit_sha = repo_info["commit_id"]
                                if not latest_commit_sha:
                                    try:
                                        git_repo = Repo(repo_manager_path)
                                        latest_commit_sha = git_repo.head.commit.hexsha
                                    except Exception:
                                        if hasattr(repo, "get_branch"):
                                            branch_details = repo.get_branch(branch)
                                            latest_commit_sha = branch_details.commit.sha
                            except Exception as e:
                                logger.warning(f"Could not determine commit SHA: {e}")
                            latest_commit_sha = latest_commit_sha or commit_id or "unknown"
                        try:
                            if repo is None:
                                repo_metadata = {}
                            elif isinstance(repo, Repo):
                                repo_metadata = ParseHelper.extract_local_repo_metadata(repo)
                            else:
                                repo_metadata = ParseHelper.extract_remote_repo_metadata(repo)
                        except Exception as e:
                            logger.warning(f"Could not extract repo metadata: {e}")
                            repo_metadata = {}
                        repo_metadata["error_message"] = None
                        project_metadata = json.dumps(repo_metadata).encode("utf-8")
                        ProjectService.update_project(
                            self.db,
                            project_id,
                            properties=project_metadata,
                            commit_id=latest_commit_sha,
                            status=ProjectStatusEnum.CLONED.value,
                        )
                        logger.info(
                            f"ParsingHelper: Project directory setup complete using recreated worktree: {repo_manager_path}"
                        )
                        return repo_manager_path, project_id
                    except HTTPException:
                        raise
                    except Exception as e:
                        logger.error(f"Failed to recreate worktree: {e}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to recreate worktree: {str(e)}",
                        )
                else:
                    # Bare repo also missing - need full re-clone
                    logger.error(
                        f"RepoManager path {repo_manager_path} is missing and bare repo not found at {bare_repo_path}. "
                        "Full re-clone required."
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "Repository cache path is missing and bare repo not found. "
                            "Please trigger a new parse to re-clone the repository."
                        ),
                    )
            else:
                logger.error(
                    f"RepoManager path {repo_manager_path} is missing and cannot determine repo name for recreation."
                )
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Repository cache path is missing and cannot recreate worktree. "
                        "Please trigger a new parse to re-clone the repository."
                    ),
                )

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
            # Worktree-only mode: If we reach here, worktree creation failed
            logger.error(
                "Worktree creation failed and tarball fallback is disabled. "
                "Only RepoManager worktrees are supported."
            )
            raise HTTPException(
                status_code=500,
                detail=(
                    "Failed to create worktree for repository parsing. "
                    "Please check RepoManager status and try again."
                ),
            )

        # Use repo instead of repo_details for metadata extraction
        # repo is always the MockRepo (remote) or Repo (local) object with required methods
        # repo_details can be ParsingRequest in dev mode, which lacks these methods
        repo_metadata = self.extract_repository_metadata(repo)
        repo_metadata["error_message"] = None
        project_metadata = json.dumps(repo_metadata).encode("utf-8")
        ProjectService.update_project(
            self.db,
            project_id,
            properties=project_metadata,
            commit_id=latest_commit_sha,
            status=ProjectStatusEnum.CLONED.value,
        )

        # Note: When RepoManager is enabled as primary source (see clone_or_copy_repository),
        # repos are cloned directly to .repos via _clone_to_repo_manager.
        # The _copy_repo_to_repo_manager fallback is only for backward compatibility
        # when RepoManager is enabled but direct clone failed.
        # This is intentionally skipped when RepoManager is the primary source of truth.
        # If needed for backward compat with non-primary mode, uncomment below:
        #
        # if self.repo_manager and extracted_dir and os.path.exists(extracted_dir):
        #     try:
        #         await self._copy_repo_to_repo_manager(
        #             normalized_full_name,
        #             extracted_dir,
        #             branch,
        #             latest_commit_sha,
        #             user_id,
        #             repo_metadata,
        #         )
        #     except Exception as e:
        #         logger.warning(
        #             f"Failed to copy repo to repo manager: {e}. Continuing with parsing."
        #         )

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
        except Exception:
            logger.exception("Error creating worktree for repo manager")
            raise

    async def _clone_to_repo_manager(
        self,
        github_repo,
        repo_name: str,
        branch: Optional[str],
        commit_id: Optional[str],
        user_id: str,
        auth: Any,
        *,
        auth_token: Optional[str] = None,
    ) -> Optional[str]:
        """
        Add repository to RepoManager using git clone/worktree.

        If the base repo already exists in RepoManager, creates a worktree
        for the requested branch/commit (very fast - no download needed).

        If the base repo doesn't exist, clones it first then creates worktree.
        When auth_token (user's GitHub OAuth token) is provided, uses
        RepoManager.prepare_for_parsing() so the token is used for cloning.

        Args:
            github_repo: PyGithub Repository object
            repo_name: Full repository name (e.g., 'owner/repo')
            branch: Branch name
            commit_id: Commit SHA (optional)
            user_id: User ID
            auth: Authentication object for GitHub API
            auth_token: Optional user GitHub OAuth token for cloning private repos

        Returns:
            Path to the repository worktree in .repos, or None if failed
        """
        if not self.repo_manager:
            return None

        try:
            # Determine the base repo path in .repos (e.g., .repos/owner/repo)
            base_repo_path = self.repo_manager._get_repo_local_path(repo_name)
            ref = commit_id if commit_id else branch

            if not ref:
                logger.warning(
                    f"No branch or commit_id provided for {repo_name}, cannot clone"
                )
                return None

            # Worktree path for this specific branch/commit
            # Use commit_id directly if provided to ensure exact match
            if commit_id:
                worktree_name = commit_id.replace("/", "_").replace("\\", "_")
                logger.info(
                    f"ParsingHelper: Creating worktree for exact commit_id={commit_id}, "
                    f"worktree_name={worktree_name}"
                )
            else:
                worktree_name = branch.replace("/", "_").replace("\\", "_")
                logger.info(
                    f"ParsingHelper: Creating worktree for branch={branch}, "
                    f"worktree_name={worktree_name}"
                )
            worktree_path = base_repo_path / "worktrees" / worktree_name

            logger.info(
                f"ParsingHelper: Worktree path will be: {worktree_path} "
                f"(for commit_id={commit_id}, branch={branch})"
            )

            # Check if base repo already exists (has .git directory)
            base_git_dir = base_repo_path / ".git"

            if base_git_dir.exists():
                # Base repo exists - just create worktree (fast path!)
                logger.info(
                    f"ParsingHelper: Base repo exists at {base_repo_path}, "
                    f"creating worktree for {ref}"
                )

                try:
                    base_repo = Repo(base_repo_path)

                    # Fetch latest to ensure we have the commit
                    logger.info(f"ParsingHelper: Fetching latest for {repo_name}")
                    for remote in base_repo.remotes:
                        try:
                            remote.fetch()
                        except Exception as e:
                            logger.warning(f"Failed to fetch from {remote.name}: {e}")

                    # Create worktree for the requested ref
                    worktree_path_str = await self._create_git_worktree(
                        base_repo=base_repo,
                        worktree_path=worktree_path,
                        ref=ref,
                        is_commit=commit_id is not None,
                    )

                    if worktree_path_str:
                        # Always get actual commit SHA from worktree to ensure accuracy
                        actual_commit_id = None
                        try:
                            worktree_repo = Repo(worktree_path_str)
                            actual_commit_id = worktree_repo.head.commit.hexsha
                            logger.info(
                                f"ParsingHelper: Worktree created at {worktree_path_str}, "
                                f"actual commit_id={actual_commit_id} "
                                f"(requested commit_id={commit_id}, branch={branch})"
                            )
                            # Verify commit_id matches if it was specified
                            if commit_id and actual_commit_id != commit_id:
                                logger.warning(
                                    f"ParsingHelper: Commit mismatch! Requested {commit_id}, "
                                    f"but worktree has {actual_commit_id}. Using actual commit_id."
                                )
                        except Exception as e:
                            logger.warning(
                                f"Could not get commit SHA from worktree: {e}"
                            )
                            # Fallback to requested commit_id or fetch from branch
                            actual_commit_id = commit_id
                            if not actual_commit_id:
                                try:
                                    branch_info = github_repo.get_branch(branch)
                                    actual_commit_id = branch_info.commit.sha
                                except Exception as e2:
                                    logger.warning(
                                        f"Could not get commit SHA from branch: {e2}"
                                    )

                        # Register with RepoManager
                        try:
                            repo_metadata = ParseHelper.extract_remote_repo_metadata(
                                github_repo
                            )
                        except Exception:
                            repo_metadata = {}

                        self.repo_manager.register_repo(
                            repo_name=repo_name,
                            local_path=worktree_path_str,
                            branch=branch,
                            commit_id=actual_commit_id,
                            user_id=user_id,
                            metadata=repo_metadata,
                        )

                        logger.info(
                            f"ParsingHelper: Created worktree for {repo_name}@{ref} "
                            f"at {worktree_path_str} (fast path - no download)"
                        )
                        return worktree_path_str

                except Exception as e:
                    logger.warning(
                        f"Failed to use existing base repo, will re-clone: {e}"
                    )
                    # Fall through to fresh clone

            # Base repo doesn't exist - need to clone it
            logger.info(
                f"ParsingHelper: Base repo not found, cloning {repo_name} to {base_repo_path}"
            )

            # When user's GitHub OAuth token is available, use RepoManager.prepare_for_parsing
            # so the token is used for cloning (prioritized over environment tokens).
            if auth_token:
                try:
                    worktree_path_str = self.repo_manager.prepare_for_parsing(
                        repo_name,
                        ref,
                        auth_token=auth_token,
                        user_id=user_id,
                        is_commit=bool(commit_id),
                    )
                    if worktree_path_str:
                        logger.info(
                            f"ParsingHelper: Cloned {repo_name} via prepare_for_parsing with user token"
                        )
                        return worktree_path_str
                except Exception as e:
                    logger.warning(
                        f"ParsingHelper: prepare_for_parsing with user token failed for {repo_name}: {e}. Falling back to clone with API auth."
                    )

            # Build clone URL with authentication
            clone_url = await self._build_clone_url(github_repo, auth)

            if not clone_url:
                logger.error(f"Could not build clone URL for {repo_name}")
                return None

            # Create parent directory
            base_repo_path.parent.mkdir(parents=True, exist_ok=True)

            # Clone the repository
            logger.info(f"ParsingHelper: Cloning {repo_name} (this may take a moment)")
            try:
                base_repo = Repo.clone_from(
                    clone_url,
                    str(base_repo_path),
                    branch=branch or github_repo.default_branch,
                    depth=None,  # Full clone to support worktrees
                )
                logger.info(f"ParsingHelper: Successfully cloned {repo_name}")
            except Exception as e:
                logger.exception(f"Failed to clone {repo_name}: {e}")
                return None

            # Now create worktree for the specific ref
            worktree_path_str = await self._create_git_worktree(
                base_repo=base_repo,
                worktree_path=worktree_path,
                ref=ref,
                is_commit=commit_id is not None,
            )

            if not worktree_path_str:
                # Worktree creation failed - don't return bare repo (not usable for parsing)
                logger.error(
                    f"Worktree creation failed for {repo_name}. Bare repo exists but cannot be used for parsing."
                )
                return None

            # Always get actual commit SHA from worktree to ensure accuracy
            actual_commit_id = None
            try:
                worktree_repo = Repo(worktree_path_str)
                actual_commit_id = worktree_repo.head.commit.hexsha
                logger.info(
                    f"ParsingHelper: Worktree created at {worktree_path_str}, "
                    f"actual commit_id={actual_commit_id} "
                    f"(requested commit_id={commit_id}, branch={branch})"
                )
                # Verify commit_id matches if it was specified
                if commit_id and actual_commit_id != commit_id:
                    logger.warning(
                        f"ParsingHelper: Commit mismatch! Requested {commit_id}, "
                        f"but worktree has {actual_commit_id}. Using actual commit_id."
                    )
            except Exception as e:
                logger.warning(f"Could not get commit SHA from worktree: {e}")
                # Fallback to requested commit_id or fetch from branch
                actual_commit_id = commit_id
                if not actual_commit_id:
                    try:
                        branch_info = github_repo.get_branch(branch)
                        actual_commit_id = branch_info.commit.sha
                    except Exception as e2:
                        logger.warning(f"Could not get commit SHA from branch: {e2}")

            # Extract metadata
            try:
                repo_metadata = ParseHelper.extract_remote_repo_metadata(github_repo)
            except Exception:
                repo_metadata = {}

            # Register with RepoManager
            self.repo_manager.register_repo(
                repo_name=repo_name,
                local_path=worktree_path_str,
                branch=branch,
                commit_id=actual_commit_id,
                user_id=user_id,
                metadata=repo_metadata,
            )

            logger.info(
                f"ParsingHelper: Successfully cloned and registered {repo_name}@{ref} "
                f"in RepoManager at {worktree_path_str}"
            )

            return worktree_path_str

        except Exception as e:
            logger.exception(f"Failed to add {repo_name} to RepoManager: {e}")
            return None

    async def _build_clone_url(self, github_repo, auth: Any) -> Optional[str]:
        """Build authenticated clone URL for the repository."""
        try:
            clone_url = github_repo.clone_url

            if auth:
                # Insert token into URL for authentication
                # Format: https://token@github.com/owner/repo.git
                from urllib.parse import urlparse, urlunparse

                parsed = urlparse(clone_url)

                # Get token from auth object
                token = None
                if hasattr(auth, "token"):
                    token = auth.token
                elif hasattr(auth, "password"):
                    token = auth.password

                if token:
                    # Reconstruct URL with token
                    netloc_with_auth = f"{token}@{parsed.netloc}"
                    clone_url = urlunparse(
                        (
                            parsed.scheme,
                            netloc_with_auth,
                            parsed.path,
                            parsed.params,
                            parsed.query,
                            parsed.fragment,
                        )
                    )

            return clone_url
        except Exception as e:
            logger.warning(f"Failed to build clone URL: {e}")
            return github_repo.clone_url if hasattr(github_repo, "clone_url") else None

    async def _create_git_worktree(
        self,
        base_repo: Repo,
        worktree_path: Path,
        ref: str,
        is_commit: bool,
    ) -> Optional[str]:
        """
        Create a git worktree for the specified ref.

        Args:
            base_repo: The base git repository
            worktree_path: Path where worktree should be created
            ref: Branch name or commit SHA
            is_commit: True if ref is a commit SHA, False if it's a branch

        Returns:
            Path to the worktree, or None if creation failed
        """
        try:
            # Remove existing worktree if it exists
            if worktree_path.exists():
                try:
                    base_repo.git.worktree("remove", str(worktree_path), force=True)
                except Exception:
                    shutil.rmtree(worktree_path, ignore_errors=True)
                    # Clean up stale worktree references after manual deletion
                    # Per git docs: manual deletion without 'git worktree remove'
                    # leaves administrative files that need pruning
                    try:
                        base_repo.git.worktree("prune")
                    except Exception:
                        pass  # Non-critical, continue with worktree creation

            # Create worktree directory parent
            worktree_path.parent.mkdir(parents=True, exist_ok=True)

            if is_commit:
                # For specific commit, use detached HEAD
                base_repo.git.worktree("add", "--detach", str(worktree_path), ref)
            else:
                # For branch, try to track it
                try:
                    base_repo.git.worktree("add", str(worktree_path), ref)
                except GitCommandError:
                    # Branch might not exist locally, try with remote tracking
                    try:
                        base_repo.git.worktree(
                            "add",
                            "--track",
                            "-b",
                            ref,
                            str(worktree_path),
                            f"origin/{ref}",
                        )
                    except GitCommandError:
                        # Last resort: detached HEAD at origin/branch
                        base_repo.git.worktree(
                            "add", "--detach", str(worktree_path), f"origin/{ref}"
                        )

            logger.info(f"Created git worktree at {worktree_path}")
            return str(worktree_path)

        except Exception as e:
            logger.exception(f"Failed to create worktree at {worktree_path}: {e}")
            return None

    def _initialize_base_repo(self, base_repo_path: Path, extracted_dir: str) -> Repo:
        """
        Initialize or get the base git repository.

        If the base repo doesn't exist, initialize it and copy the extracted repo.
        If it exists, return the existing repo.
        """

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

    def extract_repository_metadata(self, repo):
        if isinstance(repo, Repo):
            metadata = ParseHelper.extract_local_repo_metadata(repo)
        else:
            metadata = ParseHelper.extract_remote_repo_metadata(repo)
        return metadata

    @staticmethod
    def extract_local_repo_metadata(repo: Repo):
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

    @staticmethod
    def get_local_repo_languages(path: str | os.PathLike[str]) -> dict[str, int]:
        root = Path(path).resolve()
        if not root.exists():
            return {}

        language_bytes = defaultdict(int)
        total_bytes = 0

        stack = [root]

        while stack:
            current = stack.pop()

            try:
                entries = current.iterdir()
                for entry in entries:
                    try:
                        if not entry.is_symlink() and entry.is_dir():
                            stack.append(entry)
                        elif not entry.is_symlink() and entry.is_file():
                            size = entry.stat().st_size
                            total_bytes += size

                            if entry.suffix == ".py":
                                language_bytes["Python"] += size
                            elif entry.suffix == ".ts":
                                language_bytes["TypeScript"] += size
                            elif entry.suffix == ".js":
                                language_bytes["JavaScript"] += size
                            else:
                                language_bytes["Other"] += size

                    except OSError:
                        # Permission issues, broken files, etc.
                        continue
            except OSError:
                continue

        return dict(language_bytes) if total_bytes else {}

    @staticmethod
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
            logger.info(
                f"check_commit_status: Branch is empty (pinned commit parse) - "
                f"sticking to commit and not updating it for: {project_id}"
            )
            return True

        if len(repo_name.split("/")) < 2:
            # Local repo, always parse local repos
            logger.info("check_commit_status: Local repo detected, forcing reparse")
            return False

        try:
            # If current_commit_id is None, we should reparse
            if current_commit_id is None:
                logger.info(
                    f"check_commit_status: Project {project_id} has no commit_id, will reparse"
                )
                return False

            # Run blocking GitHub API (get_repo, get_branch) in thread pool to avoid blocking the event loop
            def _fetch_latest_commit_sync() -> Optional[str]:
                _github, repo = self.github_service.get_repo(repo_name)
                branch = repo.get_branch(branch_name)
                return branch.commit.sha

            logger.info(
                f"check_commit_status: Branch-based parse - getting repo info for {repo_name}"
            )
            latest_commit_id = await asyncio.to_thread(
                _fetch_latest_commit_sync
            )

            # Compare current commit with latest commit
            is_up_to_date = current_commit_id == latest_commit_id
            logger.info(
                f"check_commit_status: Project {project_id} commit status for branch {branch_name}: "
                f"{'Up to date' if is_up_to_date else 'Outdated'} - "
                f"Current: {current_commit_id}, Latest: {latest_commit_id}"
            )

            return is_up_to_date
        except Exception:
            logger.exception(
                "check_commit_status: Error fetching latest commit",
                repo_name=repo_name,
                branch_name=branch_name,
                project_id=project_id,
            )
            return False

import os
import shutil
import uuid
from typing import Any, Tuple
from urllib.parse import urlparse, urlunparse
from pathlib import Path
from collections import defaultdict


from fastapi import HTTPException
from git import GitCommandError, Repo
from sqlalchemy.orm import Session

from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.parsing.graph_construction.parsing_schema import RepoDetails
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
            except Exception:
                logger.exception("Failed to fetch repository")
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
                            # else:
                            #     lang_count["other"] += 1
                        except Exception as e:
                            logger.warning(f"Error processing file {file_path}: {e}")
                            continue
                    else:
                        logger.warning(
                            f"Could not read file with any encoding: {file_path}"
                        )
                        continue
        except (TypeError, FileNotFoundError, PermissionError):
            logger.exception("Error accessing directory", repo_dir=repo_dir)

        # Determine the predominant language based on counts
        predominant_language = max(lang_count, key=lang_count.get)
        return predominant_language if lang_count[predominant_language] > 0 else "other"

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
        except Exception:
            logger.exception(
                "check_commit_status: Error fetching latest commit",
                repo_name=repo_name,
                branch_name=branch_name,
                project_id=project_id,
            )
            return False

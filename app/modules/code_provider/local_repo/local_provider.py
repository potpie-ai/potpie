import os
from typing import Any, Dict, List, Optional

from git import GitCommandError, InvalidGitRepositoryError, NoSuchPathError, Repo
import pathspec

from app.modules.code_provider.base.code_provider_interface import (
    AuthMethod,
    ICodeProvider,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class LocalProvider(ICodeProvider):
    """Filesystem-backed code provider implementation for local Git repositories."""

    def __init__(self, default_repo_path: Optional[str] = None):
        """
        Initialize local provider.

        Args:
            default_repo_path: Optional default repository path. If not provided,
                             repo_name parameter in methods should be the full path.
        """
        self.default_repo_path = (
            os.path.abspath(os.path.expanduser(default_repo_path))
            if default_repo_path
            else None
        )
        # Add client property for ProviderWrapper compatibility
        self.client = None  # Local provider doesn't have a client

    # ============ Authentication ============

    def authenticate(self, credentials: Dict[str, Any], method: AuthMethod) -> Any:
        """Authentication is not required for local repositories."""
        logger.debug(
            "LocalProvider.authenticate called; no action taken for local repos"
        )
        return None

    def get_supported_auth_methods(self) -> List[AuthMethod]:
        return []

    # ============ Repository Helpers ============

    def _get_repo(self, repo_name: Optional[str]) -> Repo:
        path = repo_name or self.default_repo_path
        if not path:
            raise ValueError(
                "Repository path is required for local provider operations"
            )

        expanded_path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(expanded_path):
            raise FileNotFoundError(f"Local repository at {expanded_path} not found")

        try:
            return Repo(expanded_path)
        except (InvalidGitRepositoryError, NoSuchPathError) as exc:
            raise ValueError(f"Path {expanded_path} is not a git repository") from exc

    def _get_gitignore_spec(self, repo_path: str) -> Optional[pathspec.PathSpec]:
        """
        Get gitignore PathSpec for the repository.

        Args:
            repo_path: Absolute path to repository

        Returns:
            PathSpec object or None if no .gitignore exists
        """
        gitignore_path = os.path.join(repo_path, ".gitignore")

        if not os.path.exists(gitignore_path):
            return None

        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                patterns = f.read().splitlines()

            # Add common exclusions
            patterns.extend(
                [
                    ".git",
                    "__pycache__",
                    "*.pyc",
                    "node_modules",
                    ".DS_Store",
                ]
            )

            return pathspec.PathSpec.from_lines("gitwildmatch", patterns)
        except Exception as e:
            logger.warning(f"Error reading .gitignore: {e}")
            return None

    def _should_include_file(self, file_path: str) -> bool:
        """
        Check if file should be included based on extension.

        Args:
            file_path: Path to file

        Returns:
            True if file should be included
        """
        # Exclude image/media files
        exclude_extensions = {
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
        }

        ext = os.path.splitext(file_path)[1].lstrip(".").lower()
        return ext not in exclude_extensions

    # ============ Repository Operations ============

    def get_repository(self, repo_name: str) -> Dict[str, Any]:
        """
        Get repository metadata.

        Args:
            repo_name: Path to the local repository

        Returns:
            Dictionary with repository metadata
        """
        repo = self._get_repo(repo_name)
        repo_path = os.path.abspath(
            os.path.expanduser(repo_name or self.default_repo_path)
        )

        return {
            "id": hash(repo_path),  # Generate pseudo-ID from path hash
            "name": os.path.basename(repo_path),
            "full_name": repo_path,
            "owner": os.path.expanduser("~").split(os.sep)[-1],  # Current user
            "default_branch": repo.active_branch.name,
            "private": True,  # Local repos are always private
            "url": f"file://{repo_path}",
            "description": None,
            "language": None,  # Could detect from gitattributes if needed
        }

    def check_repository_access(self, repo_name: str) -> bool:
        try:
            self._get_repo(repo_name)
            return True
        except Exception:
            return False

    # ============ Content Operations ============

    def get_file_content(
        self,
        repo_name: str,
        file_path: str,
        ref: Optional[str] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        """
        Get file content from local repository.

        Args:
            repo_name: Path to the local repository
            file_path: Relative path to file within repository
            ref: Branch name or commit SHA (defaults to active branch)
            start_line: Optional starting line number (1-indexed, inclusive)
            end_line: Optional ending line number (1-indexed, inclusive)

        Returns:
            File content as string

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If ref is invalid
        """
        from app.modules.code_provider.git_safe import safe_git_repo_operation

        expanded_path = os.path.abspath(
            os.path.expanduser(repo_name or self.default_repo_path)
        )

        def _get_content(repo):
            # Use active branch if no ref specified
            actual_ref = ref
            if not actual_ref:
                actual_ref = repo.active_branch.name

            # Use git show to read file without checking out
            # This avoids modifying working directory state
            try:
                file_content = repo.git.show(f"{actual_ref}:{file_path}")
            except GitCommandError as e:
                if "does not exist" in str(e) or "path not in" in str(e):
                    raise FileNotFoundError(
                        f"File not found: {file_path} at ref {actual_ref}"
                    ) from e
                elif "unknown revision" in str(e):
                    raise ValueError(f"Invalid ref: {actual_ref}") from e
                else:
                    raise

            # Apply line range if specified
            if start_line is not None or end_line is not None:
                lines = file_content.splitlines()

                # Convert to 0-indexed
                start_idx = (start_line - 1) if start_line else 0
                end_idx = end_line if end_line else len(lines)

                # Extract line range
                file_content = "\n".join(lines[start_idx:end_idx])

            return file_content

        operation_name = f"get_file_content({file_path}@{ref or 'default'})"
        # Use explicit timeout to prevent blocking when called from within an outer timeout
        # The outer timeout (e.g., 20s in _get_current_content) must be longer than this
        # to prevent orphaned threads from nested ThreadPoolExecutors
        return safe_git_repo_operation(
            expanded_path,
            _get_content,
            max_retries=1,  # Reduced retries for faster failure
            timeout=15.0,  # Explicit timeout shorter than typical outer timeouts
            operation_name=operation_name,
        )

    def get_repository_structure(
        self,
        repo_name: str,
        path: str = "",
        ref: Optional[str] = None,
        max_depth: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Get repository directory structure.

        Args:
            repo_name: Path to the local repository
            path: Relative path within repository (default: root)
            ref: Branch name or commit SHA (default: active branch)
            max_depth: Maximum directory depth to traverse

        Returns:
            List of dictionaries with structure:
            [{
                "name": "filename",
                "path": "relative/path",
                "type": "file" | "directory",
                "size": 1234,
                "children": [...]  # For directories
            }]
        """
        repo = self._get_repo(repo_name)
        repo_path = os.path.abspath(
            os.path.expanduser(repo_name or self.default_repo_path)
        )

        # Get gitignore spec
        gitignore_spec = self._get_gitignore_spec(repo_path)

        # Build full path
        full_path = os.path.join(repo_path, path) if path else repo_path

        # Traverse directory structure
        return self._traverse_directory(
            full_path, repo_path, gitignore_spec, current_depth=0, max_depth=max_depth
        )

    def _traverse_directory(
        self,
        dir_path: str,
        repo_root: str,
        gitignore_spec: Optional[pathspec.PathSpec],
        current_depth: int,
        max_depth: int,
    ) -> List[Dict[str, Any]]:
        """
        Recursively traverse directory structure.

        Args:
            dir_path: Absolute path to directory
            repo_root: Absolute path to repository root
            gitignore_spec: PathSpec for gitignore filtering
            current_depth: Current traversal depth
            max_depth: Maximum depth to traverse

        Returns:
            List of file/directory dictionaries
        """
        if current_depth >= max_depth:
            return []

        result = []

        try:
            entries = os.listdir(dir_path)
        except (PermissionError, FileNotFoundError):
            return []

        for entry in sorted(entries):
            # Skip .git directory only
            if entry == ".git":
                continue

            entry_path = os.path.join(dir_path, entry)
            relative_path = os.path.relpath(entry_path, repo_root)

            # Check gitignore
            if gitignore_spec and gitignore_spec.match_file(relative_path):
                continue

            if os.path.isfile(entry_path):
                # Skip non-text files
                if not self._should_include_file(entry_path):
                    continue

                file_info = {
                    "name": entry,
                    "path": relative_path,
                    "type": "file",
                    "size": os.path.getsize(entry_path),
                }
                result.append(file_info)

            elif os.path.isdir(entry_path):
                children = self._traverse_directory(
                    entry_path,
                    repo_root,
                    gitignore_spec,
                    current_depth + 1,
                    max_depth,
                )

                dir_info = {
                    "name": entry,
                    "path": relative_path,
                    "type": "directory",
                    "children": children,
                }
                result.append(dir_info)

        return result

    # ============ Branch Operations ============

    def list_branches(self, repo_name: str) -> List[str]:
        repo = self._get_repo(repo_name)

        branches = [head.name for head in repo.heads]

        # Try to move the currently checked-out branch to the front
        try:
            active = repo.active_branch.name
        except TypeError:
            # Detached HEAD or no branches; leave list as-is
            active = None
        except Exception as exc:
            logger.debug(f"LocalProvider: unable to determine active branch: {exc}")
            active = None

        if active and active in branches:
            branches.remove(active)
            branches.insert(0, active)

        return branches

    def get_branch(self, repo_name: str, branch_name: str) -> Dict[str, Any]:
        """
        Get branch metadata.

        Args:
            repo_name: Path to the local repository
            branch_name: Name of the branch

        Returns:
            Dictionary with branch metadata

        Raises:
            ValueError: If branch doesn't exist
        """
        repo = self._get_repo(repo_name)

        try:
            branch = repo.heads[branch_name]
        except IndexError:
            raise ValueError(f"Branch not found: {branch_name}")

        return {
            "name": branch_name,
            "commit_sha": branch.commit.hexsha,
            "protected": False,  # Local repos don't have protected branch concept
        }

    def create_branch(
        self, repo_name: str, branch_name: str, base_branch: str
    ) -> Dict[str, Any]:
        """
        Create a new branch.

        Args:
            repo_name: Path to the local repository
            branch_name: Name of the new branch
            base_branch: Name of the base branch to branch from

        Returns:
            Dictionary with new branch metadata

        Raises:
            ValueError: If base branch doesn't exist or new branch already exists
        """
        repo = self._get_repo(repo_name)

        # Check if base branch exists
        if base_branch not in repo.heads:
            raise ValueError(f"Base branch not found: {base_branch}")

        # Check if new branch already exists
        if branch_name in repo.heads:
            raise ValueError(f"Branch already exists: {branch_name}")

        try:
            # Create new branch from base
            new_branch = repo.create_head(branch_name, repo.heads[base_branch])

            return {
                "name": branch_name,
                "commit_sha": new_branch.commit.hexsha,
                "protected": False,
            }
        except Exception as e:
            raise ValueError(f"Failed to create branch: {e}") from e

    def compare_branches(
        self, repo_name: str, base_branch: str, head_branch: str
    ) -> Dict[str, Any]:
        """
        Compare two branches.

        Args:
            repo_name: Path to the local repository
            base_branch: Base branch name
            head_branch: Head branch name

        Returns:
            Dictionary with comparison results:
            {
                "files": [
                    {
                        "filename": "path/to/file",
                        "patch": "diff output",
                        "status": "modified|added|deleted"
                    }
                ],
                "commits": 12  # Number of commits difference
            }
        """
        repo = self._get_repo(repo_name)

        # Validate branches exist
        if base_branch not in repo.heads:
            raise ValueError(f"Base branch not found: {base_branch}")
        if head_branch not in repo.heads:
            raise ValueError(f"Head branch not found: {head_branch}")

        # Get diff between branches
        try:
            diff_output = repo.git.diff(
                f"{base_branch}..{head_branch}",
                unified=0,  # No context lines
            )
        except GitCommandError as e:
            raise ValueError(f"Failed to compare branches: {e}") from e

        # Parse diff output
        files = self._parse_diff(diff_output)

        # Count commits
        try:
            commit_count = len(list(repo.iter_commits(f"{base_branch}..{head_branch}")))
        except GitCommandError:
            commit_count = 0

        return {
            "files": files,
            "commits": commit_count,
        }

    def _parse_diff(self, diff_output: str) -> List[Dict[str, str]]:
        """
        Parse git diff output into file changes.

        Args:
            diff_output: Raw git diff output

        Returns:
            List of file change dictionaries
        """
        files = []
        current_file = None
        current_patch = []

        for line in diff_output.splitlines():
            if line.startswith("diff --git"):
                # Save previous file
                if current_file:
                    files.append(
                        {
                            "filename": current_file,
                            "patch": "\n".join(current_patch),
                            "status": "modified",  # Simplified status detection
                        }
                    )

                # Extract filename (a/path/to/file -> path/to/file)
                parts = line.split()
                if len(parts) >= 3:
                    current_file = parts[2].lstrip("a/")
                    current_patch = [line]
            elif current_file:
                current_patch.append(line)

        # Save last file
        if current_file:
            files.append(
                {
                    "filename": current_file,
                    "patch": "\n".join(current_patch),
                    "status": "modified",
                }
            )

        return files

    # ============ Pull Request Operations ============

    def list_pull_requests(
        self, repo_name: str, state: str = "open", limit: int = 10
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError("LocalProvider does not support pull requests")

    def get_pull_request(
        self, repo_name: str, pr_number: int, include_diff: bool = False
    ) -> Dict[str, Any]:
        raise NotImplementedError("LocalProvider does not support pull requests")

    def create_pull_request(
        self,
        repo_name: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str,
        reviewers: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError("LocalProvider does not support pull requests")

    def add_pull_request_comment(
        self,
        repo_name: str,
        pr_number: int,
        body: str,
        commit_id: Optional[str] = None,
        path: Optional[str] = None,
        line: Optional[int] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "LocalProvider does not support pull request comments"
        )

    def create_pull_request_review(
        self,
        repo_name: str,
        pr_number: int,
        body: str,
        event: str,
        comments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError("LocalProvider does not support pull request reviews")

    # ============ Issue Operations ============

    def list_issues(
        self, repo_name: str, state: str = "open", limit: int = 10
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError("LocalProvider does not support issues")

    def get_issue(self, repo_name: str, issue_number: int) -> Dict[str, Any]:
        raise NotImplementedError("LocalProvider does not support issues")

    def create_issue(
        self, repo_name: str, title: str, body: str, labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Issues are not supported for local repositories."""
        raise NotImplementedError("Issues are not supported for local repositories")

    # ============ File Modification ============

    def create_or_update_file(
        self,
        repo_name: str,
        file_path: str,
        content: str,
        message: str,
        branch: str,
        author_name: Optional[str] = None,
        author_email: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create or update a file and commit the change.

        Args:
            repo_name: Path to the local repository
            file_path: Relative path to file
            content: File content
            message: Commit message
            branch: Branch name to commit to
            author_name: Optional commit author name
            author_email: Optional commit author email

        Returns:
            Dictionary with commit information
        """
        repo = self._get_repo(repo_name)
        repo_path = os.path.abspath(
            os.path.expanduser(repo_name or self.default_repo_path)
        )

        # Checkout target branch
        if branch not in repo.heads:
            raise ValueError(f"Branch not found: {branch}")

        repo.heads[branch].checkout()

        # Write file
        full_file_path = os.path.join(repo_path, file_path)
        os.makedirs(os.path.dirname(full_file_path), exist_ok=True)

        with open(full_file_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Stage file
        repo.index.add([file_path])

        # Create commit
        from git import Actor

        if author_name and author_email:
            author = Actor(author_name, author_email)
            commit = repo.index.commit(message, author=author)
        else:
            commit = repo.index.commit(message)

        return {
            "commit_sha": commit.hexsha,
            "message": message,
            "author": commit.author.name,
        }

    # ============ User/Organization Operations ============

    def list_user_repositories(
        self, username: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List user repositories.

        For local provider, this could list all git repos in a directory,
        but is not currently implemented.
        """
        raise NotImplementedError(
            "Repository listing is not supported for local provider. "
            "Future enhancement: could scan directory for git repositories."
        )

    def get_user_organizations(
        self, username: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Organizations don't apply to local repositories."""
        raise NotImplementedError(
            "Organizations are not supported for local repositories."
        )

    # ============ Provider Metadata ============

    def get_provider_name(self) -> str:
        """Return provider name."""
        return "local"

    def get_api_base_url(self) -> str:
        """Return base URL (file:// URL for local repos)."""
        if self.default_repo_path:
            return f"file://{self.default_repo_path}"
        return "file://local"

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Rate limits don't apply to local repositories."""
        return {
            "limit": None,
            "remaining": None,
            "reset_at": None,
        }

"""
Code Provider Wrapper with Repository Manager Integration

This wrapper enhances ICodeProvider implementations by using local repository
copies managed by IRepoManager when available, falling back to the wrapped
provider for operations that require remote access or when local copies don't exist.

Uses git worktree to manage multiple branches/commits efficiently.
"""

import os
from typing import List, Dict, Any, Optional

from app.modules.code_provider.base.code_provider_interface import (
    ICodeProvider,
    AuthMethod,
)
from app.modules.repo_manager.repo_manager_interface import IRepoManager
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class RepoManagerCodeProviderWrapper(ICodeProvider):
    """
    Wrapper around ICodeProvider that uses local repository copies when available.

    This wrapper:
    - Overrides get_file_content and get_repository_structure to use local copies
    - Uses git worktree to handle different branches/commits
    - Falls back to wrapped provider when local copy doesn't exist
    - Delegates all other methods to the wrapped provider
    """

    def __init__(self, provider: ICodeProvider, repo_manager: IRepoManager):
        """
        Initialize the wrapper.

        Args:
            provider: The underlying ICodeProvider instance to wrap
            repo_manager: The IRepoManager instance for managing local copies
        """
        self._provider = provider
        self._repo_manager = repo_manager

    # ============ Delegate all methods to wrapped provider ============

    def authenticate(self, credentials: Dict[str, Any], method: AuthMethod) -> Any:
        """Delegate to wrapped provider."""
        return self._provider.authenticate(credentials, method)

    def get_supported_auth_methods(self) -> List[AuthMethod]:
        """Delegate to wrapped provider."""
        return self._provider.get_supported_auth_methods()

    def get_repository(self, repo_name: str) -> Dict[str, Any]:
        """Delegate to wrapped provider."""
        return self._provider.get_repository(repo_name)

    def check_repository_access(self, repo_name: str) -> bool:
        """Delegate to wrapped provider."""
        return self._provider.check_repository_access(repo_name)

    def list_branches(self, repo_name: str) -> List[str]:
        """Delegate to wrapped provider."""
        return self._provider.list_branches(repo_name)

    def get_branch(self, repo_name: str, branch_name: str) -> Dict[str, Any]:
        """Delegate to wrapped provider."""
        return self._provider.get_branch(repo_name, branch_name)

    def create_branch(
        self, repo_name: str, branch_name: str, base_branch: str
    ) -> Dict[str, Any]:
        """Delegate to wrapped provider."""
        return self._provider.create_branch(repo_name, branch_name, base_branch)

    def compare_branches(
        self, repo_name: str, base_branch: str, head_branch: str
    ) -> Dict[str, Any]:
        """Delegate to wrapped provider."""
        return self._provider.compare_branches(repo_name, base_branch, head_branch)

    def list_pull_requests(
        self, repo_name: str, state: str = "open", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Delegate to wrapped provider."""
        return self._provider.list_pull_requests(repo_name, state, limit)

    def get_pull_request(
        self, repo_name: str, pr_number: int, include_diff: bool = False
    ) -> Dict[str, Any]:
        """Delegate to wrapped provider."""
        return self._provider.get_pull_request(repo_name, pr_number, include_diff)

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
        """Delegate to wrapped provider."""
        return self._provider.create_pull_request(
            repo_name, title, body, head_branch, base_branch, reviewers, labels
        )

    def add_pull_request_comment(
        self,
        repo_name: str,
        pr_number: int,
        body: str,
        commit_id: Optional[str] = None,
        path: Optional[str] = None,
        line: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Delegate to wrapped provider."""
        return self._provider.add_pull_request_comment(
            repo_name, pr_number, body, commit_id, path, line
        )

    def create_pull_request_review(
        self,
        repo_name: str,
        pr_number: int,
        body: str,
        event: str,
        comments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Delegate to wrapped provider."""
        return self._provider.create_pull_request_review(
            repo_name, pr_number, body, event, comments
        )

    def list_issues(
        self, repo_name: str, state: str = "open", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Delegate to wrapped provider."""
        return self._provider.list_issues(repo_name, state, limit)

    def get_issue(self, repo_name: str, issue_number: int) -> Dict[str, Any]:
        """Delegate to wrapped provider."""
        return self._provider.get_issue(repo_name, issue_number)

    def create_issue(
        self, repo_name: str, title: str, body: str, labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Delegate to wrapped provider."""
        return self._provider.create_issue(repo_name, title, body, labels)

    def create_or_update_file(
        self,
        repo_name: str,
        file_path: str,
        content: str,
        commit_message: str,
        branch: str,
        author_name: Optional[str] = None,
        author_email: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delegate to wrapped provider."""
        return self._provider.create_or_update_file(
            repo_name,
            file_path,
            content,
            commit_message,
            branch,
            author_name,
            author_email,
        )

    def list_user_repositories(
        self, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Delegate to wrapped provider."""
        return self._provider.list_user_repositories(user_id)

    def get_user_organizations(self) -> List[Dict[str, Any]]:
        """Delegate to wrapped provider."""
        return self._provider.get_user_organizations()

    def get_provider_name(self) -> str:
        """Delegate to wrapped provider."""
        return self._provider.get_provider_name()

    def get_api_base_url(self) -> str:
        """Delegate to wrapped provider."""
        return self._provider.get_api_base_url()

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Delegate to wrapped provider."""
        return self._provider.get_rate_limit_info()

    def get_client(self) -> Optional[Any]:
        """
        Get the underlying provider client by delegating to wrapped provider.

        Uses the interface method to respect abstraction.
        """
        return self._provider.get_client()

    @property
    def client(self):
        """
        Property for backward compatibility with code that directly accesses provider.client.

        Delegates to get_client() method which uses the interface abstraction.
        """
        return self.get_client()

    # ============ Override methods to use local copies ============

    def get_file_content(
        self,
        repo_name: str,
        file_path: str,
        ref: Optional[str] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        """
        Get file content from local copy if available, otherwise fallback to provider.

        Uses git worktree to access the correct branch/commit.
        """
        # Try to get local copy path
        worktree_path = self._get_worktree_path(repo_name, ref)
        if worktree_path:
            try:
                # Update last accessed time
                self._update_last_accessed(repo_name, ref)

                # Read file from local filesystem
                full_path = os.path.join(worktree_path, file_path)
                if not os.path.exists(full_path):
                    # File doesn't exist in worktree - check if underlying provider is LocalProvider
                    # If so, try to read from base repo path directly (avoid git show which blocks)
                    from app.modules.code_provider.local_repo.local_provider import (
                        LocalProvider,
                    )

                    if isinstance(self._provider, LocalProvider):
                        # Try to read from base repo path directly without using git show
                        base_repo_path = self._repo_manager.get_repo_path(repo_name)
                        if base_repo_path:
                            base_file_path = os.path.join(base_repo_path, file_path)
                            if os.path.exists(base_file_path):
                                logger.info(
                                    f"[REPO_MANAGER] File not in worktree, reading from base repo: "
                                    f"{base_file_path} for {repo_name}@{ref}"
                                )
                                with open(
                                    base_file_path,
                                    "r",
                                    encoding="utf-8",
                                    errors="replace",
                                ) as f:
                                    content = f.read()
                                # Apply line range if specified
                                if start_line is not None or end_line is not None:
                                    lines = content.split("\n")
                                    start = (
                                        (start_line - 1)
                                        if start_line is not None
                                        else 0
                                    )
                                    end = (
                                        end_line if end_line is not None else len(lines)
                                    )
                                    content = "\n".join(lines[start:end])
                                return content

                    # File doesn't exist - raise FileNotFoundError instead of falling back to git show
                    logger.warning(
                        f"[REPO_MANAGER] File {file_path} not found in worktree at {full_path} "
                        f"or base repo for {repo_name}@{ref}. File may not exist at this ref."
                    )
                    raise FileNotFoundError(
                        f"File '{file_path}' not found in worktree or base repo for {repo_name}@{ref}"
                    )

                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()

                # Apply line range if specified
                if start_line is not None or end_line is not None:
                    lines = content.split("\n")
                    start = (start_line - 1) if start_line is not None else 0
                    end = end_line if end_line is not None else len(lines)
                    content = "\n".join(lines[start:end])

                logger.info(
                    f"[REPO_MANAGER] Retrieved file content from local copy: "
                    f"{repo_name}/{file_path}@{ref} (path: {full_path})"
                )
                return content

            except FileNotFoundError:
                # Re-raise FileNotFoundError - don't fall back to git show which blocks
                raise
            except Exception as e:
                # For other exceptions, check if provider is LocalProvider
                # If so, don't fall back to git show - re-raise instead
                from app.modules.code_provider.local_repo.local_provider import (
                    LocalProvider,
                )

                if isinstance(self._provider, LocalProvider):
                    logger.error(
                        f"[REPO_MANAGER] Error reading file from worktree: {e}, "
                        f"for {repo_name}/{file_path}@{ref}. Not falling back to git show to avoid blocking."
                    )
                    raise
                # For non-LocalProvider, it's safe to fall back (e.g., GitHub API)
                logger.warning(
                    f"[REPO_MANAGER] Error reading file from local copy: {e}, "
                    f"falling back to provider API for {repo_name}/{file_path}@{ref}"
                )
                logger.info(
                    f"[PROVIDER_API] Fetching file content: {repo_name}/{file_path}@{ref}"
                )
                return self._provider.get_file_content(
                    repo_name, file_path, ref, start_line, end_line
                )

        # Fallback to provider
        logger.info(
            f"[REPO_MANAGER] No local copy available for {repo_name}@{ref}, "
            f"using provider API"
        )
        logger.info(
            f"[PROVIDER_API] Fetching file content: {repo_name}/{file_path}@{ref}"
        )
        return self._provider.get_file_content(
            repo_name, file_path, ref, start_line, end_line
        )

    def get_repository_structure(
        self,
        repo_name: str,
        path: str = "",
        ref: Optional[str] = None,
        max_depth: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Get repository structure from local copy if available, otherwise fallback to provider.

        Uses git worktree to access the correct branch/commit.
        """
        # Try to get local copy path
        worktree_path = self._get_worktree_path(repo_name, ref)
        if worktree_path:
            try:
                # Update last accessed time
                self._update_last_accessed(repo_name, ref)

                # Build structure from local filesystem
                structure = self._build_structure_from_filesystem(
                    worktree_path, path, max_depth
                )

                logger.info(
                    f"[REPO_MANAGER] Retrieved repository structure from local copy: "
                    f"{repo_name}@{ref} (path: {worktree_path}, depth: {max_depth})"
                )
                return structure

            except Exception as e:
                logger.warning(
                    f"[REPO_MANAGER] Error reading structure from local copy: {e}, "
                    f"falling back to provider API for {repo_name}@{ref}"
                )
                logger.info(
                    f"[PROVIDER_API] Fetching repository structure: {repo_name}@{ref} (path: {path})"
                )
                return self._provider.get_repository_structure(
                    repo_name, path, ref, max_depth
                )

        # Fallback to provider
        logger.info(
            f"[REPO_MANAGER] No local copy available for {repo_name}@{ref}, "
            f"using provider API"
        )
        logger.info(
            f"[PROVIDER_API] Fetching repository structure: {repo_name}@{ref} (path: {path})"
        )
        return self._provider.get_repository_structure(repo_name, path, ref, max_depth)

    # ============ Helper methods ============

    def _get_worktree_path(
        self, repo_name: str, ref: Optional[str] = None
    ) -> Optional[str]:
        """
        Get the worktree path for a repository and ref (branch/commit).

        First checks if a worktree is already registered in Redis for the specific ref.
        If not, tries to find the base repo and create/access a worktree.

        Args:
            repo_name: Repository name
            ref: Branch name or commit SHA

        Returns:
            Path to the worktree, or None if repo is not available locally
        """
        # Parse ref to determine if it's a branch or commit
        branch = None
        commit_id = None
        if ref:
            is_commit = len(ref) >= 7 and all(
                c in "0123456789abcdefABCDEF" for c in ref
            )
            if is_commit:
                commit_id = ref
            else:
                branch = ref

        logger.debug(
            f"[REPO_MANAGER] Looking up worktree for {repo_name}@{ref} "
            f"(branch={branch}, commit_id={commit_id})"
        )

        # Try multiple lookup strategies since repo might be registered with different combinations
        # 1. Try with commit_id only (most specific for commits)
        if commit_id:
            registered_path = self._repo_manager.get_repo_path(
                repo_name, commit_id=commit_id
            )
            if registered_path and os.path.exists(registered_path):
                logger.info(
                    f"[REPO_MANAGER] Found registered worktree for {repo_name}@commit:{commit_id} "
                    f"at {registered_path}"
                )
                return registered_path
            logger.debug(
                f"[REPO_MANAGER] No worktree found for {repo_name}@commit:{commit_id}"
            )

        # 2. Try with branch only
        if branch:
            registered_path = self._repo_manager.get_repo_path(repo_name, branch=branch)
            if registered_path and os.path.exists(registered_path):
                logger.info(
                    f"[REPO_MANAGER] Found registered worktree for {repo_name}@branch:{branch} "
                    f"at {registered_path}"
                )
                return registered_path
            logger.debug(
                f"[REPO_MANAGER] No worktree found for {repo_name}@branch:{branch}"
            )

        # 3. Try with both branch and commit_id (in case it was registered that way)
        if branch and commit_id:
            registered_path = self._repo_manager.get_repo_path(
                repo_name, branch=branch, commit_id=commit_id
            )
            if registered_path and os.path.exists(registered_path):
                logger.info(
                    f"[REPO_MANAGER] Found registered worktree for {repo_name}@branch:{branch}:commit:{commit_id} "
                    f"at {registered_path}"
                )
                return registered_path
            logger.debug(
                f"[REPO_MANAGER] No worktree found for {repo_name}@branch:{branch}:commit:{commit_id}"
            )

        # 4. Try searching all repos for this repo_name to find any matching commit_id
        if commit_id:
            all_repos = self._repo_manager.list_available_repos()
            for repo_info in all_repos:
                if (
                    repo_info.get("repo_name") == repo_name
                    and repo_info.get("commit_id") == commit_id
                ):
                    found_path = repo_info.get("local_path")
                    if found_path and os.path.exists(found_path):
                        logger.info(
                            f"[REPO_MANAGER] Found registered worktree via search for {repo_name}@commit:{commit_id} "
                            f"at {found_path}"
                        )
                        return found_path

        # If no ref specified, check for base repo
        if not ref:
            base_path = self._repo_manager.get_repo_path(repo_name)
            if base_path and os.path.exists(base_path):
                logger.info(
                    f"[REPO_MANAGER] Found base repo for {repo_name} at {base_path}"
                )
                return base_path
            logger.debug(f"[REPO_MANAGER] No base repo found for {repo_name}")
            return None

        # Try to find base repo and create worktree
        # Check if any version of the repo exists (without ref)
        if not self._repo_manager.is_repo_available(repo_name):
            logger.debug(
                f"[REPO_MANAGER] Repository {repo_name} not available in local storage"
            )
            return None

        # Get base repo path (try without ref to find the base)
        base_path = self._repo_manager.get_repo_path(repo_name)
        if not base_path or not os.path.exists(base_path):
            logger.debug(
                f"[REPO_MANAGER] No base path found for repository {repo_name}"
            )
            return None

        # Try to create/access worktree from base repo
        try:
            from app.modules.code_provider.git_safe import safe_git_repo_operation

            def _setup_worktree(repo):
                # Get or create worktree for this ref
                return self._ensure_worktree(repo, ref, commit_id is not None)

            # Use a short timeout to avoid blocking the caller
            # This is often called from within an outer timeout (e.g., 20s in _get_current_content)
            # so we need a shorter timeout here to prevent orphaned threads
            worktree_path = safe_git_repo_operation(
                base_path,
                _setup_worktree,
                max_retries=1,  # Reduced retries for faster failure
                timeout=10.0,  # Short timeout - worktree setup should be fast
                operation_name=f"setup_worktree({repo_name}@{ref})",
            )
            logger.debug(
                f"[REPO_MANAGER] Created/accessed worktree for {repo_name}@{ref} at {worktree_path}"
            )
            return worktree_path

        except Exception as e:
            logger.warning(
                f"[REPO_MANAGER] Error setting up worktree for {repo_name}@{ref}: {e}"
            )
            return None

    def _ensure_worktree(self, repo: "Repo", ref: str, is_commit: bool) -> str:
        """
        Ensure a worktree exists for the given ref.

        Args:
            repo: Git repository object
            ref: Branch name or commit SHA
            is_commit: Whether ref is a commit SHA

        Returns:
            Path to the worktree
        """
        from git import Repo, GitCommandError

        # Generate worktree path based on ref
        base_path = repo.working_tree_dir or repo.git_dir
        worktree_dir = os.path.join(
            os.path.dirname(base_path), "worktrees", ref.replace("/", "_")
        )

        # Check if worktree already exists
        if os.path.exists(worktree_dir):
            try:
                # Verify it's a valid worktree
                worktree_repo = Repo(worktree_dir)
                if is_commit:
                    current_commit = worktree_repo.head.commit.hexsha
                    if current_commit.startswith(ref):
                        return worktree_dir
                else:
                    # Check if branch matches
                    if worktree_repo.active_branch.name == ref:
                        return worktree_dir
            except Exception:
                # Worktree exists but is invalid, remove it
                logger.warning(f"Invalid worktree at {worktree_dir}, removing")
                try:
                    repo.git.worktree("remove", worktree_dir, force=True)
                except Exception:
                    pass

        # Create new worktree
        try:
            os.makedirs(os.path.dirname(worktree_dir), exist_ok=True)

            if is_commit:
                # Checkout specific commit
                repo.git.worktree("add", worktree_dir, ref, "--detach")
            else:
                # Checkout branch (create if doesn't exist)
                try:
                    repo.git.worktree("add", worktree_dir, ref)
                except GitCommandError:
                    # Branch might not exist locally, try to fetch and create
                    # First check if origin remote exists
                    remotes = [r.name for r in repo.remotes]
                    if "origin" in remotes:
                        try:
                            # Try to fetch the branch from origin
                            repo.git.fetch("origin", f"{ref}:{ref}")
                            repo.git.worktree("add", worktree_dir, ref)
                        except GitCommandError as fetch_error:
                            # Fetch failed, try local branch creation
                            logger.warning(
                                f"Failed to fetch {ref} from origin: {fetch_error}. "
                                f"Trying to create from local branches."
                            )
                            # Check if branch exists locally
                            local_branches = [branch.name for branch in repo.heads]
                            if ref in local_branches:
                                # Branch exists locally, try worktree again
                                repo.git.worktree("add", worktree_dir, ref)
                            else:
                                # Branch doesn't exist, create it from HEAD without checking out
                                # Use worktree add with -b flag to create branch in worktree
                                current_head = repo.head.commit.hexsha
                                repo.git.worktree(
                                    "add", "-b", ref, worktree_dir, current_head
                                )
                    else:
                        # No origin remote, work with local branches only
                        logger.info(
                            f"No 'origin' remote found, working with local branches only for {ref}"
                        )
                        # Check if branch exists locally
                        local_branches = [branch.name for branch in repo.heads]
                        if ref in local_branches:
                            # Branch exists locally, create worktree
                            repo.git.worktree("add", worktree_dir, ref)
                        else:
                            # Branch doesn't exist, create it in the worktree from current HEAD
                            # This creates the branch in the worktree without affecting the main repo
                            current_head = repo.head.commit.hexsha
                            repo.git.worktree(
                                "add", "-b", ref, worktree_dir, current_head
                            )
                            logger.info(
                                f"Created new branch '{ref}' in worktree from HEAD ({current_head[:8]})"
                            )

            logger.info(f"Created worktree for {ref} at {worktree_dir}")
            return worktree_dir

        except Exception as e:
            logger.error(f"Failed to create worktree for {ref}: {e}")
            # Fallback to base repo
            return repo.working_tree_dir or repo.git_dir

    def _build_structure_from_filesystem(
        self, base_path: str, path: str, max_depth: int
    ) -> List[Dict[str, Any]]:
        """
        Build repository structure from local filesystem.

        Args:
            base_path: Base path of the repository
            path: Relative path within repository
            max_depth: Maximum depth to traverse

        Returns:
            List of file/directory dictionaries
        """
        structure = []
        full_path = os.path.join(base_path, path) if path else base_path

        if not os.path.exists(full_path):
            return structure

        try:
            for item in os.listdir(full_path):
                # Skip .git directory only
                if item == ".git":
                    continue

                item_path = os.path.join(full_path, item)
                rel_path = os.path.join(path, item) if path else item

                item_info = {
                    "name": item,
                    "path": rel_path,
                    "type": "directory" if os.path.isdir(item_path) else "file",
                }

                if os.path.isdir(item_path):
                    # Recursively get subdirectory structure if within max_depth
                    if max_depth > 1:
                        item_info["children"] = self._build_structure_from_filesystem(
                            base_path, rel_path, max_depth - 1
                        )
                else:
                    # Add file size
                    try:
                        item_info["size"] = os.path.getsize(item_path)
                    except Exception:
                        item_info["size"] = 0

                structure.append(item_info)

        except Exception as e:
            logger.warning(f"Error building structure from {full_path}: {e}")

        return structure

    def _update_last_accessed(self, repo_name: str, ref: Optional[str] = None) -> None:
        """
        Update last accessed time in repo manager.

        Args:
            repo_name: Repository name
            ref: Branch or commit reference
        """
        try:
            # Parse ref to determine branch vs commit
            branch = None
            commit_id = None

            if ref:
                # Heuristic: if it looks like a commit SHA, treat as commit
                if len(ref) >= 7 and all(c in "0123456789abcdefABCDEF" for c in ref):
                    commit_id = ref
                else:
                    branch = ref

            self._repo_manager.update_last_accessed(
                repo_name, branch=branch, commit_id=commit_id
            )
        except Exception as e:
            logger.debug(f"Failed to update last accessed time: {e}")

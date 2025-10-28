import logging
import os
from typing import Any, Dict, List, Optional

from git import InvalidGitRepositoryError, NoSuchPathError, Repo

from app.modules.code_provider.base.code_provider_interface import (
    AuthMethod,
    ICodeProvider,
)

logger = logging.getLogger(__name__)


class LocalProvider(ICodeProvider):
    """Filesystem-backed implementation limited to branch enumeration."""

    def __init__(self, default_repo_path: Optional[str] = None):
        self.default_repo_path = (
            os.path.abspath(os.path.expanduser(default_repo_path))
            if default_repo_path
            else None
        )

    # ============ Authentication ============

    def authenticate(self, credentials: Dict[str, Any], method: AuthMethod) -> Any:
        """Authentication is not required for local repositories."""
        logger.debug("LocalProvider.authenticate called; no action taken for local repos")
        return None

    def get_supported_auth_methods(self) -> List[AuthMethod]:
        return []

    # ============ Repository Helpers ============

    def _get_repo(self, repo_name: Optional[str]) -> Repo:
        path = repo_name or self.default_repo_path
        if not path:
            raise ValueError("Repository path is required for local provider operations")

        expanded_path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(expanded_path):
            raise FileNotFoundError(f"Local repository at {expanded_path} not found")

        try:
            return Repo(expanded_path)
        except (InvalidGitRepositoryError, NoSuchPathError) as exc:
            raise ValueError(f"Path {expanded_path} is not a git repository") from exc

    # ============ Repository Operations ============

    def get_repository(self, repo_name: str) -> Dict[str, Any]:
        raise NotImplementedError("LocalProvider does not support repository metadata")

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
        raise NotImplementedError("LocalProvider does not support file content access")

    def get_repository_structure(
        self,
        repo_name: str,
        path: str = "",
        ref: Optional[str] = None,
        max_depth: int = 4,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError("LocalProvider does not support repository structure")

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
        raise NotImplementedError("LocalProvider does not support branch metadata")

    def create_branch(
        self, repo_name: str, branch_name: str, base_branch: str
    ) -> Dict[str, Any]:
        raise NotImplementedError("LocalProvider does not support branch creation")

    def compare_branches(
        self, repo_name: str, base_branch: str, head_branch: str
    ) -> Dict[str, Any]:
        raise NotImplementedError("LocalProvider does not support branch comparison")

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
        raise NotImplementedError("LocalProvider does not support pull request comments")

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

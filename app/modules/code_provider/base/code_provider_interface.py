from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class AuthMethod(str, Enum):
    """Supported authentication methods across providers."""

    PERSONAL_ACCESS_TOKEN = "pat"
    OAUTH_TOKEN = "oauth"
    APP_INSTALLATION = "app"
    BASIC_AUTH = "basic"


class ICodeProvider(ABC):
    """
    Abstract interface for code provider implementations.
    All code providers (GitHub, GitBucket, GitLab, Bitbucket) must implement this.
    """

    # ============ Authentication ============

    @abstractmethod
    def authenticate(self, credentials: dict[str, Any], method: AuthMethod) -> Any:
        """
        Authenticate with the code provider.

        Args:
            credentials: Dict containing auth credentials
                - For PAT: {"token": "your_token"}
                - For OAuth: {"access_token": "user_token"}
                - For App: {"app_id": "...", "private_key": "...", "installation_id": "..."}
                - For Basic: {"username": "...", "password": "..."}
            method: Authentication method to use

        Returns:
            Authenticated client instance (provider-specific)
        """
        pass

    @abstractmethod
    def get_supported_auth_methods(self) -> list[AuthMethod]:
        """Return list of supported authentication methods for this provider."""
        pass

    # ============ Repository Operations ============

    @abstractmethod
    def get_repository(self, repo_name: str) -> dict[str, Any]:
        """
        Get repository details.

        Returns:
            Dict with: id, name, full_name, owner, default_branch, private, url
        """
        pass

    @abstractmethod
    def check_repository_access(self, repo_name: str) -> bool:
        """Check if repository exists and is accessible with current auth."""
        pass

    # ============ Content Operations ============

    @abstractmethod
    def get_file_content(
        self,
        repo_name: str,
        file_path: str,
        ref: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """Get file content from repository (decoded as string)."""
        pass

    @abstractmethod
    def get_repository_structure(
        self,
        repo_name: str,
        path: str = "",
        ref: str | None = None,
        max_depth: int = 4,
    ) -> list[dict[str, Any]]:
        """Get repository directory structure recursively."""
        pass

    # ============ Branch Operations ============

    @abstractmethod
    def list_branches(self, repo_name: str) -> list[str]:
        """List all branches (default branch first)."""
        pass

    @abstractmethod
    def get_branch(self, repo_name: str, branch_name: str) -> dict[str, Any]:
        """Get branch details (name, commit_sha, protected)."""
        pass

    @abstractmethod
    def create_branch(
        self, repo_name: str, branch_name: str, base_branch: str
    ) -> dict[str, Any]:
        """Create a new branch from base branch."""
        pass

    @abstractmethod
    def compare_branches(
        self, repo_name: str, base_branch: str, head_branch: str
    ) -> dict[str, Any]:
        """
        Compare two branches and return file changes with patches.

        Args:
            repo_name: Repository name (e.g., 'owner/repo')
            base_branch: Base branch to compare from
            head_branch: Head branch to compare to

        Returns:
            Dict with:
                - files: List of changed files with patches
                - commits: Number of commits different
                Example: {
                    'files': [
                        {'filename': 'path/to/file.py', 'patch': '@@ ...', 'status': 'modified'},
                        ...
                    ],
                    'commits': 2
                }
        """
        pass

    # ============ Pull Request Operations ============

    @abstractmethod
    def list_pull_requests(
        self, repo_name: str, state: str = "open", limit: int = 10
    ) -> list[dict[str, Any]]:
        """List pull requests."""
        pass

    @abstractmethod
    def get_pull_request(
        self, repo_name: str, pr_number: int, include_diff: bool = False
    ) -> dict[str, Any]:
        """Get pull request details with optional diff."""
        pass

    @abstractmethod
    def get_pull_request_commits(
        self, repo_name: str, pr_number: int
    ) -> list[dict[str, Any]]:
        """Get pull request commits."""
        pass

    @abstractmethod
    def get_pull_request_review_comments(
        self, repo_name: str, pr_number: int, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get pull request review comments."""
        pass

    @abstractmethod
    def get_pull_request_issue_comments(
        self, repo_name: str, pr_number: int, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get pull request issue comments."""
        pass

    @abstractmethod
    def create_pull_request(
        self,
        repo_name: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str,
        reviewers: list[str] | None = None,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a pull request."""
        pass

    @abstractmethod
    def add_pull_request_comment(
        self,
        repo_name: str,
        pr_number: int,
        body: str,
        commit_id: str | None = None,
        path: str | None = None,
        line: int | None = None,
    ) -> dict[str, Any]:
        """Add comment to pull request (general or inline)."""
        pass

    @abstractmethod
    def create_pull_request_review(
        self,
        repo_name: str,
        pr_number: int,
        body: str,
        event: str,  # "COMMENT", "APPROVE", "REQUEST_CHANGES"
        comments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create a pull request review with optional inline comments."""
        pass

    # ============ Issue Operations ============

    @abstractmethod
    def list_issues(
        self, repo_name: str, state: str = "open", limit: int = 10
    ) -> list[dict[str, Any]]:
        """List issues in repository."""
        pass

    @abstractmethod
    def get_issue(self, repo_name: str, issue_number: int) -> dict[str, Any]:
        """Get issue details."""
        pass

    @abstractmethod
    def create_issue(
        self, repo_name: str, title: str, body: str, labels: list[str] | None = None
    ) -> dict[str, Any]:
        """Create an issue."""
        pass

    # ============ File Modification Operations ============

    @abstractmethod
    def create_or_update_file(
        self,
        repo_name: str,
        file_path: str,
        content: str,
        commit_message: str,
        branch: str,
        author_name: str | None = None,
        author_email: str | None = None,
    ) -> dict[str, Any]:
        """Create or update a file in repository."""
        pass

    # ============ User/Organization Operations ============

    @abstractmethod
    def list_user_repositories(
        self, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        """List repositories accessible to authenticated user."""
        pass

    @abstractmethod
    def get_user_organizations(self) -> list[dict[str, Any]]:
        """Get organizations for authenticated user."""
        pass

    # ============ Provider Metadata ============

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider name (e.g., 'github', 'gitbucket', 'gitlab')."""
        pass

    @abstractmethod
    def get_api_base_url(self) -> str:
        """Return base API URL for this provider instance."""
        pass

    @abstractmethod
    def get_rate_limit_info(self) -> dict[str, Any]:
        """
        Get current rate limit information.

        Returns:
            Dict with: limit, remaining, reset_at
        """
        pass

    def get_client(self) -> Any | None:
        """
        Get the underlying provider client (e.g., PyGithub client).

        This is an optional method for backward compatibility with code that needs
        direct access to provider-specific clients. Not all providers have clients
        (e.g., LocalProvider returns None).

        Returns:
            Provider-specific client instance, or None if not available
        """
        # Default implementation returns None
        # Providers that have clients should override this
        return getattr(self, "client", None)

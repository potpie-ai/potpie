import logging
from typing import List, Dict, Any, Optional
import chardet
from github import Github
from github.Auth import AppAuth
from github.GithubException import GithubException

from app.modules.code_provider.base.code_provider_interface import (
    ICodeProvider,
    AuthMethod,
)

logger = logging.getLogger(__name__)


class GitHubProvider(ICodeProvider):
    """GitHub implementation of ICodeProvider interface."""

    def __init__(self, base_url: str = "https://api.github.com"):
        self.base_url = base_url
        self.client: Optional[Github] = None
        self.auth_method: Optional[AuthMethod] = None
        self._token_pool: List[str] = []

    # ============ Authentication ============

    def authenticate(self, credentials: Dict[str, Any], method: AuthMethod) -> Github:
        """Authenticate with GitHub."""
        self.auth_method = method

        if method == AuthMethod.PERSONAL_ACCESS_TOKEN:
            token = credentials.get("token")
            if not token:
                raise ValueError("PAT authentication requires 'token' in credentials")
            self.client = Github(token, base_url=self.base_url)

        elif method == AuthMethod.OAUTH_TOKEN:
            access_token = credentials.get("access_token")
            if not access_token:
                raise ValueError("OAuth authentication requires 'access_token'")
            self.client = Github(access_token, base_url=self.base_url)

        elif method == AuthMethod.APP_INSTALLATION:
            app_id = credentials.get("app_id")
            private_key = credentials.get("private_key")
            installation_id = credentials.get("installation_id")

            if not all([app_id, private_key]):
                raise ValueError("App auth requires app_id and private_key")

            # Format private key
            if not private_key.startswith("-----BEGIN"):
                private_key = f"-----BEGIN RSA PRIVATE KEY-----\n{private_key}\n-----END RSA PRIVATE KEY-----\n"

            auth = AppAuth(app_id=app_id, private_key=private_key)

            if installation_id:
                app_auth = auth.get_installation_auth(installation_id)
            else:
                # Use JWT for app-level operations
                app_auth = auth

            self.client = Github(auth=app_auth, base_url=self.base_url)

        else:
            raise ValueError(f"Unsupported authentication method: {method}")

        return self.client

    def get_supported_auth_methods(self) -> List[AuthMethod]:
        return [
            AuthMethod.PERSONAL_ACCESS_TOKEN,
            AuthMethod.OAUTH_TOKEN,
            AuthMethod.APP_INSTALLATION,
        ]

    def _ensure_authenticated(self):
        """Ensure client is authenticated."""
        if not self.client:
            raise RuntimeError("Provider not authenticated. Call authenticate() first.")

    # ============ Repository Operations ============

    def get_repository(self, repo_name: str) -> Dict[str, Any]:
        """Get repository details."""
        self._ensure_authenticated()

        try:
            repo = self.client.get_repo(repo_name)
            return {
                "id": repo.id,
                "name": repo.name,
                "full_name": repo.full_name,
                "owner": repo.owner.login,
                "default_branch": repo.default_branch,
                "private": repo.private,
                "url": repo.html_url,
                "description": repo.description,
                "language": repo.language,
            }
        except GithubException as e:
            logger.error(f"Failed to get repository {repo_name}: {e}")
            raise

    def check_repository_access(self, repo_name: str) -> bool:
        """Check repository access."""
        try:
            self.get_repository(repo_name)
            return True
        except:
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
        """Get file content."""
        self._ensure_authenticated()

        repo = self.client.get_repo(repo_name)
        file_contents = repo.get_contents(file_path, ref=ref)

        # Decode content
        content = file_contents.decoded_content
        if isinstance(content, bytes):
            # Try UTF-8 first, fall back to chardet
            try:
                content = content.decode("utf-8")
            except UnicodeDecodeError:
                detected = chardet.detect(content)
                encoding = detected.get("encoding", "utf-8")
                content = content.decode(encoding, errors="ignore")

        # Extract line range if specified
        if start_line is not None or end_line is not None:
            lines = content.splitlines()
            start = (start_line - 1) if start_line else 0
            end = end_line if end_line else len(lines)
            content = "\n".join(lines[start:end])

        return content

    def get_repository_structure(
        self,
        repo_name: str,
        path: str = "",
        ref: Optional[str] = None,
        max_depth: int = 4,
    ) -> List[Dict[str, Any]]:
        """Get repository structure recursively."""
        self._ensure_authenticated()

        try:
            repo = self.client.get_repo(repo_name)
        except GithubException as e:
            logger.error(f"GitHubProvider: Failed to get repo {repo_name}: {e}")
            raise

        def _recurse(current_path: str, depth: int) -> List[Dict[str, Any]]:
            if depth > max_depth:
                return []

            result = []
            try:
                # Don't pass ref parameter if it's None - let PyGithub use default branch
                if ref is not None:
                    contents = repo.get_contents(current_path, ref=ref)
                else:
                    contents = repo.get_contents(current_path)

                # Check if contents is None
                if contents is None:
                    logger.error(
                        f"GitHubProvider: get_contents returned None for path '{current_path}', ref={ref}. "
                        f"This usually means the path doesn't exist or auth failed."
                    )
                    return []

                if not isinstance(contents, list):
                    contents = [contents]

                for item in contents:
                    entry = {
                        "name": item.name,
                        "path": item.path,
                        "type": item.type,
                        "size": item.size,
                        "sha": item.sha,
                    }
                    result.append(entry)

                    # Recurse into directories
                    if item.type == "dir":
                        entry["children"] = _recurse(item.path, depth + 1)

            except GithubException as e:
                logger.warning(
                    f"GitHubProvider: Failed to get contents for {current_path}: {e}"
                )
            except Exception as e:
                logger.error(
                    f"GitHubProvider: Unexpected error getting contents for {current_path}: {e}",
                    exc_info=True,
                )

            return result

        return _recurse(path, 0)

    # ============ Branch Operations ============

    def list_branches(self, repo_name: str) -> List[str]:
        """List branches."""
        self._ensure_authenticated()

        repo = self.client.get_repo(repo_name)
        branches = [branch.name for branch in repo.get_branches()]

        # Put default branch first
        default = repo.default_branch
        if default in branches:
            branches.remove(default)
            branches.insert(0, default)

        return branches

    def get_branch(self, repo_name: str, branch_name: str) -> Dict[str, Any]:
        """Get branch details."""
        self._ensure_authenticated()

        repo = self.client.get_repo(repo_name)
        branch = repo.get_branch(branch_name)

        return {
            "name": branch.name,
            "commit_sha": branch.commit.sha,
            "protected": branch.protected,
        }

    def create_branch(
        self, repo_name: str, branch_name: str, base_branch: str
    ) -> Dict[str, Any]:
        """Create branch."""
        self._ensure_authenticated()

        try:
            repo = self.client.get_repo(repo_name)

            # Get base branch ref
            base_ref = repo.get_git_ref(f"heads/{base_branch}")

            # Check if new branch already exists
            try:
                repo.get_git_ref(f"heads/{branch_name}")
                return {
                    "success": False,
                    "error": f"Branch '{branch_name}' already exists",
                }
            except GithubException as e:
                if e.status != 404:
                    raise

            # Create new branch
            new_ref = repo.create_git_ref(
                ref=f"refs/heads/{branch_name}", sha=base_ref.object.sha
            )

            return {
                "success": True,
                "branch_name": branch_name,
                "commit_sha": new_ref.object.sha,
            }

        except GithubException as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": e.status if hasattr(e, "status") else None,
            }

    def compare_branches(
        self, repo_name: str, base_branch: str, head_branch: str
    ) -> Dict[str, Any]:
        """
        Compare two branches using GitHub's compare API.

        Args:
            repo_name: Repository name (e.g., 'owner/repo')
            base_branch: Base branch to compare from
            head_branch: Head branch to compare to

        Returns:
            Dict with files (list of file changes with patches) and commits count
        """
        self._ensure_authenticated()

        try:
            repo = self.client.get_repo(repo_name)
            comparison = repo.compare(base_branch, head_branch)

            # Extract file changes with patches
            files = []
            for file in comparison.files:
                file_data = {
                    "filename": file.filename,
                    "status": file.status,
                    "additions": file.additions,
                    "deletions": file.deletions,
                    "changes": file.changes,
                }
                if file.patch:
                    file_data["patch"] = file.patch
                files.append(file_data)

            logger.info(
                f"[GITHUB] Compared branches {base_branch}...{head_branch}: {len(files)} files, {comparison.total_commits} commits"
            )

            return {
                "files": files,
                "commits": comparison.total_commits,
            }

        except GithubException as e:
            logger.error(f"[GITHUB] Error comparing branches: {str(e)}")
            raise

    # ============ Pull Request Operations ============

    def list_pull_requests(
        self, repo_name: str, state: str = "open", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """List pull requests."""
        self._ensure_authenticated()

        repo = self.client.get_repo(repo_name)
        pulls = repo.get_pulls(state=state)[:limit]

        return [
            {
                "number": pr.number,
                "title": pr.title,
                "state": pr.state,
                "created_at": pr.created_at.isoformat(),
                "updated_at": pr.updated_at.isoformat(),
                "head_branch": pr.head.ref,
                "base_branch": pr.base.ref,
                "url": pr.html_url,
                "author": pr.user.login,
            }
            for pr in pulls
        ]

    def get_pull_request(
        self, repo_name: str, pr_number: int, include_diff: bool = False
    ) -> Dict[str, Any]:
        """Get pull request details."""
        self._ensure_authenticated()

        repo = self.client.get_repo(repo_name)
        pr = repo.get_pull(pr_number)

        result = {
            "number": pr.number,
            "title": pr.title,
            "body": pr.body,
            "state": pr.state,
            "created_at": pr.created_at.isoformat(),
            "updated_at": pr.updated_at.isoformat(),
            "head_branch": pr.head.ref,
            "base_branch": pr.base.ref,
            "url": pr.html_url,
            "author": pr.user.login,
        }

        if include_diff:
            files = pr.get_files()
            result["files"] = [
                {
                    "filename": f.filename,
                    "status": f.status,
                    "additions": f.additions,
                    "deletions": f.deletions,
                    "patch": f.patch,
                }
                for f in files
            ]

        return result

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
        """Create pull request."""
        self._ensure_authenticated()

        try:
            repo = self.client.get_repo(repo_name)

            # Validate branches exist
            try:
                repo.get_git_ref(f"heads/{head_branch}")
            except GithubException as e:
                return {
                    "success": False,
                    "error": f"Head branch '{head_branch}' not found: {str(e)}",
                }

            try:
                repo.get_git_ref(f"heads/{base_branch}")
            except GithubException as e:
                return {
                    "success": False,
                    "error": f"Base branch '{base_branch}' not found: {str(e)}",
                }

            # Create PR
            pr = repo.create_pull(
                title=title, body=body, head=head_branch, base=base_branch
            )

            # Add reviewers
            if reviewers:
                try:
                    pr.create_review_request(reviewers=reviewers)
                except GithubException as e:
                    logger.warning(f"Error adding reviewers: {e}")

            # Add labels
            if labels:
                try:
                    pr.add_to_labels(*labels)
                except GithubException as e:
                    logger.warning(f"Error adding labels: {e}")

            return {"success": True, "pr_number": pr.number, "url": pr.html_url}

        except GithubException as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": e.status if hasattr(e, "status") else None,
            }

    def add_pull_request_comment(
        self,
        repo_name: str,
        pr_number: int,
        body: str,
        commit_id: Optional[str] = None,
        path: Optional[str] = None,
        line: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Add PR comment."""
        self._ensure_authenticated()

        try:
            repo = self.client.get_repo(repo_name)
            pr = repo.get_pull(pr_number)

            if path and line:
                # Inline comment
                commits = list(pr.get_commits())
                latest_commit = commits[-1]

                comment = pr.create_review_comment(
                    body=body, commit=latest_commit, path=path, line=line
                )
            else:
                # General comment
                comment = pr.create_issue_comment(body)

            return {"success": True, "comment_id": comment.id}

        except GithubException as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": e.status if hasattr(e, "status") else None,
            }

    def create_pull_request_review(
        self,
        repo_name: str,
        pr_number: int,
        body: str,
        event: str,
        comments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create PR review."""
        self._ensure_authenticated()

        try:
            repo = self.client.get_repo(repo_name)
            pr = repo.get_pull(pr_number)

            commits = list(pr.get_commits())
            latest_commit = commits[-1]

            review_comments = []
            if comments:
                for c in comments:
                    review_comments.append(
                        {"path": c["path"], "position": c["line"], "body": c["body"]}
                    )

            review = pr.create_review(
                commit=latest_commit, body=body, event=event, comments=review_comments
            )

            return {"success": True, "review_id": review.id}

        except GithubException as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": e.status if hasattr(e, "status") else None,
            }

    # ============ Issue Operations ============

    def list_issues(
        self, repo_name: str, state: str = "open", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """List issues."""
        self._ensure_authenticated()

        repo = self.client.get_repo(repo_name)
        issues = repo.get_issues(state=state)[:limit]

        return [
            {
                "number": issue.number,
                "title": issue.title,
                "state": issue.state,
                "created_at": issue.created_at.isoformat(),
                "updated_at": issue.updated_at.isoformat(),
                "url": issue.html_url,
                "author": issue.user.login,
            }
            for issue in issues
        ]

    def get_issue(self, repo_name: str, issue_number: int) -> Dict[str, Any]:
        """Get issue details."""
        self._ensure_authenticated()

        repo = self.client.get_repo(repo_name)
        issue = repo.get_issue(issue_number)

        return {
            "number": issue.number,
            "title": issue.title,
            "body": issue.body,
            "state": issue.state,
            "created_at": issue.created_at.isoformat(),
            "updated_at": issue.updated_at.isoformat(),
            "url": issue.html_url,
            "author": issue.user.login,
        }

    def create_issue(
        self, repo_name: str, title: str, body: str, labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create issue."""
        self._ensure_authenticated()

        try:
            repo = self.client.get_repo(repo_name)
            issue = repo.create_issue(title=title, body=body, labels=labels or [])

            return {
                "success": True,
                "issue_number": issue.number,
                "url": issue.html_url,
            }
        except GithubException as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": e.status if hasattr(e, "status") else None,
            }

    # ============ File Modification Operations ============

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
        """Create or update file."""
        self._ensure_authenticated()

        try:
            repo = self.client.get_repo(repo_name)

            # Check if file exists
            try:
                file = repo.get_contents(file_path, ref=branch)
                sha = file.sha
                file_exists = True
            except GithubException as e:
                if e.status == 404:
                    file_exists = False
                    sha = None
                else:
                    raise

            # Prepare commit kwargs
            commit_kwargs = {"message": commit_message}
            if author_name and author_email:
                from github.InputGitAuthor import InputGitAuthor

                commit_kwargs["author"] = InputGitAuthor(author_name, author_email)

            # Update or create
            if file_exists:
                result = repo.update_file(
                    path=file_path,
                    content=content,
                    sha=sha,
                    branch=branch,
                    **commit_kwargs,
                )
            else:
                result = repo.create_file(
                    path=file_path, content=content, branch=branch, **commit_kwargs
                )

            return {"success": True, "commit_sha": result["commit"].sha}

        except GithubException as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": e.status if hasattr(e, "status") else None,
            }

    # ============ User/Organization Operations ============

    def list_user_repositories(
        self, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List user repositories."""
        self._ensure_authenticated()

        if user_id:
            user = self.client.get_user(user_id)
            repos = user.get_repos()
        else:
            repos = self.client.get_user().get_repos()

        return [
            {
                "id": repo.id,
                "name": repo.name,
                "full_name": repo.full_name,
                "owner": repo.owner.login,
                "private": repo.private,
                "url": repo.html_url,
            }
            for repo in repos
        ]

    def get_user_organizations(self) -> List[Dict[str, Any]]:
        """Get user organizations."""
        self._ensure_authenticated()

        orgs = self.client.get_user().get_orgs()

        return [
            {
                "id": org.id,
                "login": org.login,
                "name": org.name,
                "avatar_url": org.avatar_url,
            }
            for org in orgs
        ]

    # ============ Provider Metadata ============

    def get_provider_name(self) -> str:
        return "github"

    def get_api_base_url(self) -> str:
        return self.base_url

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limit info."""
        self._ensure_authenticated()

        rate_limit = self.client.get_rate_limit()

        return {
            "limit": rate_limit.core.limit,
            "remaining": rate_limit.core.remaining,
            "reset_at": rate_limit.core.reset.isoformat(),
        }

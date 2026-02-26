from typing import Any, Dict, List, Optional
import chardet

from app.modules.code_provider.base.code_provider_interface import (
    ICodeProvider,
    AuthMethod,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class GitLabProvider(ICodeProvider):
    """
    GitLab implementation of ICodeProvider interface.

    Uses the python-gitlab library to interact with the GitLab REST API v4.
    Supports both gitlab.com and self-hosted GitLab instances.

    Key differences from GitHub:
    - Uses "Merge Requests" instead of "Pull Requests"
    - Uses "namespaces" instead of "owners"
    - Rate limit info is from response headers, not a dedicated endpoint
    - Archive URL format differs from GitHub
    """

    def __init__(self, base_url: str = "https://gitlab.com"):
        """
        Initialize GitLab provider.

        Args:
            base_url: GitLab instance URL (default: https://gitlab.com)
        """
        self.base_url = base_url.rstrip("/")
        self.client = None
        self.auth_method: Optional[AuthMethod] = None

        logger.info(f"Initialized GitLab provider with base_url: {self.base_url}")

    # ============ Authentication ============

    def authenticate(self, credentials: Dict[str, Any], method: AuthMethod) -> Any:
        """Authenticate with GitLab."""
        try:
            import gitlab
        except ImportError:
            raise ImportError(
                "python-gitlab is required for GitLab support. "
                "Install it with: pip install python-gitlab"
            )

        self.auth_method = method

        if method == AuthMethod.PERSONAL_ACCESS_TOKEN:
            token = credentials.get("token")
            if not token:
                raise ValueError("PAT authentication requires 'token' in credentials")
            self.client = gitlab.Gitlab(url=self.base_url, private_token=token)
            logger.info(
                f"Authenticating with GitLab PAT (base URL: {self.base_url})"
            )

        elif method == AuthMethod.OAUTH_TOKEN:
            access_token = credentials.get("access_token")
            if not access_token:
                raise ValueError("OAuth authentication requires 'access_token'")
            self.client = gitlab.Gitlab(url=self.base_url, oauth_token=access_token)
            logger.info("Authenticating with GitLab OAuth token")

        elif method == AuthMethod.BASIC_AUTH:
            username = credentials.get("username")
            password = credentials.get("password")
            if not username or not password:
                raise ValueError("Basic auth requires 'username' and 'password'")
            self.client = gitlab.Gitlab(
                url=self.base_url,
                http_username=username,
                http_password=password,
            )
            logger.info(f"Authenticating with GitLab Basic Auth for user: {username}")

        else:
            raise ValueError(f"Unsupported authentication method: {method}")

        # Verify the authentication works
        try:
            self.client.auth()
            logger.info("GitLab authentication successful")
        except Exception as e:
            logger.warning(f"GitLab auth verification failed: {e} (continuing anyway)")

        return self.client

    def get_supported_auth_methods(self) -> List[AuthMethod]:
        return [
            AuthMethod.PERSONAL_ACCESS_TOKEN,
            AuthMethod.OAUTH_TOKEN,
            AuthMethod.BASIC_AUTH,
        ]

    def _ensure_authenticated(self):
        """Ensure client is initialized."""
        if not self.client:
            raise RuntimeError("Provider not authenticated. Call authenticate() first.")

    def _get_project(self, repo_name: str):
        """
        Get GitLab project by path (namespace/project).

        Args:
            repo_name: Repository in 'namespace/project' format

        Returns:
            GitLab Project object
        """
        self._ensure_authenticated()
        try:
            import gitlab
            return self.client.projects.get(repo_name)
        except Exception as e:
            logger.error(f"GitLabProvider: Failed to get project {repo_name}: {e}")
            raise

    # ============ Repository Operations ============

    def get_repository(self, repo_name: str) -> Dict[str, Any]:
        """Get repository (project) details."""
        project = self._get_project(repo_name)

        return {
            "id": project.id,
            "name": project.name,
            "full_name": project.path_with_namespace,
            "owner": project.namespace["path"],
            "default_branch": project.default_branch or "main",
            "private": project.visibility != "public",
            "url": project.web_url,
            "clone_url": project.http_url_to_repo,
            "description": project.description or "",
            "language": None,  # GitLab doesn't expose primary language via API
            "size": project.statistics.get("repository_size", 0) if hasattr(project, "statistics") and project.statistics else 0,
            "stars": project.star_count if hasattr(project, "star_count") else 0,
            "forks": project.forks_count if hasattr(project, "forks_count") else 0,
            "watchers": project.star_count if hasattr(project, "star_count") else 0,
            "open_issues": project.open_issues_count if hasattr(project, "open_issues_count") else 0,
            "created_at": project.created_at if hasattr(project, "created_at") else None,
            "updated_at": project.last_activity_at if hasattr(project, "last_activity_at") else None,
        }

    def check_repository_access(self, repo_name: str) -> bool:
        """Check if repository exists and is accessible."""
        try:
            self.get_repository(repo_name)
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
        """Get file content from repository."""
        project = self._get_project(repo_name)

        # Use default branch if ref not specified
        effective_ref = ref or project.default_branch or "main"

        try:
            f = project.files.get(file_path=file_path, ref=effective_ref)
            content_bytes = f.decode()

            if isinstance(content_bytes, bytes):
                try:
                    content = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    detected = chardet.detect(content_bytes)
                    encoding = detected.get("encoding", "utf-8")
                    content = content_bytes.decode(encoding, errors="ignore")
            else:
                content = str(content_bytes)

        except Exception as e:
            logger.error(
                f"GitLabProvider: Failed to get file {file_path} from {repo_name}@{effective_ref}: {e}"
            )
            raise

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
        """Get repository directory structure recursively."""
        project = self._get_project(repo_name)
        default_ref = ref or project.default_branch or "main"

        def _recurse(current_path: str, depth: int) -> List[Dict[str, Any]]:
            if depth > max_depth:
                return []

            result = []
            try:
                items = project.repository_tree(
                    path=current_path,
                    ref=default_ref,
                    recursive=False,
                    get_all=True,
                )

                for item in items:
                    entry = {
                        "name": item["name"],
                        "path": item["path"],
                        "type": "dir" if item["type"] == "tree" else "file",
                        "size": 0,
                        "sha": item["id"],
                    }
                    result.append(entry)

                    # Recurse into directories
                    if item["type"] == "tree":
                        entry["children"] = _recurse(item["path"], depth + 1)

            except Exception as e:
                logger.warning(
                    f"GitLabProvider: Failed to get tree for {current_path}@{default_ref}: {e}"
                )

            return result

        return _recurse(path, 0)

    # ============ Branch Operations ============

    def list_branches(self, repo_name: str) -> List[str]:
        """List all branches (default branch first)."""
        project = self._get_project(repo_name)

        branches = [b.name for b in project.branches.list(get_all=True)]

        # Put default branch first
        default = project.default_branch
        if default and default in branches:
            branches.remove(default)
            branches.insert(0, default)

        return branches

    def get_branch(self, repo_name: str, branch_name: str) -> Dict[str, Any]:
        """Get branch details."""
        project = self._get_project(repo_name)
        branch = project.branches.get(branch_name)

        return {
            "name": branch.name,
            "commit_sha": branch.commit["id"],
            "protected": branch.protected,
        }

    def create_branch(
        self, repo_name: str, branch_name: str, base_branch: str
    ) -> Dict[str, Any]:
        """Create a new branch from base branch."""
        project = self._get_project(repo_name)

        # Check if branch already exists
        try:
            project.branches.get(branch_name)
            return {
                "success": False,
                "error": f"Branch '{branch_name}' already exists",
            }
        except Exception:
            pass  # Branch doesn't exist, proceed

        try:
            new_branch = project.branches.create(
                {"branch": branch_name, "ref": base_branch}
            )
            return {
                "success": True,
                "branch_name": new_branch.name,
                "commit_sha": new_branch.commit["id"],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def compare_branches(
        self, repo_name: str, base_branch: str, head_branch: str
    ) -> Dict[str, Any]:
        """
        Compare two branches.

        GitLab uses 'diffs' instead of GitHub's 'files' with patches.
        """
        project = self._get_project(repo_name)

        try:
            comparison = project.repository_compare(base_branch, head_branch)

            files = []
            for diff in comparison.get("diffs", []):
                # Map GitLab diff status to GitHub-compatible status
                if diff.get("new_file"):
                    status = "added"
                elif diff.get("deleted_file"):
                    status = "removed"
                elif diff.get("renamed_file"):
                    status = "renamed"
                else:
                    status = "modified"

                file_data = {
                    "filename": diff.get("new_path") or diff.get("old_path", ""),
                    "status": status,
                    "additions": 0,
                    "deletions": 0,
                    "changes": 0,
                }

                # Extract patch and count additions/deletions
                patch = diff.get("diff", "")
                if patch:
                    file_data["patch"] = patch
                    for line in patch.split("\n"):
                        if line.startswith("+") and not line.startswith("+++"):
                            file_data["additions"] += 1
                        elif line.startswith("-") and not line.startswith("---"):
                            file_data["deletions"] += 1
                    file_data["changes"] = (
                        file_data["additions"] + file_data["deletions"]
                    )

                files.append(file_data)

            logger.info(
                f"[GITLAB] Compared branches {base_branch}...{head_branch}: "
                f"{len(files)} files, {len(comparison.get('commits', []))} commits"
            )

            return {
                "files": files,
                "commits": len(comparison.get("commits", [])),
            }

        except Exception as e:
            logger.error(f"[GITLAB] Error comparing branches: {str(e)}")
            raise

    # ============ Merge Request (Pull Request) Operations ============

    def list_pull_requests(
        self, repo_name: str, state: str = "open", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """List merge requests (GitLab's equivalent of pull requests)."""
        project = self._get_project(repo_name)

        # GitLab uses 'opened' instead of 'open'
        gl_state = "opened" if state == "open" else state

        mrs = project.mergerequests.list(state=gl_state, page=1, per_page=limit)

        return [
            {
                "number": mr.iid,
                "title": mr.title,
                "state": mr.state,
                "created_at": mr.created_at,
                "updated_at": mr.updated_at,
                "head_branch": mr.source_branch,
                "base_branch": mr.target_branch,
                "url": mr.web_url,
                "author": mr.author["username"] if mr.author else None,
            }
            for mr in mrs
        ]

    def get_pull_request(
        self, repo_name: str, pr_number: int, include_diff: bool = False
    ) -> Dict[str, Any]:
        """Get merge request details."""
        project = self._get_project(repo_name)
        mr = project.mergerequests.get(pr_number)

        result = {
            "number": mr.iid,
            "title": mr.title,
            "body": mr.description or "",
            "state": mr.state,
            "created_at": mr.created_at,
            "updated_at": mr.updated_at,
            "head_branch": mr.source_branch,
            "base_branch": mr.target_branch,
            "url": mr.web_url,
            "author": mr.author["username"] if mr.author else None,
        }

        if include_diff:
            try:
                changes = mr.changes()
                result["files"] = [
                    {
                        "filename": change.get("new_path", change.get("old_path", "")),
                        "status": (
                            "added"
                            if change.get("new_file")
                            else (
                                "removed"
                                if change.get("deleted_file")
                                else "modified"
                            )
                        ),
                        "additions": 0,
                        "deletions": 0,
                        "patch": change.get("diff", ""),
                    }
                    for change in changes.get("changes", [])
                ]
            except Exception as e:
                logger.warning(f"GitLabProvider: Could not get MR diff: {e}")

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
        """Create a merge request."""
        project = self._get_project(repo_name)

        try:
            mr_data = {
                "source_branch": head_branch,
                "target_branch": base_branch,
                "title": title,
                "description": body,
            }

            if labels:
                mr_data["labels"] = ",".join(labels)

            mr = project.mergerequests.create(mr_data)

            # Add reviewers if specified
            if reviewers:
                reviewer_ids = []
                for username in reviewers:
                    try:
                        users = self.client.users.list(username=username)
                        if users:
                            reviewer_ids.append(users[0].id)
                    except Exception as e:
                        logger.warning(
                            f"GitLabProvider: Could not find reviewer {username}: {e}"
                        )

                if reviewer_ids:
                    try:
                        mr.reviewer_ids = reviewer_ids
                        mr.save()
                    except Exception as e:
                        logger.warning(
                            f"GitLabProvider: Could not add reviewers: {e}"
                        )

            return {"success": True, "pr_number": mr.iid, "url": mr.web_url}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def add_pull_request_comment(
        self,
        repo_name: str,
        pr_number: int,
        body: str,
        commit_id: Optional[str] = None,
        path: Optional[str] = None,
        line: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Add comment to merge request (general or inline)."""
        project = self._get_project(repo_name)
        mr = project.mergerequests.get(pr_number)

        try:
            if path and line and commit_id:
                # Inline comment via discussion
                diff_refs = getattr(mr, "diff_refs", {}) or {}
                discussion = mr.discussions.create(
                    {
                        "body": body,
                        "position": {
                            "base_sha": diff_refs.get("base_sha", ""),
                            "start_sha": diff_refs.get("start_sha", ""),
                            "head_sha": commit_id,
                            "position_type": "text",
                            "new_path": path,
                            "new_line": line,
                        },
                    }
                )
                return {"success": True, "comment_id": discussion.id}
            else:
                # General comment
                note = mr.notes.create({"body": body})
                return {"success": True, "comment_id": note.id}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def create_pull_request_review(
        self,
        repo_name: str,
        pr_number: int,
        body: str,
        event: str,  # "COMMENT", "APPROVE", "REQUEST_CHANGES"
        comments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Create a review on a merge request.

        GitLab doesn't have a direct review concept like GitHub,
        but we approximate it using notes and the approve API.
        """
        project = self._get_project(repo_name)
        mr = project.mergerequests.get(pr_number)

        try:
            # Add the main review comment
            note = mr.notes.create({"body": body})

            # Add inline comments as discussions
            if comments:
                diff_refs = getattr(mr, "diff_refs", {}) or {}
                for c in comments:
                    try:
                        mr.discussions.create(
                            {
                                "body": c.get("body", ""),
                                "position": {
                                    "base_sha": diff_refs.get("base_sha", ""),
                                    "start_sha": diff_refs.get("start_sha", ""),
                                    "head_sha": diff_refs.get("head_sha", ""),
                                    "position_type": "text",
                                    "new_path": c.get("path", ""),
                                    "new_line": c.get("line", 1),
                                },
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            f"GitLabProvider: Failed to add inline comment: {e}"
                        )

            # Handle event type
            if event == "APPROVE":
                try:
                    mr.approve()
                except Exception as e:
                    logger.warning(f"GitLabProvider: Failed to approve MR: {e}")

            return {"success": True, "review_id": note.id}

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ============ Issue Operations ============

    def list_issues(
        self, repo_name: str, state: str = "open", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """List issues in repository."""
        project = self._get_project(repo_name)

        # GitLab uses 'opened' instead of 'open'
        gl_state = "opened" if state == "open" else state

        issues = project.issues.list(state=gl_state, page=1, per_page=limit)

        return [
            {
                "number": issue.iid,
                "title": issue.title,
                "state": issue.state,
                "created_at": issue.created_at,
                "updated_at": issue.updated_at,
                "url": issue.web_url,
                "author": issue.author["username"] if issue.author else None,
            }
            for issue in issues
        ]

    def get_issue(self, repo_name: str, issue_number: int) -> Dict[str, Any]:
        """Get issue details."""
        project = self._get_project(repo_name)
        issue = project.issues.get(issue_number)

        return {
            "number": issue.iid,
            "title": issue.title,
            "body": issue.description or "",
            "state": issue.state,
            "created_at": issue.created_at,
            "updated_at": issue.updated_at,
            "url": issue.web_url,
            "author": issue.author["username"] if issue.author else None,
        }

    def create_issue(
        self,
        repo_name: str,
        title: str,
        body: str,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create an issue."""
        project = self._get_project(repo_name)

        try:
            issue_data = {"title": title, "description": body}
            if labels:
                issue_data["labels"] = ",".join(labels)

            issue = project.issues.create(issue_data)
            return {"success": True, "issue_number": issue.iid, "url": issue.web_url}

        except Exception as e:
            return {"success": False, "error": str(e)}

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
        """Create or update a file in repository."""
        project = self._get_project(repo_name)

        try:
            # Check if file exists
            file_exists = False
            try:
                project.files.get(file_path=file_path, ref=branch)
                file_exists = True
            except Exception:
                file_exists = False

            file_data = {
                "file_path": file_path,
                "branch": branch,
                "content": content,
                "commit_message": commit_message,
            }
            if author_name:
                file_data["author_name"] = author_name
            if author_email:
                file_data["author_email"] = author_email

            if file_exists:
                project.files.update(file_data)
            else:
                project.files.create(file_data)

            # Get latest commit SHA from the branch
            try:
                branch_info = project.branches.get(branch)
                commit_sha = branch_info.commit["id"]
            except Exception:
                commit_sha = ""

            return {"success": True, "commit_sha": commit_sha}

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ============ User/Organization Operations ============

    def list_user_repositories(
        self, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List repositories accessible to authenticated user."""
        self._ensure_authenticated()

        try:
            if user_id:
                # Try to find user by username and list their projects
                users = self.client.users.list(username=user_id, get_all=False)
                if users:
                    # List visible projects for this user
                    projects = self.client.projects.list(
                        user_id=users[0].id, get_all=True
                    )
                else:
                    projects = []
            else:
                # List all projects the authenticated user is a member of
                projects = self.client.projects.list(membership=True, get_all=True)

        except Exception as e:
            logger.error(f"GitLabProvider: Failed to list repositories: {e}")
            projects = []

        return [
            {
                "id": p.id,
                "name": p.name,
                "full_name": p.path_with_namespace,
                "owner": p.namespace["path"],
                "private": p.visibility != "public",
                "url": p.web_url,
            }
            for p in projects
        ]

    def get_user_organizations(self) -> List[Dict[str, Any]]:
        """Get groups (organizations) for authenticated user."""
        self._ensure_authenticated()

        try:
            groups = self.client.groups.list(all=True)
        except Exception as e:
            logger.error(f"GitLabProvider: Failed to list groups: {e}")
            groups = []

        return [
            {
                "id": g.id,
                "login": g.path,
                "name": g.name,
                "avatar_url": getattr(g, "avatar_url", None),
            }
            for g in groups
        ]

    # ============ Archive Operations ============

    def get_archive_link(self, repo_name: str, format_type: str, ref: str) -> str:
        """
        Get archive download URL for a GitLab repository.

        GitLab archive URL format:
        https://gitlab.com/namespace/project/-/archive/ref/project-ref.tar.gz
        """
        self._ensure_authenticated()

        project_name = repo_name.split("/")[-1] if "/" in repo_name else repo_name
        safe_ref = ref.replace("/", "-")

        if format_type == "tarball":
            return (
                f"{self.base_url}/{repo_name}/-/archive/{ref}"
                f"/{project_name}-{safe_ref}.tar.gz"
            )
        else:
            return (
                f"{self.base_url}/{repo_name}/-/archive/{ref}"
                f"/{project_name}-{safe_ref}.zip"
            )

    # ============ Provider Metadata ============

    def get_provider_name(self) -> str:
        return "gitlab"

    def get_api_base_url(self) -> str:
        return self.base_url

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """
        Get rate limit info.

        GitLab doesn't have a dedicated rate limit endpoint like GitHub.
        The limits are in response headers (RateLimit-Limit, RateLimit-Remaining).
        We return known defaults.
        """
        self._ensure_authenticated()

        return {
            "limit": 3600,  # GitLab default: 3600 requests/hour per user
            "remaining": -1,  # Unknown without inspecting response headers
            "reset_at": None,
        }

    def get_client(self) -> Optional[Any]:
        return self.client

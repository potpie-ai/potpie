from typing import Any, Dict, List, Optional, Set
import chardet
from github import Github
from github.GithubException import GithubException

from app.modules.code_provider.base.code_provider_interface import (
    ICodeProvider,
    AuthMethod,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class GitBucketProvider(ICodeProvider):
    """
    GitBucket implementation of ICodeProvider interface.

    GitBucket is GitHub API v3 compatible, so we can reuse PyGithub library
    with custom base URL. Key differences:
    - No GitHub App authentication support
    - Partial GitHub API feature set
    - Self-hosted, requires custom base_url
    """

    def __init__(self, base_url: str):
        """
        Initialize GitBucket provider.

        Args:
            base_url: GitBucket API endpoint (e.g., 'http://localhost:8080/api/v3')
        """
        if not base_url:
            raise ValueError("GitBucket requires base_url parameter")

        # Ensure base_url doesn't end with /
        self.base_url = base_url.rstrip("/")
        self.client: Optional[Github] = None
        self.auth_method: Optional[AuthMethod] = None

        logger.info(f"Initialized GitBucket provider with base_url: {self.base_url}")

    # ============ Authentication ============

    def authenticate(self, credentials: Dict[str, Any], method: AuthMethod) -> Github:
        """Authenticate with GitBucket."""
        self.auth_method = method

        if method == AuthMethod.PERSONAL_ACCESS_TOKEN:
            token = credentials.get("token")
            if not token:
                raise ValueError("PAT authentication requires 'token' in credentials")
            self.client = Github(token, base_url=self.base_url)
            logger.info("Authenticated with GitBucket using PAT")

        elif method == AuthMethod.BASIC_AUTH:
            username = credentials.get("username")
            password = credentials.get("password")
            if not username or not password:
                raise ValueError("Basic auth requires 'username' and 'password'")
            # PyGithub supports basic auth via login/password
            self.client = Github(username, password, base_url=self.base_url)
            logger.info(
                f"Authenticated with GitBucket using Basic Auth for user: {username}"
            )

        elif method == AuthMethod.OAUTH_TOKEN:
            # GitBucket supports OAuth tokens (since v4.31.0)
            access_token = credentials.get("access_token")
            if not access_token:
                raise ValueError("OAuth authentication requires 'access_token'")
            self.client = Github(access_token, base_url=self.base_url)
            logger.info("Authenticated with GitBucket using OAuth token")

        elif method == AuthMethod.APP_INSTALLATION:
            raise NotImplementedError(
                "GitBucket does not support GitHub App authentication. "
                "Please use Personal Access Token (PAT) or Basic Authentication."
            )

        else:
            raise ValueError(f"Unsupported authentication method: {method}")

        return self.client

    def get_supported_auth_methods(self) -> List[AuthMethod]:
        """GitBucket supports PAT, Basic Auth, and OAuth (no App Installation)."""
        return [
            AuthMethod.PERSONAL_ACCESS_TOKEN,
            AuthMethod.BASIC_AUTH,
            AuthMethod.OAUTH_TOKEN,
        ]

    def _ensure_authenticated(self):
        """Ensure client is authenticated."""
        if not self.client:
            raise RuntimeError("Provider not authenticated. Call authenticate() first.")

    def _get_repo(self, repo_name: str):
        """
        Get repository object with normalized repo name conversion.

        Converts normalized repo name (e.g., 'user/repo') back to GitBucket's
        actual identifier format (e.g., 'root/repo') for API calls.

        Args:
            repo_name: Normalized repository name

        Returns:
            Repository object from PyGithub
        """
        from app.modules.parsing.utils.repo_name_normalizer import (
            get_actual_repo_name_for_lookup,
        )

        actual_repo_name = get_actual_repo_name_for_lookup(repo_name, "gitbucket")
        return self.client.get_repo(actual_repo_name)

    # ============ Repository Operations ============

    def get_repository(self, repo_name: str) -> Dict[str, Any]:
        """Get repository details."""
        self._ensure_authenticated()

        from app.modules.parsing.utils.repo_name_normalizer import (
            get_actual_repo_name_for_lookup,
            normalize_repo_name,
        )

        actual_repo_name = get_actual_repo_name_for_lookup(repo_name, "gitbucket")

        logger.info(
            f"GitBucket: Attempting to get repository '{repo_name}' (actual: '{actual_repo_name}')"
        )
        try:
            repo = self._get_repo(repo_name)
            logger.info(
                f"GitBucket: Successfully retrieved repository '{repo_name}' - ID: {repo.id}, Default branch: {repo.default_branch}"
            )

            # Normalize full_name and owner to use actual username instead of "root"
            normalized_full_name = normalize_repo_name(repo.full_name, "gitbucket")
            normalized_owner = (
                normalized_full_name.split("/")[0]
                if "/" in normalized_full_name
                else repo.owner.login
            )

            repo_data = {
                "id": repo.id,
                "name": repo.name,
                "full_name": normalized_full_name,
                "owner": normalized_owner,
                "default_branch": repo.default_branch,
                "private": repo.private,
                "url": repo.html_url,
                "description": repo.description,
                "language": repo.language,
            }
            logger.debug(f"GitBucket: Repository data for '{repo_name}': {repo_data}")
            return repo_data
        except GithubException as e:
            logger.exception(
                f"GitBucket: Failed to get repository '{repo_name}'",
                repo_name=repo_name,
                status=getattr(e, "status", "Unknown"),
            )

            # Handle specific GitBucket API differences
            if hasattr(e, "status") and e.status == 404:
                logger.error(
                    f"GitBucket: Repository '{repo_name}' not found. This might be due to:"
                )
                logger.error("  1. Repository doesn't exist")
                logger.error("  2. Insufficient permissions")
                logger.error(
                    "  3. Repository name format issue (expected: 'root/repo' for GitBucket)"
                )
                logger.error(
                    f"  4. GitBucket instance not accessible at {self.base_url}"
                )

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

        repo = self._get_repo(repo_name)
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
            repo = self._get_repo(repo_name)
        except GithubException:
            logger.exception("GitBucket: Failed to get repository", repo_name=repo_name)
            raise
        except Exception:
            logger.exception(
                "GitBucket: Unexpected error getting repository", repo_name=repo_name
            )
            raise

        # GitBucket doesn't handle ref=None well, so resolve it to the default branch
        if ref is None:
            try:
                ref = repo.default_branch
                logger.debug(f"GitBucket: Using default branch '{ref}' for ref")
            except Exception as e:
                logger.warning(
                    f"GitBucket: Could not get default branch, using 'main': {e}"
                )
                ref = "main"

        def _recurse(current_path: str, depth: int) -> List[Dict[str, Any]]:
            logger.debug(
                f"GitBucket: _recurse called with path='{current_path}', depth={depth}, max_depth={max_depth}"
            )

            if depth > max_depth:
                logger.warning(
                    f"GitBucket: Max depth {max_depth} reached for path '{current_path}' - stopping recursion"
                )
                return []

            # Validate path
            if not current_path or current_path.strip() == "":
                current_path = ""

            result = []
            try:
                logger.debug(
                    f"GitBucket: Getting contents for path '{current_path}' at depth {depth} with ref='{ref}'"
                )

                # GitBucket may have issues with get_contents for some paths
                # Try to use the raw API if standard method fails
                try:
                    contents = repo.get_contents(current_path, ref=ref)
                except (GithubException, Exception) as e:
                    error_msg = str(e)
                    logger.warning(
                        f"GitBucket: Standard get_contents failed for '{current_path}': {error_msg}"
                    )
                    logger.debug(
                        f"GitBucket: Error type: {type(e).__name__}, checking for URL error..."
                    )

                    # Check if this is the "no URL" error that GitBucket sometimes returns
                    # Also check for "Returned object contains" which is part of the full error message
                    if (
                        "no URL" in error_msg
                        or "400" in error_msg
                        or "Returned object contains" in error_msg
                    ):
                        logger.info(
                            f"GitBucket: Attempting raw API fallback for '{current_path}'"
                        )
                        # Try alternative approach using raw API and simple dict objects
                        try:
                            # Construct the API URL manually
                            if current_path:
                                url = f"{repo.url}/contents/{current_path}?ref={ref}"
                            else:
                                url = f"{repo.url}/contents?ref={ref}"

                            logger.debug(f"GitBucket: Using raw API: {url}")
                            headers, data = repo._requester.requestJsonAndCheck(
                                "GET", url
                            )

                            # Create simple namespace objects instead of ContentFile objects
                            # to avoid PyGithub's assumptions about GitBucket's response format
                            from types import SimpleNamespace

                            if isinstance(data, list):
                                contents = [
                                    SimpleNamespace(
                                        name=item.get("name", ""),
                                        path=item.get("path", ""),
                                        type=item.get("type", "file"),
                                        size=item.get("size", 0),
                                        sha=item.get("sha", ""),
                                        url=item.get("url", ""),
                                    )
                                    for item in data
                                ]
                            else:
                                contents = [
                                    SimpleNamespace(
                                        name=data.get("name", ""),
                                        path=data.get("path", ""),
                                        type=data.get("type", "file"),
                                        size=data.get("size", 0),
                                        sha=data.get("sha", ""),
                                        url=data.get("url", ""),
                                    )
                                ]
                            logger.info(
                                f"GitBucket: Raw API fallback succeeded for '{current_path}', found {len(contents)} items"
                            )
                        except Exception:
                            logger.exception(
                                "GitBucket: Raw API fallback also failed",
                                current_path=current_path,
                                repo_name=repo_name,
                            )
                            raise
                    else:
                        raise

                # Handle both single item and list responses
                if not isinstance(contents, list):
                    contents = [contents]

                logger.debug(
                    f"GitBucket: Found {len(contents)} items in path '{current_path}'"
                )

                for item in contents:
                    # Safely extract attributes with fallbacks for GitBucket compatibility
                    # Access raw attributes directly to avoid PyGithub's lazy loading which fails with GitBucket
                    try:
                        # Try to access raw internal attributes first (avoid triggering _complete)
                        item_type = (
                            item._type.value if hasattr(item, "_type") else "file"
                        )
                        item_path = item._path.value if hasattr(item, "_path") else ""
                        item_name = item._name.value if hasattr(item, "_name") else ""
                        item_size = (
                            item._size.value
                            if hasattr(item, "_size") and item._size.value is not None
                            else 0
                        )
                        item_sha = item._sha.value if hasattr(item, "_sha") else ""
                    except Exception as e:
                        logger.warning(
                            f"GitBucket: Error accessing raw attributes for item: {e}"
                        )
                        # Fallback to trying getattr (which might trigger lazy loading)
                        try:
                            item_type = getattr(item, "type", "file")
                            item_path = getattr(item, "path", "")
                            item_name = getattr(item, "name", "")
                            item_size = (
                                getattr(item, "size", 0) if hasattr(item, "size") else 0
                            )
                            item_sha = getattr(item, "sha", "")
                        except:
                            # Last resort: use empty defaults
                            item_type = "file"
                            item_path = ""
                            item_name = ""
                            item_size = 0
                            item_sha = ""

                    entry = {
                        "name": item_name,
                        "path": item_path,
                        "type": item_type,
                        "size": item_size,
                        "sha": item_sha,
                    }
                    result.append(entry)

                    # Recurse into directories
                    if item_type == "dir":
                        logger.debug(
                            f"GitBucket: Found directory '{item_path}', recursing at depth {depth + 1}"
                        )
                        try:
                            children = _recurse(item_path, depth + 1)
                            entry["children"] = children
                            logger.debug(
                                f"GitBucket: Directory '{item_path}' returned {len(children)} children"
                            )
                        except GithubException:
                            logger.exception(
                                "GitBucket: GithubException recursing into directory",
                                item_path=item_path,
                                repo_name=repo_name,
                            )
                            entry["children"] = []
                        except Exception:
                            logger.exception(
                                "GitBucket: Unexpected exception recursing into directory",
                                item_path=item_path,
                                repo_name=repo_name,
                            )
                            entry["children"] = []

            except GithubException:
                logger.exception(
                    "GitBucket: GithubException getting contents",
                    current_path=current_path,
                    repo_name=repo_name,
                )
                # Return empty result instead of failing completely
            except Exception:
                logger.exception(
                    "GitBucket: Unexpected error getting contents",
                    current_path=current_path,
                    repo_name=repo_name,
                )

            return result

        return _recurse(path, 0)

    # ============ Branch Operations ============

    def list_branches(self, repo_name: str) -> List[str]:
        """List branches."""
        self._ensure_authenticated()

        repo = self._get_repo(repo_name)
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

        from app.modules.parsing.utils.repo_name_normalizer import (
            get_actual_repo_name_for_lookup,
        )

        actual_repo_name = get_actual_repo_name_for_lookup(repo_name, "gitbucket")

        logger.info(
            f"GitBucket: Getting branch '{branch_name}' for repository '{repo_name}' (actual: '{actual_repo_name}')"
        )
        try:
            repo = self._get_repo(repo_name)
            branch = repo.get_branch(branch_name)

            branch_data = {
                "name": branch.name,
                "commit_sha": branch.commit.sha,
                "protected": branch.protected,
            }
            logger.info(
                f"GitBucket: Successfully retrieved branch '{branch_name}' - SHA: {branch.commit.sha}"
            )
            logger.debug(f"GitBucket: Branch data for '{branch_name}': {branch_data}")
            return branch_data
        except GithubException as e:
            logger.exception(
                "GitBucket: Failed to get branch",
                branch_name=branch_name,
                repo_name=repo_name,
                status=getattr(e, "status", "Unknown"),
            )

            # Handle specific GitBucket API differences
            if hasattr(e, "status") and e.status == 404:
                logger.error(
                    f"GitBucket: Branch '{branch_name}' not found in repository '{repo_name}'. This might be due to:"
                )
                logger.error("  1. Branch doesn't exist")
                logger.error("  2. Repository access issues")
                logger.error("  3. GitBucket API compatibility issues")

            raise

    def create_branch(
        self, repo_name: str, branch_name: str, base_branch: str
    ) -> Dict[str, Any]:
        """Create branch."""
        self._ensure_authenticated()

        try:
            repo = self._get_repo(repo_name)

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
        Compare two branches using commits API (GitBucket workaround).

        GitBucket doesn't fully support the /compare endpoint, so we iterate
        through commits on the head branch until we reach the common ancestor.

        Args:
            repo_name: Repository name (e.g., 'owner/repo')
            base_branch: Base branch to compare from
            head_branch: Head branch to compare to

        Returns:
            Dict with files (list of file changes with patches) and commits count
        """
        self._ensure_authenticated()

        try:
            repo = self._get_repo(repo_name)

            # Get commits on the head branch
            logger.info(f"[GITBUCKET] Getting commits for branch: {head_branch}")
            head_commits = repo.get_commits(sha=head_branch)

            max_commits = 50  # Safety limit

            # Get commits on the base branch for comparison
            base_commit_shas: Set[str] = set()
            for idx, base_commit in enumerate(repo.get_commits(sha=base_branch)):
                base_commit_shas.add(base_commit.sha)
                if idx + 1 >= max_commits:
                    break

            # Track files and their patches
            files_dict = {}
            commit_count = 0

            # Iterate through head branch commits until we find common ancestor
            for commit in head_commits:
                if commit.sha in base_commit_shas:
                    logger.info(
                        f"[GITBUCKET] Reached common ancestor at commit {commit.sha[:7]}"
                    )
                    break

                commit_count += 1
                logger.info(
                    f"[GITBUCKET] Processing commit {commit.sha[:7]}: {commit.commit.message.split(chr(10))[0]}"
                )

                # Extract files from this commit
                for file in commit.files:
                    # Only add file if we haven't seen it yet (keep first occurrence)
                    if file.filename not in files_dict:
                        file_data = {
                            "filename": file.filename,
                            "status": file.status,
                            "additions": file.additions,
                            "deletions": file.deletions,
                            "changes": file.changes,
                        }
                        if file.patch:
                            file_data["patch"] = file.patch
                        files_dict[file.filename] = file_data
                        logger.info(f"[GITBUCKET] Added file: {file.filename}")

                # Safety check
                if commit_count >= max_commits:
                    logger.warning(
                        f"[GITBUCKET] Reached commit limit of {max_commits}, stopping"
                    )
                    break

            # Convert dict to list
            files = list(files_dict.values())

            logger.info(
                f"[GITBUCKET] Compared branches {base_branch}...{head_branch}: {len(files)} files, {commit_count} commits"
            )

            return {
                "files": files,
                "commits": commit_count,
            }

        except GithubException:
            logger.exception(
                "[GITBUCKET] Error comparing branches",
                base_branch=base_branch,
                head_branch=head_branch,
                repo_name=repo_name,
            )
            raise

    # ============ Pull Request Operations ============

    def list_pull_requests(
        self, repo_name: str, state: str = "open", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """List pull requests."""
        self._ensure_authenticated()

        repo = self._get_repo(repo_name)
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

        repo = self._get_repo(repo_name)
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
            repo = self._get_repo(repo_name)

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

            # Add reviewers (may not be fully supported by GitBucket)
            if reviewers:
                try:
                    pr.create_review_request(reviewers=reviewers)
                except GithubException as e:
                    logger.warning(
                        f"Error adding reviewers (GitBucket may not support this): {e}"
                    )

            # Add labels (may not be fully supported by GitBucket)
            if labels:
                try:
                    pr.add_to_labels(*labels)
                except GithubException as e:
                    logger.warning(
                        f"Error adding labels (GitBucket may not support this): {e}"
                    )

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
            repo = self._get_repo(repo_name)
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
            repo = self._get_repo(repo_name)
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
            logger.warning(
                f"PR review creation may not be fully supported by GitBucket: {e}"
            )
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

        repo = self._get_repo(repo_name)
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

        repo = self._get_repo(repo_name)
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
            repo = self._get_repo(repo_name)
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
            repo = self._get_repo(repo_name)

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

        from app.modules.parsing.utils.repo_name_normalizer import normalize_repo_name

        if user_id:
            user = self.client.get_user(user_id)
            repos = user.get_repos()
        else:
            repos = self.client.get_user().get_repos()

        result = []
        for repo in repos:
            # Normalize full_name and owner to use actual username instead of "root"
            normalized_full_name = normalize_repo_name(repo.full_name, "gitbucket")
            normalized_owner = (
                normalized_full_name.split("/")[0]
                if "/" in normalized_full_name
                else repo.owner.login
            )

            result.append(
                {
                    "id": repo.id,
                    "name": repo.name,
                    "full_name": normalized_full_name,
                    "owner": normalized_owner,
                    "private": repo.private,
                    "url": repo.html_url,
                }
            )
        return result

    def get_user_organizations(self) -> List[Dict[str, Any]]:
        """
        Get user organizations.

        Note: GitBucket uses "Groups" which are returned as organizations
        for API compatibility.
        """
        self._ensure_authenticated()

        try:
            orgs = self.client.get_user().get_orgs()

            return [
                {
                    "id": org.id,
                    "login": org.login,
                    "name": (
                        org.name if hasattr(org, "name") and org.name else org.login
                    ),
                    "avatar_url": (
                        org.avatar_url if hasattr(org, "avatar_url") else None
                    ),
                }
                for org in orgs
            ]
        except GithubException as e:
            logger.warning(f"Failed to get organizations (GitBucket Groups): {e}")
            return []

    # ============ Archive Operations ============

    def get_archive_link(self, repo_name: str, format_type: str, ref: str) -> str:
        """Get archive download link for repository."""
        self._ensure_authenticated()

        # Convert normalized repo name back to GitBucket format for API calls
        from app.modules.parsing.utils.repo_name_normalizer import (
            get_actual_repo_name_for_lookup,
        )

        actual_repo_name = get_actual_repo_name_for_lookup(repo_name, "gitbucket")

        logger.info(
            f"GitBucket: Getting archive link for repo '{repo_name}' (actual: '{actual_repo_name}'), format: '{format_type}', ref: '{ref}'"
        )

        try:
            self._get_repo(repo_name)

            # GitBucket uses a different URL format than GitHub API
            # The correct format is: http://hostname/owner/repo/archive/ref.format
            # We need to extract the base URL without /api/v3 and construct the proper path

            # Extract the base URL (remove /api/v3 if present)
            base_url = self.base_url
            if base_url.endswith("/api/v3"):
                base_url = base_url[:-7]  # Remove '/api/v3'

            # Construct the correct GitBucket archive URL using actual repo name
            if format_type == "tarball":
                archive_url = f"{base_url}/{actual_repo_name}/archive/{ref}.tar.gz"
            elif format_type == "zipball":
                archive_url = f"{base_url}/{actual_repo_name}/archive/{ref}.zip"
            else:
                raise ValueError(f"Unsupported archive format: {format_type}")

            logger.info(f"GitBucket: Constructed archive URL: {archive_url}")

            # Test the URL to make sure it works
            import requests

            try:
                response = requests.head(archive_url, timeout=10)
                if response.status_code == 200:
                    logger.info(
                        f"GitBucket: Archive URL is accessible - Status: {response.status_code}"
                    )
                    return archive_url
                else:
                    logger.warning(
                        f"GitBucket: Archive URL returned status {response.status_code}"
                    )
                    # Still return the URL as it might work with authentication
                    return archive_url
            except requests.exceptions.RequestException as e:
                logger.warning(f"GitBucket: Error testing archive URL: {e}")
                # Still return the URL as it might work with authentication
                return archive_url

        except GithubException as e:
            logger.exception(
                "GitBucket: Failed to get archive link",
                repo_name=repo_name,
                status=getattr(e, "status", "Unknown"),
            )

            # Handle specific GitBucket API differences
            if hasattr(e, "status") and e.status == 404:
                logger.error(
                    f"GitBucket: Repository '{repo_name}' not found for archive download. This might be due to:"
                )
                logger.error("  1. Repository doesn't exist")
                logger.error("  2. Insufficient permissions")
                logger.error("  3. GitBucket archive feature not available")
                logger.error("  4. Repository name format issue")

            raise
        except Exception:
            logger.exception(
                "GitBucket: Unexpected error getting archive link", repo_name=repo_name
            )
            raise

    # ============ Provider Metadata ============

    def get_provider_name(self) -> str:
        return "gitbucket"

    def get_api_base_url(self) -> str:
        return self.base_url

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limit info."""
        self._ensure_authenticated()

        try:
            rate_limit = self.client.get_rate_limit()
            return {
                "limit": rate_limit.core.limit,
                "remaining": rate_limit.core.remaining,
                "reset_at": rate_limit.core.reset.isoformat(),
            }
        except GithubException as e:
            # GitBucket might not fully implement rate limit API
            logger.warning(
                f"Failed to get rate limit info (GitBucket may not support this): {e}"
            )
            return {"limit": None, "remaining": None, "reset_at": None}

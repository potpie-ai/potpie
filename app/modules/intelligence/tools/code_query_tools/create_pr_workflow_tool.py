"""
Create PR Workflow Tool

A composite tool that orchestrates the full PR workflow in a single operation:
1. Applies changes from CodeChangesManager to worktree
2. Creates a new branch
3. Commits all changes
4. Pushes the branch to remote
5. Creates a pull request

This reduces LLM round-trips and eliminates the need for delegation to github_agent.
"""

import os
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool

from app.modules.projects.projects_service import ProjectService
from app.modules.repo_manager import RepoManager
from app.modules.repo_manager.sync_helper import (
    get_or_create_worktree_path,
    get_or_create_edits_worktree_path,
)
from app.modules.code_provider.git_safe import safe_git_repo_operation, GitOperationError
from app.modules.code_provider.provider_factory import CodeProviderFactory
from app.modules.intelligence.tools.code_changes_manager import CodeChangesManager
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class CreatePRWorkflowInput(BaseModel):
    """Input for the composite PR workflow tool."""

    project_id: str = Field(
        ..., description="Project ID that references the repository"
    )
    conversation_id: str = Field(
        ..., description="Conversation ID where changes are stored in CodeChangesManager"
    )
    branch_name: str = Field(
        ..., description="Name of the new branch to create (e.g., 'feature/my-feature')"
    )
    commit_message: str = Field(
        ..., description="Git commit message describing all the changes"
    )
    pr_title: str = Field(..., description="Title for the pull request")
    pr_body: str = Field(
        default="", description="Body/description for the pull request"
    )
    base_branch: str = Field(
        default="main", description="Base branch to create PR against (default: main)"
    )


class CreatePRWorkflowTool:
    """
    Composite tool that creates a PR from CodeChangesManager changes in one operation.

    This tool orchestrates the full workflow:
    1. Applies all changes from CodeChangesManager to the worktree filesystem
    2. Creates a new branch from the base branch
    3. Commits all changes in a single commit
    4. Pushes the branch to the remote
    5. Creates a pull request

    Use this when you have batched file changes ready to create a PR.
    """

    name: str = "create_pr_workflow"
    description: str = """Creates a PR from CodeChangesManager changes in one composite operation.

This tool orchestrates the full PR workflow:
1. Applies all changes from CodeChangesManager to the worktree
2. Creates a new branch from the base branch
3. Commits all changes in a single commit
4. Pushes the branch to remote
5. Creates a pull request

Use this when you have batched file changes in CodeChangesManager and want to create a PR.

Args:
    project_id: The repository project ID (UUID)
    conversation_id: The conversation ID where changes are stored
    branch_name: New branch name to create (e.g., 'feature/my-feature')
    commit_message: Comprehensive commit message for all changes
    pr_title: Title for the pull request
    pr_body: Body/description for the pull request (optional)
    base_branch: Base branch to create PR against (default: 'main')

Returns:
    Dictionary with:
    - success: bool indicating overall success
    - step: Which step failed (if applicable)
    - branch: The created branch name
    - commit_hash: The commit hash (if created)
    - pr_number: The PR number (if created)
    - pr_url: The PR URL (if created)
    - files_committed: List of files committed
    - error: Error message if failed
"""
    args_schema: type[BaseModel] = CreatePRWorkflowInput

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self.project_service = ProjectService(sql_db)

        # Initialize repo manager if enabled
        self.repo_manager = None
        try:
            repo_manager_enabled = (
                os.getenv("REPO_MANAGER_ENABLED", "false").lower() == "true"
            )
            if repo_manager_enabled:
                self.repo_manager = RepoManager()
                logger.info("CreatePRWorkflowTool: RepoManager initialized")
        except Exception as e:
            logger.warning(f"CreatePRWorkflowTool: Failed to initialize RepoManager: {e}")

    def _get_project_details(self, project_id: str) -> Dict[str, str]:
        """Get project details and validate user access."""
        details = self.project_service.get_project_from_db_by_id_sync(project_id)  # type: ignore[arg-type]
        if not details or "project_name" not in details:
            raise ValueError(f"Cannot find repo details for project_id: {project_id}")
        if details["user_id"] != self.user_id:
            raise ValueError(
                f"Cannot find repo details for project_id: {project_id} for current user"
            )
        return details

    def _get_worktree_path(
        self,
        project_id: str,
        conversation_id: str,
        user_id: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        """Get the edits worktree path for this conversation. Returns (path, failure_reason)."""
        return get_or_create_edits_worktree_path(
            self.repo_manager,
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
            sql_db=self.sql_db,
        )

    def _get_auth_token(self, repo_name: str) -> Optional[str]:
        """Get authentication token using priority chain (same as sync_helper).

        Priority 1: GitHub App token
        Priority 2: User OAuth token from DB
        Priority 3: Environment token (via repo_manager)
        """
        # Priority 1: Try GitHub App token if available
        try:
            from app.modules.code_provider.provider_factory import (
                CodeProviderFactory,
            )

            provider = CodeProviderFactory.create_github_app_provider(repo_name)
            if (
                provider
                and hasattr(provider, "client")
                and hasattr(provider.client, "_Github__requester")
            ):
                requester = provider.client._Github__requester
                if hasattr(requester, "auth") and requester.auth:
                    token = requester.auth.token
                    logger.info(
                        f"CreatePRWorkflowTool: Using GitHub App token for {repo_name}"
                    )
                    return token
        except Exception:
            pass

        # Priority 2: Get User OAuth token from DB
        if self.sql_db and self.user_id:
            try:
                from app.modules.code_provider.github.github_service import (
                    GithubService,
                )

                token = GithubService(self.sql_db).get_github_oauth_token(
                    self.user_id
                )
                if token:
                    logger.info(
                        f"CreatePRWorkflowTool: Using user OAuth token for {repo_name}"
                    )
                    return token
            except Exception:
                pass

        # Priority 3: Environment token (via repo_manager)
        if self.repo_manager:
            token = self.repo_manager._get_github_token()
            if token:
                logger.info(
                    f"CreatePRWorkflowTool: Using environment token for {repo_name}"
                )
                return token

        logger.warning(f"CreatePRWorkflowTool: No authentication token found")
        return None

    def _apply_changes(
        self, project_id: str, conversation_id: str, worktree_path: str
    ) -> Dict[str, Any]:
        """Step 1: Apply changes from CodeChangesManager to worktree."""
        try:
            changes_manager = CodeChangesManager(conversation_id=conversation_id)
            all_changes = changes_manager.changes

            if not all_changes:
                return {
                    "success": True,
                    "message": "No changes to apply",
                    "files_applied": [],
                    "files_deleted": [],
                }

            worktree_path_abs = os.path.abspath(worktree_path)
            files_applied: List[str] = []
            files_deleted: List[str] = []
            files_skipped: List[str] = []

            def _is_path_safe(fpath: str) -> tuple[bool, str]:
                """Validate fpath is inside worktree (prevents path traversal)."""
                full = os.path.abspath(os.path.join(worktree_path, fpath))
                try:
                    common = os.path.commonpath([worktree_path_abs, full])
                    if common != worktree_path_abs:
                        return False, f"{fpath} (path traversal)"
                    return True, full
                except ValueError:
                    return False, f"{fpath} (path traversal)"

            for fpath, change in all_changes.items():
                if change.change_type.value == "delete":
                    safe, path_or_reason = _is_path_safe(fpath)
                    if not safe:
                        logger.error(
                            f"CreatePRWorkflowTool: Path traversal blocked: {fpath}"
                        )
                        files_skipped.append(path_or_reason)
                        continue
                    full_path = path_or_reason
                    try:
                        if os.path.exists(full_path):
                            os.remove(full_path)
                        files_deleted.append(fpath)
                        logger.info(f"CreatePRWorkflowTool: Deleted {fpath}")
                    except Exception as e:
                        logger.error(
                            f"CreatePRWorkflowTool: Failed to delete {fpath}: {e}"
                        )
                        files_skipped.append(f"{fpath} (delete error: {str(e)})")
                    continue

                if change.content is None:
                    files_skipped.append(f"{fpath} (no content)")
                    continue

                safe, path_or_reason = _is_path_safe(fpath)
                if not safe:
                    logger.error(
                        f"CreatePRWorkflowTool: Path traversal blocked: {fpath}"
                    )
                    files_skipped.append(path_or_reason)
                    continue
                full_path = path_or_reason

                parent_dir = os.path.dirname(full_path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)

                try:
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(change.content)
                    files_applied.append(fpath)
                    logger.info(f"CreatePRWorkflowTool: Applied changes to {fpath}")
                except Exception as e:
                    logger.error(
                        f"CreatePRWorkflowTool: Failed to write {fpath}: {e}"
                    )
                    files_skipped.append(f"{fpath} (write error: {str(e)})")

            # Check for critical failures
            failure_skips = [
                s
                for s in files_skipped
                if "(write error" in s or "(delete error" in s or "(path traversal" in s
            ]
            success = len(failure_skips) == 0

            return {
                "success": success,
                "files_applied": files_applied,
                "files_deleted": files_deleted,
                "files_skipped": files_skipped,
            }

        except Exception as e:
            logger.exception("CreatePRWorkflowTool: Error applying changes")
            return {"success": False, "error": str(e)}

    def _create_branch_commit_and_push(
        self,
        worktree_path: str,
        branch_name: str,
        commit_message: str,
        files_to_stage: List[str],
        repo_name: str,
    ) -> Dict[str, Any]:
        """Steps 2, 3, 4: Create branch, commit, and push in one git operation."""

        def _git_operation(repo):
            # Step 2: Create or reuse the target branch
            existing_branch = next(
                (b for b in repo.branches if b.name == branch_name), None
            )
            if existing_branch is not None:
                # Branch already exists — checkout only if not already on it
                try:
                    if repo.active_branch.name != branch_name:
                        existing_branch.checkout()
                        logger.info(
                            f"CreatePRWorkflowTool: Checked out existing branch {branch_name}"
                        )
                    else:
                        logger.info(
                            f"CreatePRWorkflowTool: Already on branch {branch_name}"
                        )
                except TypeError:
                    # Detached HEAD — checkout the branch
                    existing_branch.checkout()
            else:
                logger.info(f"CreatePRWorkflowTool: Creating branch {branch_name}")
                new_branch = repo.create_head(branch_name)
                new_branch.checkout()

            # Step 3: Stage files (additions, modifications, and deletions)
            if files_to_stage:
                repo.git.add("-A", "--", *files_to_stage)

            # Check if there are changes to commit
            diff = repo.git.diff("--cached", "--name-only")
            if not diff.strip():
                # No new diff after staging — written files already match HEAD.
                # Verify that the previous commit actually contains all the
                # intended files before deciding to skip the commit step.
                try:
                    on_target = repo.active_branch.name == branch_name
                except TypeError:
                    on_target = False

                if on_target and repo.head.commit:
                    # Get files changed in the last commit
                    try:
                        committed_files = set(
                            repo.git.diff(
                                "HEAD~1", "HEAD", "--name-only"
                            ).strip().split("\n")
                        )
                    except Exception:
                        # HEAD~1 may not exist (first commit) — treat all as committed
                        committed_files = set(files_to_stage)

                    missing = [
                        f for f in files_to_stage
                        if f not in committed_files
                    ]
                    if missing:
                        # Some files are missing from the last commit.
                        # Stage and amend the commit to include them.
                        logger.info(
                            f"CreatePRWorkflowTool: {len(missing)} file(s) missing "
                            f"from previous commit, amending: {missing}"
                        )
                        repo.git.add("-A", "--", *missing)
                        repo.git.commit("--amend", "--no-edit")

                    commit_hash = repo.head.commit.hexsha
                    staged_files = files_to_stage
                    logger.info(
                        f"CreatePRWorkflowTool: Branch {branch_name} already has "
                        f"changes committed ({commit_hash[:8]}), skipping to push"
                    )
                else:
                    return {
                        "success": False,
                        "error": "No changes to commit after staging",
                        "step": "commit",
                    }
            else:
                staged_files = [f.strip() for f in diff.strip().split("\n") if f.strip()]

                # Create commit
                commit = repo.index.commit(commit_message)
                commit_hash = commit.hexsha
                logger.info(f"CreatePRWorkflowTool: Created commit {commit_hash[:8]}")

            # Step 4: Push to remote with authentication
            logger.info(f"CreatePRWorkflowTool: Pushing {branch_name} to origin")

            # Get authentication token and configure remote URL
            auth_token = self._get_auth_token(repo_name)
            remote_url = None
            if auth_token and self.repo_manager:
                try:
                    # Get current remote URL (strip to avoid libcurl port parse errors)
                    remote_url = (
                        repo.git.remote("get-url", "origin").strip().rstrip("/")
                    )
                    auth_url = self.repo_manager._build_authenticated_url(
                        remote_url, auth_token, repo_name, self.user_id
                    )
                    if auth_url:
                        auth_url = auth_url.rstrip("/")
                    if auth_url:
                        repo.git.remote("set-url", "origin", auth_url)
                    logger.info(
                        f"CreatePRWorkflowTool: Configured authenticated remote URL"
                    )
                except Exception as e:
                    logger.warning(
                        f"CreatePRWorkflowTool: Failed to configure auth URL: {e}"
                    )

            try:
                # Check if remote branch already exists to decide push strategy.
                # If it does, use --force-with-lease in case we amended the commit.
                remote_refs = [
                    ref.name
                    for ref in repo.remotes.origin.refs
                ] if repo.remotes else []
                remote_branch_ref = f"origin/{branch_name}"
                if remote_branch_ref in remote_refs:
                    repo.git.push("--force-with-lease", "--set-upstream", "origin", branch_name)
                    logger.info(f"CreatePRWorkflowTool: Force-with-lease push successful (remote branch existed)")
                else:
                    repo.git.push("--set-upstream", "origin", branch_name)
                    logger.info(f"CreatePRWorkflowTool: Push successful")
            finally:
                # Reset remote URL back to original (remove credentials)
                if auth_token and self.repo_manager and remote_url:
                    try:
                        repo.git.remote("set-url", "origin", remote_url)
                        logger.debug(
                            f"CreatePRWorkflowTool: Reset remote URL to original"
                        )
                    except Exception as e:
                        logger.debug(
                            f"CreatePRWorkflowTool: Failed to reset remote URL: {e}"
                        )

            return {
                "success": True,
                "branch": branch_name,
                "commit_hash": commit_hash,
                "files_committed": staged_files,
            }

        return safe_git_repo_operation(
            worktree_path,
            _git_operation,
            operation_name="create_branch_commit_and_push",
            timeout=120.0,  # Longer timeout for combined operation
        )

    def _extract_patch_from_commit(
        self, worktree_path: str, file_path: str, commit_hash: str
    ) -> Optional[str]:
        """
        Extract unified diff patch from a specific commit for a file.

        Uses: git show --pretty=format: --patch {commit_hash} -- {file_path}

        Args:
            worktree_path: Path to the git worktree
            file_path: Relative path to the file
            commit_hash: The commit hash to extract patch from

        Returns:
            Unified diff patch string, or None if extraction failed
        """
        try:
            from app.modules.code_provider.git_safe import safe_git_repo_operation

            def _extract_operation(repo):
                try:
                    # Extract patch using git show
                    patch = repo.git.show(
                        "--pretty=format:",  # No commit metadata, just the diff
                        "--patch",
                        commit_hash,
                        "--",
                        file_path,
                    )
                    if patch and patch.strip():
                        logger.info(
                            f"CreatePRWorkflowTool: Extracted patch for {file_path} from {commit_hash[:8]} "
                            f"({len(patch)} chars)"
                        )
                        return patch
                    return None
                except Exception as e:
                    logger.warning(
                        f"CreatePRWorkflowTool: Failed to extract patch for {file_path}: {e}"
                    )
                    return None

            result = safe_git_repo_operation(
                worktree_path,
                _extract_operation,
                operation_name="extract_patch",
                timeout=10.0,
            )
            return result if isinstance(result, str) else None

        except Exception as e:
            logger.warning(
                f"CreatePRWorkflowTool: Error extracting patch for {file_path}: {e}"
            )
            return None

    def _store_patch_in_redis(
        self, conversation_id: str, file_path: str, patch: str
    ) -> bool:
        """
        Store extracted patch in Redis for later PR creation.

        Redis key format: pr_patches:{conversation_id}:{file_path}

        Args:
            conversation_id: The conversation ID
            file_path: Relative path to the file
            patch: The unified diff patch string

        Returns:
            True if stored successfully, False otherwise
        """
        try:
            import redis
            from app.core.config_provider import ConfigProvider

            config = ConfigProvider()
            client = redis.from_url(config.get_redis_url())

            # Sanitize file path for Redis key (replace / with : to avoid key issues)
            safe_file_path = file_path.replace("/", ":")
            key = f"pr_patches:{conversation_id}:{safe_file_path}"

            # Store with same TTL as code changes (24 hours)
            from app.modules.intelligence.tools.code_changes_manager import (
                CODE_CHANGES_TTL_SECONDS,
            )

            client.setex(key, CODE_CHANGES_TTL_SECONDS, patch)
            logger.info(
                f"CreatePRWorkflowTool: Stored patch for {file_path} in Redis "
                f"(key={key}, {len(patch)} chars)"
            )
            return True

        except Exception as e:
            logger.warning(
                f"CreatePRWorkflowTool: Failed to store patch for {file_path}: {e}"
            )
            return False

    def _get_stored_patches(
        self, conversation_id: str
    ) -> Dict[str, str]:
        """
        Retrieve all stored patches for a conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            Dictionary mapping file paths to patch strings
        """
        try:
            import redis
            from app.core.config_provider import ConfigProvider

            config = ConfigProvider()
            client = redis.from_url(config.get_redis_url())

            # Find all patch keys for this conversation
            pattern = f"pr_patches:{conversation_id}:*"
            keys = client.keys(pattern)

            patches = {}
            for key in keys:
                key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                data = client.get(key_str)
                if data:
                    # Extract file path from key (convert : back to /)
                    file_path = key_str.split(":", 2)[2].replace(":", "/")
                    patch = data.decode("utf-8") if isinstance(data, bytes) else data
                    patches[file_path] = patch

            logger.info(
                f"CreatePRWorkflowTool: Retrieved {len(patches)} patches for conversation {conversation_id}"
            )
            return patches

        except Exception as e:
            logger.warning(
                f"CreatePRWorkflowTool: Failed to retrieve patches: {e}"
            )
            return {}

    def _clear_stored_patches(self, conversation_id: str) -> bool:
        """
        Clear all stored patches for a conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            import redis
            from app.core.config_provider import ConfigProvider

            config = ConfigProvider()
            client = redis.from_url(config.get_redis_url())

            pattern = f"pr_patches:{conversation_id}:*"
            keys = client.keys(pattern)
            if keys:
                client.delete(*keys)
                logger.info(
                    f"CreatePRWorkflowTool: Cleared {len(keys)} patches for conversation {conversation_id}"
                )
            return True

        except Exception as e:
            logger.warning(
                f"CreatePRWorkflowTool: Failed to clear patches: {e}"
            )
            return False

    def _apply_patches_from_redis(
        self,
        project_id: str,
        conversation_id: str,
        worktree_path: str,
    ) -> Dict[str, Any]:
        """
        Apply stored patches using git apply with fallback strategies.

        Strategies (in order):
        1. git apply --3way --whitespace=fix
        2. git apply --3way --ignore-whitespace
        3. git apply --whitespace=fix
        4. git apply --ignore-whitespace

        Args:
            project_id: Project ID
            conversation_id: Conversation ID
            worktree_path: Path to the git worktree

        Returns:
            Dictionary with:
            - success: bool indicating success
            - files_applied: List of files successfully patched
            - files_failed: List of files that failed to patch
            - error: Error message if failed
        """
        patches = self._get_stored_patches(conversation_id)
        if not patches:
            return {
                "success": True,
                "message": "No stored patches found",
                "files_applied": [],
                "files_failed": [],
            }

        files_applied = []
        files_failed = []

        for file_path, patch in patches.items():
            success = False

            for strategy in [
                ["--3way", "--whitespace=fix"],
                ["--3way", "--ignore-whitespace"],
                ["--whitespace=fix"],
                ["--ignore-whitespace"],
            ]:
                try:
                    import subprocess

                    result = subprocess.run(
                        ["git", "apply", *strategy],
                        input=patch,
                        cwd=worktree_path,
                        capture_output=True,
                        text=True,
                    )

                    if result.returncode == 0:
                        logger.info(
                            f"CreatePRWorkflowTool: Applied patch for {file_path} "
                            f"with strategy {strategy}"
                        )
                        files_applied.append(file_path)
                        success = True
                        break
                    else:
                        logger.debug(
                            f"CreatePRWorkflowTool: Strategy {strategy} failed for {file_path}: "
                            f"{result.stderr}"
                        )

                except Exception as e:
                    logger.debug(
                        f"CreatePRWorkflowTool: Error applying patch for {file_path} "
                        f"with strategy {strategy}: {e}"
                    )

            if not success:
                logger.error(
                    f"CreatePRWorkflowTool: All strategies failed for {file_path}"
                )
                files_failed.append(file_path)

        return {
            "success": len(files_failed) == 0,
            "files_applied": files_applied,
            "files_failed": files_failed,
            "total_patches": len(patches),
        }

    def commit_file_and_extract_patch(
        self,
        project_id: str,
        conversation_id: str,
        file_path: str,
        commit_message: str,
        branch_name: str,
        base_branch: str = "main",
    ) -> Dict[str, Any]:
        """
        Commit a single file and extract its patch.

        This is used by the agent flow to commit files incrementally
        and store patches for later PR creation.

        Args:
            project_id: Project ID
            conversation_id: Conversation ID
            file_path: Path to the file to commit
            commit_message: Commit message
            branch_name: Branch name to create/commit to
            base_branch: Base branch to create the new branch from

        Returns:
            Dictionary with:
            - success: bool
            - commit_hash: The commit hash
            - patch: The extracted unified diff patch
            - error: Error message if failed
        """
        try:
            # Get project details
            details = self._get_project_details(project_id)
            repo_name = details["project_name"]
            user_id = details.get("user_id")

            # Get worktree path
            worktree_path, failure_reason = self._get_worktree_path(
                project_id, conversation_id, user_id
            )
            if not worktree_path:
                error_msg = "Worktree not found."
                if failure_reason:
                    reason_hint = {
                        "no_ref": "missing branch or commit_id",
                        "auth_failed": "all auth methods failed",
                        "clone_failed": "git clone failed",
                        "worktree_add_failed": "git worktree add failed",
                    }.get(failure_reason, failure_reason)
                    error_msg += f" Reason: {reason_hint}."
                return {"success": False, "error": error_msg}

            # Create branch, commit, and push
            git_result = self._create_branch_commit_and_push(
                worktree_path=worktree_path,
                branch_name=branch_name,
                commit_message=commit_message,
                files_to_stage=[file_path],
                repo_name=repo_name,
            )

            if not git_result["success"]:
                return {
                    "success": False,
                    "error": git_result.get("error", "Git operation failed"),
                }

            commit_hash = git_result.get("commit_hash")

            # Extract patch from the commit
            patch = self._extract_patch_from_commit(
                worktree_path, file_path, commit_hash
            )

            if patch:
                # Store patch in Redis
                self._store_patch_in_redis(conversation_id, file_path, patch)

            return {
                "success": True,
                "commit_hash": commit_hash,
                "patch": patch,
                "branch": branch_name,
                "file_path": file_path,
            }

        except Exception as e:
            logger.exception("CreatePRWorkflowTool: Error in commit_file_and_extract_patch")
            return {"success": False, "error": str(e)}

    def _create_pr(
        self,
        repo_name: str,
        head_branch: str,
        base_branch: str,
        title: str,
        body: str,
    ) -> Dict[str, Any]:
        """Step 5: Create pull request via code provider."""
        try:
            provider = CodeProviderFactory.create_provider_with_fallback(repo_name)
            g = provider.client

            # Normalize repo name for API calls
            from app.modules.parsing.utils.repo_name_normalizer import (
                normalize_repo_name,
                get_actual_repo_name_for_lookup,
            )

            provider_type = os.getenv("CODE_PROVIDER", "github").lower()
            normalized_input = normalize_repo_name(repo_name, provider_type)
            actual_repo_name = get_actual_repo_name_for_lookup(
                normalized_input, provider_type
            )

            repo = g.get_repo(actual_repo_name)

            # Create the pull request
            if provider_type == "gitbucket":
                import json

                post_parameters = {
                    "title": title,
                    "body": body,
                    "head": head_branch,
                    "base": base_branch,
                }
                try:
                    headers, data = repo._requester.requestJsonAndCheck(
                        "POST", f"{repo.url}/pulls", input=post_parameters
                    )
                    if isinstance(data, str):
                        data = json.loads(data)
                    return {
                        "success": True,
                        "pr_number": data.get("number"),
                        "pr_url": data.get("html_url"),
                    }
                except Exception as gitbucket_err:
                    err_str = str(gitbucket_err).lower()
                    if "already exists" in err_str or "duplicate" in err_str:
                        # Fall through to find existing PR below
                        pass
                    else:
                        raise

                # GitBucket: find the existing open PR
                try:
                    headers, prs = repo._requester.requestJsonAndCheck(
                        "GET",
                        f"{repo.url}/pulls",
                        parameters={"state": "open", "head": head_branch, "base": base_branch},
                    )
                    if isinstance(prs, str):
                        prs = json.loads(prs)
                    if prs:
                        return {
                            "success": True,
                            "pr_number": prs[0].get("number"),
                            "pr_url": prs[0].get("html_url"),
                            "already_existed": True,
                        }
                except Exception:
                    pass
                return {"success": False, "error": "PR already exists but could not retrieve it", "step": "create_pr"}

            else:
                # Standard GitHub
                try:
                    pr = repo.create_pull(
                        title=title, body=body, head=head_branch, base=base_branch
                    )
                    return {
                        "success": True,
                        "pr_number": pr.number,
                        "pr_url": pr.html_url,
                    }
                except Exception as gh_err:
                    err_str = str(gh_err).lower()
                    status = getattr(gh_err, "status", None)
                    if status == 422 and "already exists" in err_str:
                        # PR already exists — find and return it
                        logger.info(
                            f"CreatePRWorkflowTool: PR already exists for {head_branch}, fetching it"
                        )
                        owner = actual_repo_name.split("/")[0]
                        for state in ("open", "closed"):
                            prs = repo.get_pulls(
                                state=state,
                                head=f"{owner}:{head_branch}",
                                base=base_branch,
                            )
                            for existing_pr in prs:
                                logger.info(
                                    f"CreatePRWorkflowTool: Found existing PR #{existing_pr.number}"
                                )
                                return {
                                    "success": True,
                                    "pr_number": existing_pr.number,
                                    "pr_url": existing_pr.html_url,
                                    "already_existed": True,
                                }
                        return {
                            "success": False,
                            "error": "PR already exists but could not be retrieved",
                            "step": "create_pr",
                        }
                    raise

        except Exception as e:
            logger.exception("CreatePRWorkflowTool: Error creating PR")
            return {"success": False, "error": str(e), "step": "create_pr"}

    def _run(
        self,
        project_id: str,
        conversation_id: str,
        branch_name: str,
        commit_message: str,
        pr_title: str,
        pr_body: str = "",
        base_branch: str = "main",
    ) -> Dict[str, Any]:
        """Execute the full PR workflow."""
        try:
            # Validate repo manager
            if not self.repo_manager:
                return {
                    "success": False,
                    "error": "Repo manager not enabled",
                    "step": "init",
                }

            # Get project details
            details = self._get_project_details(project_id)
            repo_name = details["project_name"]
            user_id = details.get("user_id")

            # Get edits worktree path for this conversation
            worktree_path, failure_reason = self._get_worktree_path(
                project_id, conversation_id, user_id
            )
            if not worktree_path:
                error_msg = "Worktree not found."
                if failure_reason:
                    reason_hint = {
                        "no_ref": "missing branch or commit_id",
                        "auth_failed": "all auth methods failed (GitHub App, OAuth, env token)",
                        "clone_failed": "git clone failed",
                        "worktree_add_failed": "git worktree add failed",
                    }.get(failure_reason, failure_reason)
                    error_msg += f" Reason: {reason_hint}."
                return {
                    "success": False,
                    "error": error_msg,
                    "step": "init",
                }

            # Step 1: Apply changes (try patches first, fall back to CodeChangesManager)
            logger.info("CreatePRWorkflowTool: Step 1 - Applying changes")

            # First, try to apply stored patches (from new agent flow)
            patch_result = self._apply_patches_from_redis(
                project_id, conversation_id, worktree_path
            )

            if patch_result.get("files_applied"):
                # Patches were applied successfully
                logger.info(
                    f"CreatePRWorkflowTool: Applied {len(patch_result['files_applied'])} patches from Redis"
                )
                files_to_stage = (
                    patch_result.get("files_applied", [])
                    + patch_result.get("files_failed", [])
                )

                # If some patches failed, fall back to CodeChangesManager for those
                if patch_result.get("files_failed"):
                    logger.warning(
                        f"CreatePRWorkflowTool: {len(patch_result['files_failed'])} patches failed, "
                        f"falling back to CodeChangesManager"
                    )
                    apply_result = self._apply_changes(
                        project_id, conversation_id, worktree_path
                    )
                    if apply_result["success"]:
                        # Add files from fallback that weren't already applied
                        fallback_files = (
                            apply_result.get("files_applied", [])
                            + apply_result.get("files_deleted", [])
                        )
                        files_to_stage = list(set(files_to_stage + fallback_files))

                # Clear stored patches after applying
                self._clear_stored_patches(conversation_id)

            else:
                # No patches found, use traditional CodeChangesManager flow
                logger.info(
                    "CreatePRWorkflowTool: No patches found, using CodeChangesManager"
                )
                apply_result = self._apply_changes(
                    project_id, conversation_id, worktree_path
                )
                if not apply_result["success"]:
                    return {
                        "success": False,
                        "error": apply_result.get("error", "Failed to apply changes"),
                        "step": "apply_changes",
                    }

                files_to_stage = (
                    apply_result.get("files_applied", [])
                    + apply_result.get("files_deleted", [])
                )

            if not files_to_stage:
                return {
                    "success": False,
                    "error": "No files to commit",
                    "step": "apply_changes",
                }

            # Steps 2, 3, 4: Create branch, commit, and push
            logger.info("CreatePRWorkflowTool: Steps 2-4 - Branch, commit, push")
            git_result = self._create_branch_commit_and_push(
                worktree_path, branch_name, commit_message, files_to_stage, repo_name
            )
            if not git_result["success"]:
                return {
                    "success": False,
                    "error": git_result.get("error", "Git operation failed"),
                    "step": git_result.get("step", "git_operations"),
                    "files_applied": files_to_stage,
                }

            # Step 5: Create PR
            logger.info("CreatePRWorkflowTool: Step 5 - Creating PR")
            pr_result = self._create_pr(
                repo_name, branch_name, base_branch, pr_title, pr_body
            )
            if not pr_result["success"]:
                return {
                    "success": False,
                    "error": pr_result.get("error", "Failed to create PR"),
                    "step": "create_pr",
                    "branch": branch_name,
                    "commit_hash": git_result.get("commit_hash"),
                    "files_committed": git_result.get("files_committed"),
                }

            # Success!
            return {
                "success": True,
                "branch": branch_name,
                "commit_hash": git_result.get("commit_hash"),
                "pr_number": pr_result.get("pr_number"),
                "pr_url": pr_result.get("pr_url"),
                "files_committed": git_result.get("files_committed"),
                "message": f"Successfully created PR #{pr_result.get('pr_number')}: {pr_result.get('pr_url')}",
            }

        except GitOperationError as e:
            logger.error(f"CreatePRWorkflowTool: Git operation error: {e}")
            return {
                "success": False,
                "error": f"Git operation failed: {str(e)}",
                "step": "git_operations",
            }
        except ValueError as e:
            logger.error(f"CreatePRWorkflowTool: Value error: {e}")
            return {"success": False, "error": str(e), "step": "validation"}
        except Exception as e:
            logger.exception("CreatePRWorkflowTool: Unexpected error")
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

    async def _arun(
        self,
        project_id: str,
        conversation_id: str,
        branch_name: str,
        commit_message: str,
        pr_title: str,
        pr_body: str = "",
        base_branch: str = "main",
    ) -> Dict[str, Any]:
        """Async wrapper for _run."""
        import asyncio

        return await asyncio.to_thread(
            self._run,
            project_id,
            conversation_id,
            branch_name,
            commit_message,
            pr_title,
            pr_body,
            base_branch,
        )


def create_pr_workflow_tool(
    sql_db: Session, user_id: str
) -> Optional[StructuredTool]:
    """
    Create the composite PR workflow tool if repo manager is enabled.

    Returns None if repo manager is not enabled.
    """
    repo_manager_enabled = (
        os.getenv("REPO_MANAGER_ENABLED", "false").lower() == "true"
    )
    if not repo_manager_enabled:
        logger.debug(
            "CreatePRWorkflowTool not created: REPO_MANAGER_ENABLED is false"
        )
        return None

    tool_instance = CreatePRWorkflowTool(sql_db, user_id)
    if not tool_instance.repo_manager:
        logger.debug(
            "CreatePRWorkflowTool not created: RepoManager initialization failed"
        )
        return None

    return StructuredTool.from_function(
        coroutine=tool_instance._arun,
        func=tool_instance._run,
        name="create_pr_workflow",
        description=tool_instance.description,
        args_schema=CreatePRWorkflowInput,
    )
